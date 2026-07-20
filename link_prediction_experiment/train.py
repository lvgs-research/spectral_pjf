"""Training + scoring via the leak-safe time-chunked PIT mechanism.
Supervision is time-sorted into chunks; chunk c message-passes over edges with
time < min(anchor_ts in c), so no example sees an edge at/after its own time."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .config import ExperimentConfig
from .features import FeatureBundle
from .graph_build import (GraphStore, configured_edge_types, INTERACTION_RELS,
                          enabled_attribute_rels)
from .models import GraphMeta, build_model
from .match import match_tables, compute_tab_match


def get_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(spec)


def _chunks(ts_sorted: np.ndarray, n_chunks: int) -> List[Tuple[np.ndarray, int]]:
    """Contiguous chronological chunks of pre-sorted positions; cutoff = chunk min ts."""
    n = len(ts_sorted)
    groups = np.array_split(np.arange(n), min(n_chunks, max(n, 1)))
    out = []
    for g in groups:
        if len(g) == 0:
            continue
        out.append((g, int(ts_sorted[g[0]])))   # min ts in chunk (sorted)
    return out


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Stable log(1 - exp(x)) for x <= 0 (Mächler 2012, two-branch split at -ln2)."""
    x = torch.clamp(x, max=-1e-12)              # guard fp noise -> stay in valid domain
    return torch.where(
        x > -0.6931471805599453,
        torch.log(-torch.expm1(x)),             # |x| small
        torch.log1p(-torch.exp(x)),             # x very negative
    )


class Trainer:
    def __init__(self, cfg: ExperimentConfig, gstore: GraphStore, feats: FeatureBundle,
                 df: pl.DataFrame, *, verbose: bool = True):
        self.cfg = cfg
        self.verbose = verbose
        self.device = get_device(cfg.train.device)
        if cfg.model.kind == "mlp":
            self.mode = "mlp"
        else:
            self.mode = cfg.model.graph_mode
        self.fcfg = cfg.feature
        # CPU SpMM fast-path for mean-SAGE hetero convs (dup-free graph -> SpMM-mean
        # == scatter-mean); CPU-only since PyG's spmm is unsupported on MPS sparse.
        self.use_spmm = (self.device.type == "cpu" and self.mode == "hetero"
                         and cfg.model.kind in ("sage", "sage_oversmooth")
                         and getattr(cfg.model, "conv_type", "sage") == "sage"   # gatv2/resgated need edge_index
                         and getattr(cfg.model, "hetero_cross_agg", "sum") == "sum")  # concat = edge_index path

        self.gstore = gstore.to(self.device)
        self.feats = feats.to(self.device)
        self.df = df
        self.rng = np.random.default_rng(cfg.train.seed)
        torch.manual_seed(cfg.train.seed)

        self.meta = GraphMeta(
            num_nodes=gstore.num_nodes,
            edge_types=configured_edge_types(self.fcfg),
            node_offset=gstore.node_offset,
            n_total=gstore.n_total,
        )
        self.model = build_model(cfg, feats.in_dims, self.meta).to(self.device)
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        # default objective (fit() re-sets it); makes score_pairs safe pre-fit
        self.objective = getattr(cfg.train, "objective", "esmm")
        self.ylab_accept = None
        self.posthoc_mode = getattr(cfg.train, "posthoc_calibration", "none")
        self._cal = None        # fitted post-hoc per-head calibrators; None => uncalibrated
        self.use_match = getattr(cfg.feature, "use_match_feats", False)  # explicit match feats
        self.match_canon = getattr(cfg.feature, "match_skill_canon", False)  # skills-vocab canon
        self.match_extra = []
        self.pit_degree_nodes = getattr(cfg.feature, "pit_degree_nodes", False)  # PIT degree NODE feats
        self.pit_attr_degree_nodes = getattr(cfg.feature, "pit_attr_degree_nodes", False)  # PIT ATTR degree on focal nodes
        # enabled attribute rels, ordered -> per-type attribute-degree columns
        self._attr_deg_rels = enabled_attribute_rels(cfg.feature) if self.pit_attr_degree_nodes else []
        self.use_company_degree = getattr(cfg.feature, "use_company_degree_feats", False)  # PIT company out-deg NODE feat
        self._un_m = self._un_j = None   # cached L2-normalised seeker/job embeddings
        self._mt = self._jt = None       # cached seeker/job tabular match-attr tables

    # -- edges visible at a cutoff (focal entities retain their profile) --
    def _edges(self, cutoff: int, focal_m=None, focal_j=None):
        if self.mode == "mlp":
            return None
        eid = self.gstore.hetero_pit(cutoff, self.fcfg, focal_m, focal_j)
        return self._to_spmm(eid) if self.use_spmm else eid

    def _to_spmm(self, eid):
        """edge_index_dict -> sparse adjacency dict (adj_t[dst,src]) for SpMM."""
        nn = self.gstore.num_nodes
        out = {}
        for rel, e in eid.items():
            s, _, d = rel
            ones = torch.ones(e.shape[1], device=self.device)
            out[rel] = torch.sparse_coo_tensor(
                torch.stack([e[1], e[0]]), ones, (nn[d], nn[s])).coalesce()
        return out

    def _match_feats(self, m_np, j_np):
        """Explicit person-job match features: 4 text-emb + tabular match features."""
        if self._un_m is None:                         # cache normalised embeddings once
            # use the RAW text emb (preserved by content_off) so match survives content_off
            src = self.feats.text if self.feats.text is not None else self.feats.x
            self._un_m = F.normalize(src["seeker"], p=2, dim=-1)
            self._un_j = F.normalize(src["job"], p=2, dim=-1)
        m_t = torch.as_tensor(m_np, dtype=torch.long, device=self.device)
        j_t = torch.as_tensor(j_np, dtype=torch.long, device=self.device)
        had = self._un_m[m_t] * self._un_j[j_t]        # [N,768] per-dim agreement
        text = torch.cat([had.sum(-1, keepdim=True),                  # cosine (unit vecs)
                          had.std(-1, keepdim=True) * 10.0,
                          had.max(-1, keepdim=True).values * 10.0,
                          had.min(-1, keepdim=True).values * 10.0], dim=-1)   # [N,4]
        if self._mt is None:                           # build seeker/job attr tables once
            self._mt, self._jt = match_tables(self.match_canon)
        tab = torch.as_tensor(compute_tab_match(self._mt, self._jt, np.asarray(m_np), np.asarray(j_np)),
                              device=self.device)       # [N, MATCH_TAB_DIM]
        return torch.cat([text, tab], dim=-1)

    def _pair_feats(self, m_np, j_np, ts_np):
        """Per-pair decoder features: the match features when enabled (None otherwise)."""
        parts = []
        if self.use_match:
            parts.append(self._match_feats(m_np, j_np))
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def _node_pit_degree(self, edges):
        """Per-relation PIT interaction degree: [log1p(in), log1p(out)] per interaction
        relation for each node type. Column order must match build_model's in_dims."""
        nn = self.gstore.num_nodes
        cols = {t: [] for t in nn}
        for rel in INTERACTION_RELS:                   # fixed order (= build_model)
            s_t, _, d_t = rel
            indeg = {t: torch.zeros(nn[t], device=self.device) for t in nn}
            outdeg = {t: torch.zeros(nn[t], device=self.device) for t in nn}
            a = edges.get(rel)
            if a is not None:
                if a.is_sparse:                        # SpMM adj [n_dst, n_src]
                    outdeg[s_t] = outdeg[s_t] + torch.sparse.sum(a, dim=0).to_dense()
                    indeg[d_t] = indeg[d_t] + torch.sparse.sum(a, dim=1).to_dense()
                else:                                  # edge_index [2, E] = [src, dst]
                    ones = torch.ones(a.shape[1], device=self.device)
                    outdeg[s_t].index_add_(0, a[0], ones)
                    indeg[d_t].index_add_(0, a[1], ones)
            for t in nn:                               # per rel: [in, out] for every type
                cols[t].append(torch.log1p(indeg[t]))
                cols[t].append(torch.log1p(outdeg[t]))
        return {t: torch.stack(cols[t], dim=1) for t in nn}   # [n, 2*len(INTERACTION_RELS)]

    def _node_pit_attr_degree(self, edges):
        """Focal-node (seeker/job) per-attribute-type out-degree from the PIT graph:
        one log1p(out-degree) column per enabled attribute relation, leak-safe."""
        nn = self.gstore.num_nodes
        cols = {"seeker": [], "job": []}
        for rel in self._attr_deg_rels:                # enabled attr rels, fixed order
            s_t = rel[0]
            if s_t not in cols:                        # all attr rels are seeker/job-sourced
                continue
            a = edges.get(rel)
            d = torch.zeros(nn[s_t], device=self.device)
            if a is not None:
                if a.is_sparse:                        # SpMM adj [n_dst, n_src] -> per-src out
                    d = d + torch.sparse.sum(a, dim=0).to_dense()
                else:                                  # edge_index [2, E]=[src,dst] -> src out-deg
                    d.index_add_(0, a[0], torch.ones(a.shape[1], device=self.device))
            cols[s_t].append(torch.log1p(d))
        return {t: torch.stack(cols[t], dim=1) for t in ("seeker", "job") if cols[t]}

    def _node_pit_company_degree(self, edges):
        """Company-node PIT out-degree = log1p(# jobs posted), from posted_by edges. Returns [nC, 1]."""
        nC = self.gstore.num_nodes["company"]
        out = torch.zeros(nC, device=self.device)
        a = edges.get(("job", "posted_by", "company"))
        if a is not None:
            if a.is_sparse:                                # SpMM adj [n_dst=company, n_src=job]
                out = out + torch.sparse.sum(a, dim=1).to_dense()   # per-company (dst) count
            else:                                          # edge_index [2, E] = [job, company]
                out.index_add_(0, a[1], torch.ones(a.shape[1], device=self.device))
        return torch.log1p(out).unsqueeze(1)               # [nC, 1]

    def _aug_feats(self, edges):
        """Node features with PIT degree appended per node when enabled. Append order
        (interaction -> attribute -> company) must match build_model's in_dims."""
        x = self.feats.x
        if self.pit_degree_nodes:
            deg = self._node_pit_degree(edges)
            x = {t: torch.cat([x[t], deg[t]], dim=1) for t in x}
        if self.pit_attr_degree_nodes:
            adeg = self._node_pit_attr_degree(edges)          # seeker/job only
            x = {t: (torch.cat([x[t], adeg[t]], dim=1) if t in adeg else x[t]) for t in x}
        if self.use_company_degree:                           # company node only
            cdeg = self._node_pit_company_degree(edges)
            x = {t: (torch.cat([x[t], cdeg], dim=1) if t == "company" else x[t]) for t in x}
        return x

    def _encode(self, edges):
        """Run the encoder once -> node embeddings (reused for all pairs)."""
        if self.mode == "mlp":
            return self.model.encode(self.feats.x)
        if self.mode == "temporal":
            eid, ew = edges
            return self.model.encode(self.feats.x, eid, edge_weight_dict=ew)
        if self.mode == "homo":                             # (homo MP edges, hetero degree dict)
            homo_ei, deg = edges if isinstance(edges, tuple) else (edges, edges)
            return self.model.encode(self._aug_feats(deg), homo_ei)
        return self.model.encode(self._aug_feats(edges), edges)

    # ---- ESMM funnel-aware two-head paths ----
    def _forward_heads(self, edges, m_idx, j_idx, pf=None):
        """Encode once + decode the two ESMM heads -> (accept_logit, cond_logit)."""
        z = self._encode(edges)
        if pf is None:                                 # graph/plain-MLP: keep 3-arg signature
            return self.model.decode_pairs_heads(z, m_idx, j_idx)
        return self.model.decode_pairs_heads(z, m_idx, j_idx, pf)

    def _decode_heads(self, z, m_idx, j_idx, pf=None):
        if pf is None:
            return self.model.decode_pairs_heads(z, m_idx, j_idx)
        return self.model.decode_pairs_heads(z, m_idx, j_idx, pf)

    def _score_heads(self, bm, bj, bts, cutoff):
        edges = self._edges(cutoff, np.unique(bm), np.unique(bj))
        bm_t = torch.as_tensor(bm, dtype=torch.long, device=self.device)
        bj_t = torch.as_tensor(bj, dtype=torch.long, device=self.device)
        pf = self._pair_feats(bm, bj, bts)
        return self._forward_heads(edges, bm_t, bj_t, pf)

    # ---- ESMM head probs and the served/ranked joint score ----
    def _esmm_probs(self, a, c):
        """(accept_logit, cond_logit) -> (pa, pc) numpy per-head probs via sigmoid."""
        an = a.float().cpu().numpy(); cn = c.float().cpu().numpy()
        sig = lambda z: 1.0 / (1.0 + np.exp(-z))
        return sig(an), sig(cn)

    def _esmm_joint(self, a, c) -> np.ndarray:
        """Served/ranked ESMM score: entire-space P(pass)=sigma(a)*sigma(c), in log-space."""
        return torch.exp(F.logsigmoid(a) + F.logsigmoid(c)).float().cpu().numpy()

    def fit_posthoc_calibration(self):
        """Post-hoc calibration disabled (no-op kept for the eval call sequence)."""
        self._cal = None
        return

    # ---- per-chunk training loss (dispatch on objective) ----
    @property
    def _esmm_arch(self) -> bool:
        """True when the model uses the ESMM two-head decoder (scoring/eval routing)."""
        return self.objective in ("esmm",)

    def _chunk_loss(self, grp, cutoff, m, j, ts, ylab, ylab_accept=None):
        # funnel: ESMM entire-space multi-task (accept BCE + entire-space CTCVR product)
        acc = ylab_accept if ylab_accept is not None else self.ylab_accept
        bm, bj, bts = m[grp], j[grp], ts[grp]
        a, c = self._score_heads(bm, bj, bts, cutoff)  # accept_logit, p(pass|accept)_logit
        ya = torch.as_tensor(acc[grp], device=self.device)              # accept label
        yp = torch.as_tensor(ylab[grp], device=self.device)            # pass label
        l_accept = F.binary_cross_entropy_with_logits(a, ya)
        log_p = F.logsigmoid(a) + F.logsigmoid(c)      # log(sigmoid(a)*sigmoid(c)) = log p_pass
        l_ctcvr = -(yp * log_p + (1.0 - yp) * _log1mexp(log_p)).mean()
        return l_accept + l_ctcvr

    # ---- training (early stopping + best/last checkpointing + resume) ----
    def fit(self, ckpt_path=None):
        tc = self.cfg.train
        self.objective = getattr(tc, "objective", "esmm")
        train = self.df.filter(pl.col("dataset_split") == "train")
        # ESMM entire-space multi-task: anchor on all real exposed pairs (both labels real)
        anchor = train
        ylab = anchor["passed"].to_numpy().astype(np.float32)
        ylab_accept = anchor["accept_event_ts"].is_not_null().to_numpy().astype(np.float32)
        m = anchor["seeker_idx"].to_numpy()
        j = anchor["job_idx"].to_numpy()
        ts = anchor["exposure_ts"].to_numpy()
        if tc.max_train_pos and len(m) > tc.max_train_pos:
            sel = self.rng.choice(len(m), tc.max_train_pos, replace=False)
            m, j, ts = m[sel], j[sel], ts[sel]
            ylab = ylab[sel] if ylab is not None else None
            ylab_accept = ylab_accept[sel] if ylab_accept is not None else None
        order = np.argsort(ts, kind="stable")
        m, j, ts = m[order], j[order], ts[order]
        ylab = ylab[order] if ylab is not None else None
        ylab_accept = ylab_accept[order] if ylab_accept is not None else None
        self.ylab_accept = ylab_accept
        chunks = _chunks(ts, tc.n_time_chunks_train)

        val_set = self._build_val_select()       # fixed across epochs -> comparable
        # selection direction: AUC is higher-better, val_loss is lower-better
        self.higher_better = (tc.select_metric != "val_loss")
        best_metric, best_state, bad, start_ep = (-np.inf if self.higher_better else np.inf), None, 0, 0
        self.best_epoch, self.best_val = 0, float("nan")
        self.val_history: List[Tuple[int, float]] = []   # (epoch, val_metric) per epoch

        # resume an interrupted run of the SAME experiment (continue training)
        if ckpt_path is not None and ckpt_path.exists():
            st = torch.load(ckpt_path, weights_only=False)
            if not st.get("done"):
                self.model.load_state_dict(st["last_state"])
                best_state = st["best_state"]; best_metric = st["best_metric"]
                self.best_epoch, self.best_val = st["best_epoch"], st["best_val"]
                bad, start_ep = st["bad"], st["epoch"] + 1
                self.val_history = list(st.get("val_history") or [])
                try:
                    self.rng.bit_generator.state = st["np_rng"]
                    torch.set_rng_state(st["torch_rng"])
                except Exception:
                    pass
                if self.verbose:
                    print(f"   resume: continue from epoch {start_ep+1} "
                          f"(best {best_metric:.4f} @ {self.best_epoch})")

        for ep in range(start_ep, tc.epochs):
            self.model.train()
            chunk_order = self.rng.permutation(len(chunks))
            tot, nb = 0.0, 0
            for ci in chunk_order:
                grp, cutoff = chunks[ci]
                loss = self._chunk_loss(grp, cutoff, m, j, ts, ylab)
                if loss is None:                  # e.g. chunk with no valid pairs
                    continue
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                tot += float(loss.item()); nb += 1

            val_metric = self._val_metric(val_set)
            self.val_history.append((ep + 1, float(val_metric)))
            improved = ((val_metric > best_metric + tc.min_delta) if self.higher_better
                        else (val_metric < best_metric - tc.min_delta))
            if improved:
                best_metric, best_state, bad = val_metric, self._cpu_state(), 0
                self.best_epoch, self.best_val = ep + 1, val_metric
            else:
                bad += 1
            # save best + last-epoch weights (resume point)
            self._save_train_ckpt(ckpt_path, ep, best_state, best_metric, bad, done=False)
            if self.verbose:
                print(f"   epoch {ep+1}/{tc.epochs}  loss={tot/max(nb,1):.4f}  "
                      f"val[{tc.select_metric}]={val_metric:.4f}{'  *' if improved else ''}")
            if bad >= tc.patience:
                if self.verbose:
                    print(f"   early stop @ epoch {ep+1} (best {best_metric:.4f} @ {self.best_epoch})")
                break

        self._final_best_state = best_state if best_state is not None else self._cpu_state()
        self.model.load_state_dict(self._final_best_state)   # use best for eval
        return self

    def _cpu_state(self):
        return {kk: v.detach().cpu().clone() for kk, v in self.model.state_dict().items()}

    def _save_train_ckpt(self, ckpt_path, ep, best_state, best_metric, bad, *, done, metrics=None):
        """Persist best + last-epoch weights + training state (single file, overwritten)."""
        if ckpt_path is None:
            return
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": ep, "best_state": best_state, "last_state": self._cpu_state(),
            "best_metric": best_metric, "best_epoch": self.best_epoch, "best_val": self.best_val,
            "bad": bad, "np_rng": self.rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(), "done": done, "per_seed_metrics": metrics,
            "val_history": getattr(self, "val_history", []),
            "config": self.cfg.to_dict(), "in_dims": self.feats.in_dims, "name": self.cfg.name,
        }, ckpt_path)

    def save_final(self, ckpt_path, metrics):
        """Mark the (config,seed) done; keep last-epoch weights, store best + metrics."""
        if ckpt_path is None:
            return
        st = torch.load(ckpt_path, weights_only=False) if ckpt_path.exists() else {"last_state": self._cpu_state()}
        st["best_state"] = getattr(self, "_final_best_state", self._cpu_state())  # model==best now
        st["done"] = True
        st["per_seed_metrics"] = metrics
        st["name"] = self.cfg.name
        st["posthoc_cal"] = getattr(self, "_cal", None)   # val-fit sklearn calibrators (pickle)
        torch.save(st, ckpt_path)

    def _build_val_select(self):
        """Fixed validation set (exposed AUC) for per-epoch model selection (built once)."""
        tc = self.cfg.train
        vdf = self.df.filter(pl.col("dataset_split") == "val")
        # auc_exposed_all: exposed-AUC on real val pairs
        sub = vdf
        cap = (tc.val_select_max_pos or 0) * 3
        if cap and sub.height > cap:
            sub = sub.sample(cap, seed=tc.seed)
        return ("exposed", sub["seeker_idx"].to_numpy(), sub["job_idx"].to_numpy(),
                sub["exposure_ts"].to_numpy(), sub["passed"].to_numpy().astype(float))

    def _val_metric(self, val_set) -> float:
        _, mm, jj, tt, yy = val_set
        if len(np.unique(yy)) < 2:
            return float("nan")
        s = self.score_pairs(mm, jj, tt, self.cfg.train.n_time_chunks_eval)
        return float(roc_auc_score(yy, s))

    # ---- scoring (eval) ----
    @torch.no_grad()
    def score_pairs(self, m_idx: np.ndarray, j_idx: np.ndarray, ts: np.ndarray,
                    n_chunks: int, return_heads: bool = False):
        """Chunked-PIT scoring. return_heads=True (ESMM only) returns per-head probs
        (pa, pc) instead of the served joint pa*pc."""
        self.model.eval()
        if return_heads and not self._esmm_arch:
            raise AssertionError("return_heads requires the ESMM two-head decoder")
        out = np.empty(len(m_idx), dtype=np.float32)
        outC = np.empty(len(m_idx), dtype=np.float32) if return_heads else None   # out=pa, outC=pc
        if self.mode == "mlp":
            bm = torch.as_tensor(m_idx, dtype=torch.long, device=self.device)
            bj = torch.as_tensor(j_idx, dtype=torch.long, device=self.device)
            if self._esmm_arch:                   # scored quantity = head1*head2 = pass prob
                a, c = self._forward_heads(None, bm, bj, self._pair_feats(m_idx, j_idx, ts))
                if return_heads:
                    pa, pc = self._esmm_probs(a, c)
                    return pa.astype(np.float32), pc.astype(np.float32)
                return self._esmm_joint(a, c)
        order = np.argsort(ts, kind="stable")
        inv = np.empty_like(order); inv[order] = np.arange(len(order))
        ms, js, tss = m_idx[order], j_idx[order], ts[order]
        for grp, cutoff in _chunks(tss, n_chunks):
            edges = self._edges(cutoff, np.unique(ms[grp]), np.unique(js[grp]))
            bm = torch.as_tensor(ms[grp], dtype=torch.long, device=self.device)
            bj = torch.as_tensor(js[grp], dtype=torch.long, device=self.device)
            pf = self._pair_feats(ms[grp], js[grp], tss[grp])   # per-pair cf/match feats
            if self._esmm_arch:
                a, c = self._forward_heads(edges, bm, bj, pf)
                if return_heads:
                    pa, pc = self._esmm_probs(a, c)
                    out[grp] = pa; outC[grp] = pc
                else:
                    out[grp] = self._esmm_joint(a, c)
        if return_heads:
            return out[inv], outC[inv]
        return out[inv]

    @torch.no_grad()
    def score_pairs_exact(self, m_idx: np.ndarray, j_idx: np.ndarray, ts: np.ndarray,
                          return_heads: bool = False):
        """Leak-safe per-row PIT scoring at daily left-edge resolution: cutoff snapped
        to day start (<= anchor), so no edge at/after the anchor is visible. The day's
        graph is encoded once and reused. return_heads=True (ESMM only) returns (pa, pc)."""
        self.model.eval()
        if return_heads and not self._esmm_arch:
            raise AssertionError("return_heads requires the ESMM two-head decoder")
        if self.mode == "mlp":
            bm = torch.as_tensor(m_idx, dtype=torch.long, device=self.device)
            bj = torch.as_tensor(j_idx, dtype=torch.long, device=self.device)
            if self._esmm_arch:
                ah, ch = self._forward_heads(None, bm, bj, self._pair_feats(m_idx, j_idx, ts))
                if return_heads:
                    pa, pc = self._esmm_probs(ah, ch)
                    return pa.astype(np.float32), pc.astype(np.float32)
                return self._esmm_joint(ah, ch)
        ts = np.asarray(ts, dtype=np.int64)
        day = (ts // 86400) * 86400               # left-edge cutoff
        order = np.argsort(day, kind="stable")
        ms, js, ds, tsa = m_idx[order], j_idx[order], day[order], ts[order]
        res = np.empty(len(ms), dtype=np.float32)
        resC = np.empty(len(ms), dtype=np.float32) if return_heads else None   # pc; res doubles as pa
        uniq, starts = np.unique(ds, return_index=True)
        bounds = list(starts) + [len(ds)]
        is_mps = self.device.type == "mps"
        for gi in range(len(uniq)):
            a, b = bounds[gi], bounds[gi + 1]
            cutoff = int(uniq[gi])
            gm, gj = ms[a:b], js[a:b]
            edges = self._edges(cutoff, np.unique(gm), np.unique(gj))
            z = self._encode(edges)                  # encode ONCE
            bm = torch.as_tensor(gm, dtype=torch.long, device=self.device)
            bj = torch.as_tensor(gj, dtype=torch.long, device=self.device)
            pf = self._pair_feats(gm, gj, tsa[a:b])    # cf/match feats at each pair's anchor
            if self._esmm_arch:
                ah, ch = self._decode_heads(z, bm, bj, pf)
                if return_heads:
                    pa, pc = self._esmm_probs(ah, ch)
                    res[a:b] = pa; resC[a:b] = pc
                else:
                    res[a:b] = self._esmm_joint(ah, ch)
            del edges, z, bm, bj
            if is_mps:
                torch.mps.empty_cache()
        outA = np.empty_like(res); outA[order] = res
        if return_heads:
            outC = np.empty_like(resC); outC[order] = resC
            return outA, outC
        return outA
