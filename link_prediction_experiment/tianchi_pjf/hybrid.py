"""Hybrid warm/cold split + incremental visibility + reciprocal ESMM (Tianchi).

Random (non-temporal) cohorts; the SATISFIED label is never a graph edge, and each split's own
supervision outcome edges are held out of its message-passing graph (leak-safe).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

from ..config import ModelConfig, FeatureConfig
from ..models import build_model
from .experiment import (Artifacts, load_artifacts, make_x, make_meta, _CfgShim,
                         ATTR_RELS, _rev, NODE_TYPES, _pf, save_ckpt, CKPT_DIR,
                         _load_qwen_content)

ORDER = {"train": 0, "val": 1, "test": 2}
RESULTS_DIR = Path(__file__).resolve().parent / "_results"
SAT_REL = ("seeker", "satisfied", "job")     # context satisfied outcomes

# hybrid MP interaction relations (browse dropped; satisfied added)
_TC_INT_RELS = [("seeker", "accepts", "job"), SAT_REL]


def _tc_degree_block(ei, num_nodes, fcfg, device):
    """Leak-safe per-split degree node features (in/out for accepts+satisfied, plus per-attr out-degree)."""
    cols = {t: [] for t in NODE_TYPES}
    for rel in _TC_INT_RELS:                                   # fixed order [accepts, satisfied]
        s_t, _, d_t = rel
        indeg = {t: torch.zeros(num_nodes[t], device=device) for t in NODE_TYPES}
        outdeg = {t: torch.zeros(num_nodes[t], device=device) for t in NODE_TYPES}
        a = ei.get(rel)
        if a is not None and a.shape[1] > 0:                   # forward edge_index [2,E]=[src,dst]
            ones = torch.ones(a.shape[1], device=device)
            outdeg[s_t].index_add_(0, a[0], ones)             # src out-degree
            indeg[d_t].index_add_(0, a[1], ones)              # dst in-degree
        for t in NODE_TYPES:
            cols[t].append(torch.log1p(indeg[t])); cols[t].append(torch.log1p(outdeg[t]))
    intdeg = {t: torch.stack(cols[t], dim=1) for t in NODE_TYPES}      # [n, 4]
    attr = [r for r in ATTR_RELS
            if not (r[1] in ("has_skill", "requires_skill") and not fcfg.use_skill_edges)
            and not (r[1] in ("has_title", "requires_title") and not fcfg.use_title_edges)]
    adeg = {"seeker": [], "job": []}
    for rel in attr:                                          # seeker/job SOURCE only, ATTR_RELS order
        s_t = rel[0]
        d = torch.zeros(num_nodes[s_t], device=device)
        a = ei.get(rel)
        if a is not None and a.shape[1] > 0:
            d.index_add_(0, a[0], torch.ones(a.shape[1], device=device))
        adeg[s_t].append(torch.log1p(d))
    out = {}
    for t in NODE_TYPES:
        blk = [intdeg[t]]
        if t in ("seeker", "job") and adeg[t]:
            blk.append(torch.stack(adeg[t], dim=1))           # seeker/job += per-attr out-degree
        out[t] = torch.cat(blk, dim=1)
    return out


def _hybrid_meta(art: Artifacts, fcfg: FeatureConfig):
    """GraphMeta with the hybrid MP relation set (drop considers/browse, add satisfied)."""
    meta = make_meta(art, fcfg)
    ets = [r for r in meta.edge_types if r[1] not in ("considers", "rev_considers")]
    if getattr(fcfg, "use_interaction_edges", True):   # omit satisfied for attribute-only arms
        ets += [SAT_REL, _rev(SAT_REL)]
    meta.edge_types = ets
    return meta


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------
def _role_split(rows, role, sup_frac, rng, ensure_context=False):
    k = len(rows)
    if k == 1:
        role[rows[0]] = "context" if ensure_context else "supervision"
        return
    n_sup = max(1, int(round(sup_frac * k)))
    if ensure_context:
        n_sup = min(n_sup, k - 1)                       # keep >=1 context for warm-train users
    role[rows[:n_sup]] = "supervision"
    role[rows[n_sup:]] = "context"


def assign_hybrid_split(pairs: pl.DataFrame, seed=20260705, f_cold_val=0.15,
                        f_cold_test=0.15, warm_split=(0.6, 0.2, 0.2), sup_frac=0.35):
    """Return the pair frame + columns hsplit, role, cohort (see module docstring)."""
    rng = np.random.default_rng(seed)
    # EXPOSED pairs only (browsed∪delivered); non-exposed would leak a has-an-edge shortcut
    from .build import TIANCHI_MATCH_FEATURES
    keep = ["seeker_idx", "job_idx", "browsed", "delivered", "satisfied"]
    keep += [c for c in TIANCHI_MATCH_FEATURES if c in pairs.columns]   # carry match cols -> _sup pf
    df = (pairs.filter((pl.col("browsed") == 1) | (pl.col("delivered") == 1))
               .select(keep))
    m = df["seeker_idx"].to_numpy()
    br, dv = df["browsed"].to_numpy(), df["delivered"].to_numpy()

    users = np.unique(m)
    perm = rng.permutation(len(users))
    n = len(users); n_ct = int(f_cold_test * n); n_cv = int(f_cold_val * n)
    cohort_of = {}
    for rank, ui in enumerate(perm):
        u = int(users[ui])
        cohort_of[u] = ("cold_test" if rank < n_ct
                        else "cold_val" if rank < n_ct + n_cv else "warm")

    order = np.argsort(m, kind="stable")
    ms = m[order]
    uniq, idx_start = np.unique(ms, return_index=True)
    hsplit = np.empty(len(m), dtype=object)
    role = np.empty(len(m), dtype=object)
    cohort = np.empty(len(m), dtype=object)
    for gi, u in enumerate(uniq):
        s = idx_start[gi]; e = idx_start[gi + 1] if gi + 1 < len(uniq) else len(ms)
        rows = order[s:e].copy()
        rng.shuffle(rows)                               # shuffle a user's pairs
        u = int(u); coh = cohort_of[u]; cohort[rows] = coh
        k = len(rows)
        if coh == "cold_val":
            hsplit[rows] = "val"; _role_split(rows, role, sup_frac, rng)
        elif coh == "cold_test":
            hsplit[rows] = "test"; _role_split(rows, role, sup_frac, rng)
        else:                                           # warm: distribute train/val/test
            n_tr = max(1, int(round(warm_split[0] * k)))
            n_va = int(round(warm_split[1] * k))
            tr, va, te = rows[:n_tr], rows[n_tr:n_tr + n_va], rows[n_tr + n_va:]
            hsplit[tr] = "train"; hsplit[va] = "val"; hsplit[te] = "test"
            _role_split(tr, role, sup_frac, rng, ensure_context=True)
            role[va] = "supervision"; role[te] = "supervision"
    return df.with_columns([pl.Series("hsplit", hsplit), pl.Series("role", role),
                            pl.Series("cohort", cohort)])


def _seeker_present_order(m, hs):
    present = {}
    for u, s in zip(m, hs):
        u = int(u); o = ORDER[s]
        if present.get(u, 99) > o:
            present[u] = o
    return present


# ---------------------------------------------------------------------------
# incremental visibility
# ---------------------------------------------------------------------------
def build_visibility(art: Artifacts, sp: pl.DataFrame, split: str, fcfg: FeatureConfig,
                     device) -> dict:
    """MP edge dict for `split` (see module docstring for the cascade rule)."""
    o = ORDER[split]
    m = sp["seeker_idx"].to_numpy(); j = sp["job_idx"].to_numpy()
    dv = sp["delivered"].to_numpy(); st = sp["satisfied"].to_numpy()
    edge_o = np.array([ORDER[s] for s in sp["hsplit"].to_numpy()])   # each edge's OWN split order
    is_sup = sp["role"].to_numpy() == "supervision"
    # attribute (profile) edges are seeker-level: visible from the seeker's earliest split.
    present_mem = np.full(art.num_nodes["seeker"], 99, dtype=np.int64)
    for u, po in _seeker_present_order(m, sp["hsplit"].to_numpy()).items():
        present_mem[u] = po

    out = {}

    def _add(rel, src, dst):
        ei = torch.tensor(np.stack([src, dst]), dtype=torch.long, device=device)
        out[rel] = ei
        out[_rev(rel)] = ei.flip(0).contiguous()

    if getattr(fcfg, "use_interaction_edges", True):
        # OUTCOME edges gated by each edge's OWN split: later splits see all earlier outcomes;
        # current split's supervision outcomes held out (leak-safe). Browse is exposure-only, not MP.
        keep = (edge_o <= o) & ~((edge_o == o) & is_sup)       # +earlier (all), −current-supervision
        a_mask = (dv == 1) & keep
        s_mask = (st == 1) & keep
        _add(("seeker", "accepts", "job"), m[a_mask], j[a_mask])
        _add(SAT_REL, m[s_mask], j[s_mask])

    # attribute edges: seeker-sourced gated by seeker presence; job-sourced always visible
    for rel in ATTR_RELS:
        name = rel[1]
        if name in ("has_skill", "requires_skill") and not fcfg.use_skill_edges:
            continue
        if name in ("has_title", "requires_title") and not fcfg.use_title_edges:
            continue
        ei = art.edges[rel].numpy()
        if rel[0] == "seeker":
            ei = ei[:, present_mem[ei[0]] <= o]
        _add(rel, ei[0], ei[1])
    return {k: v.to(device) for k, v in out.items()}


# ---------------------------------------------------------------------------
# reciprocal ESMM training / eval
# ---------------------------------------------------------------------------
def _log1mexp(x):
    return torch.where(x > -0.6931, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def esmm_loss(a, c, deliver, satisfied):
    """ESMM: accept head on delivered; cond head on satisfied via P(sat)=P(deliver)·P(sat|deliver)."""
    l_accept = F.binary_cross_entropy_with_logits(a, deliver)
    log_p = F.logsigmoid(a) + F.logsigmoid(c)
    l_ctcvr = -(satisfied * log_p + (1 - satisfied) * _log1mexp(log_p.clamp(max=-1e-6))).mean()
    return l_accept + l_ctcvr


def _sup(sp, split, dev):
    d = sp.filter((pl.col("hsplit") == split) & (pl.col("role") == "supervision"))
    return dict(
        m=torch.tensor(d["seeker_idx"].to_numpy(), dtype=torch.long, device=dev),
        j=torch.tensor(d["job_idx"].to_numpy(), dtype=torch.long, device=dev),
        dv=torch.tensor(d["delivered"].to_numpy().astype(np.float32), device=dev),
        st=torch.tensor(d["satisfied"].to_numpy().astype(np.float32), device=dev),
        cohort=d["cohort"].to_numpy(),
        pf=_pf(d, dev))            # champion-matched decoder: per-pair match features (None if absent)


def eval_split(model, x, ei_S, sup, job_exposure=None) -> dict:
    """Reciprocal ESMM eval: joint/accept/cond AUC + PR-AUC, cold/warm cohorts, grouped GAUC/MAP/Hit."""
    model.eval()
    with torch.no_grad():
        z = model.encode(x, ei_S)
        a, c = model.decode_pairs_heads(z, sup["m"], sup["j"], sup.get("pf"))
        pa = torch.sigmoid(a).cpu().numpy(); pc = torch.sigmoid(c).cpu().numpy()
    joint = pa * pc
    dv = sup["dv"].cpu().numpy(); st = sup["st"].cpu().numpy()
    joint_y = ((dv == 1) & (st == 1)).astype(int)       # reciprocal: both sides yes
    warm = sup["cohort"] == "warm"
    deliv = dv == 1

    def auc(y, s, mask=None):
        if mask is not None:
            y, s = y[mask], s[mask]
        return float(roc_auc_score(y, s)) if 0 < int(y.sum()) < len(y) else float("nan")

    def ap(y, s, mask=None):
        if mask is not None:
            y, s = y[mask], s[mask]
        return float(average_precision_score(y, s)) if 0 < int(y.sum()) < len(y) else float("nan")

    out = {
        "joint_auc": auc(joint_y, joint),
        "joint_auc_warm": auc(joint_y, joint, warm),
        "joint_auc_cold": auc(joint_y, joint, ~warm),
        "accept_auc": auc((dv == 1).astype(int), pa),               # predict deliver (seeker side)
        "cond_auc": auc(st[deliv].astype(int), pc[deliv]) if deliv.sum() else float("nan"),  # sat|deliver (in-dist CVR)
        "cvr_entire": auc(joint_y, pc),                             # SSB probe: sigma(c) ALONE over ENTIRE space
        "joint_prauc": ap(joint_y, joint),
        "accept_prauc": ap((dv == 1).astype(int), pa),
        "cond_prauc": ap(st[deliv].astype(int), pc[deliv]) if deliv.sum() else float("nan"),
        "n": len(joint_y), "n_joint_pos": int(joint_y.sum()),
        "n_warm": int(warm.sum()), "n_cold": int((~warm).sum()),
    }
    if job_exposure is not None:                                  # Tech-aligned item (job) cold/warm buckets
        jp = job_exposure[sup["j"].cpu().numpy()]
        cold_j = jp == 0
        out["joint_auc_cold_job"] = auc(joint_y, joint, cold_j)
        out["joint_auc_warm_job"] = auc(joint_y, joint, ~cold_j)
        out["cvr_entire_cold_job"] = auc(joint_y, pc, cold_j)       # SSB probe on unseen jobs
        out["n_cold_job"] = int(cold_j.sum())
    # job- and user-grouped ranking GAUC / MAP@k / Hit@k over the exposed supervision pairs
    from ..evaluate import _exposed_per_group_ranking
    mj = sup["j"].cpu().numpy(); mm = sup["m"].cpu().numpy()
    out["job_ranking"] = _exposed_per_group_ranking(mj, joint_y, joint, (5, 10, 20), group_name="jobs")
    out["user_ranking"] = _exposed_per_group_ranking(mm, joint_y, joint, (5, 10, 20), group_name="users")
    return out


def audit_hybrid(art: Artifacts, sp: pl.DataFrame, ei: dict, verbose=True):
    # CTCVR/joint assume satisfied ⊆ delivered (nested funnel)
    bad_nest = sp.filter((pl.col("satisfied") == 1) & (pl.col("delivered") == 0)).height
    assert bad_nest == 0, f"{bad_nest} satisfied-but-not-delivered pairs violate the funnel nesting"
    edge_split = {(int(a), int(b)): ORDER[s]
                  for a, b, s in zip(sp["seeker_idx"].to_list(), sp["job_idx"].to_list(),
                                     sp["hsplit"].to_list())}
    for split in ("train", "val", "test"):
        o = ORDER[split]
        # browse must NOT be a message-passing relation (exposure only)
        assert ("seeker", "considers", "job") not in ei[split], f"{split}: browse edge in MP graph"
        for rel, col in ((("seeker", "accepts", "job"), "delivered"), (SAT_REL, "satisfied")):
            if rel not in ei[split]:
                continue    # attribute-only arm: no accepts/satisfied MP edges, nothing to leak
            e = ei[split][rel].cpu().numpy()
            mp_pairs = set(zip(e[0].tolist(), e[1].tolist()))
            # (1) current-split supervision outcome edges must be held out
            sup = sp.filter((pl.col("hsplit") == split) & (pl.col("role") == "supervision")
                            & (pl.col(col) == 1))
            leak = set(zip(sup["seeker_idx"].to_list(), sup["job_idx"].to_list())) & mp_pairs
            assert not leak, f"{split}/{rel[1]}: {len(leak)} supervision edges leaked into MP graph"
            # (2) no FUTURE-split edge may appear
            fut = [p for p in mp_pairs if edge_split.get(p, o) > o]
            assert not fut, f"{split}/{rel[1]}: {len(fut)} edges from a FUTURE split in MP graph"
    if verbose:
        print("[hybrid audit] PASS — deliver+satisfied supervision edges held out; no future-split "
              "edge; browse not in MP; satisfied⊆delivered", flush=True)


SAVE_EVERY = 10        # periodic partial-checkpoint cadence (epochs) for crash-safe resume


def train_hybrid(art, sp, mc: ModelConfig, fcfg: FeatureConfig, epochs=250, lr=5e-3,
                 wd=1e-5, seed=0, device="mps", verbose=True, name="hybrid", ckpt_dir=None,
                 patience=50, min_delta=1e-4, loss_kind="esmm", refresh_eval=False):
    import dataclasses as _dc
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device(device if (device != "mps" or torch.backends.mps.is_available()) else "cpu")
    ckpt_path = (Path(ckpt_dir) / f"{name}_seed{seed}.pt") if ckpt_dir is not None else None

    # ---- SKIP: reuse a finished (config,seed) checkpoint's saved metrics (refresh_eval bypasses) ----
    if ckpt_path is not None and ckpt_path.exists() and not refresh_eval:
        st0 = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        if st0.get("done") and "res" in st0:
            if verbose:
                print(f"  [resume] {name} seed{seed} already done — reusing saved metrics", flush=True)
            return st0["res"]

    meta = _hybrid_meta(art, fcfg)          # drops browse(considers), adds satisfied relation
    # content axis: append frozen Qwen node-content when use_qwen_content
    qwen = (_load_qwen_content(art.num_nodes, getattr(fcfg, "emb_model", "qwen3"))
            if getattr(fcfg, "use_qwen_content", False) else None)
    # per-split visibility graphs FIRST so degrees come from each split's own graph (leak-safe)
    ei = {s: build_visibility(art, sp, s, fcfg, dev) for s in ("train", "val", "test")}
    x_content = make_x(art, fcfg, dev, qwen=qwen)                    # content = base ⊕ qwen
    if getattr(fcfg, "tc_ones_collapse", False):                    # p3 content-off analog: base -> ones(1)
        for t in ("seeker", "job"):
            x_content[t] = torch.ones(x_content[t].size(0), 1, device=dev, dtype=x_content[t].dtype)
    content_in = {t: x_content[t].shape[1] for t in NODE_TYPES}     # pre-degree = parallel_ref content_dims

    def _xfeat(split):                                             # content ⊕ per-split leak-safe degree
        if not getattr(fcfg, "tc_degree_nodes", False):
            return x_content                                        # existing arms: byte-identical (no degree)
        deg = _tc_degree_block(ei[split], art.num_nodes, fcfg, dev)
        return {t: torch.cat([x_content[t], deg[t]], dim=1) for t in NODE_TYPES}
    xf = {s: _xfeat(s) for s in ("train", "val", "test")}
    in_dims = {t: xf["train"][t].shape[1] for t in NODE_TYPES}     # post-degree width
    cdims = content_in if mc.kind == "parallel_ref" else None
    model = build_model(_CfgShim(mc, fcfg), in_dims, meta, content_dims=cdims).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    if seed == 0:
        audit_hybrid(art, sp, ei, verbose=verbose)
    tr = _sup(sp, "train", dev); va = _sup(sp, "val", dev)
    ext = int(getattr(fcfg, "ext_pair_dim", 0))    # champion-matched decoder needs pf present
    if ext > 0:
        assert tr.get("pf") is not None and tr["pf"].shape[1] == ext, (
            f"ext_pair_dim={ext} but supervision frame has "
            f"{None if tr.get('pf') is None else tr['pf'].shape[1]} match cols "
            f"(rebuild artifact with match feats / use tianchi_prepared_bv)")
    # job exposure count from TRAIN delivered pairs -> Tech-aligned item cold/warm buckets
    n_jobs = art.num_nodes["job"]
    job_exposure = np.zeros(n_jobs, dtype=np.int64)
    _trd = sp.filter((pl.col("hsplit") == "train") & (pl.col("delivered") == 1)).group_by("job_idx").len()
    if _trd.height:
        job_exposure[_trd["job_idx"].to_numpy()] = _trd["len"].to_numpy()

    # ---- RESUME: continue from a partial (not-done) checkpoint ----
    start_ep, best_val, best_state, best_ep, bad = 0, -1.0, None, -1, 0
    if ckpt_path is not None and ckpt_path.exists():
        st0 = torch.load(ckpt_path, weights_only=False, map_location=dev)
        if not st0.get("done"):
            model.load_state_dict(st0["last_state"])
            best_state = st0.get("best_state"); best_val = st0.get("best_metric", -1.0)
            best_ep = st0.get("best_ep", -1); bad = st0.get("bad", 0)
            start_ep = int(st0.get("epoch", -1)) + 1
            try:
                np.random.set_state(st0["np_rng"]); torch.set_rng_state(st0["torch_rng"])
            except Exception:
                pass
            if verbose:
                print(f"  [resume] {name} seed{seed} from ep{start_ep} (best {best_val:.4f}@ep{best_ep})",
                      flush=True)

    def _save(ep, done, last=None, res=None):
        if ckpt_path is None:
            return
        if last is None:                    # save current (= last-epoch) weights
            last = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        extra = {"flow": "hybrid", "ext_pair_dim": ext, "epoch": ep, "bad": bad,
                 "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state()}
        if res is not None:
            extra["res"] = res
        save_ckpt(ckpt_path, name=name, seed=seed, model_cfg=_dc.asdict(mc),
                  feature_cfg=_dc.asdict(fcfg), in_dims=in_dims,
                  best_state=best_state, last_state=last, best_ep=best_ep,
                  best_metric=best_val, select_metric="joint_auc", done=done, extra=extra)

    # ---- REFRESH-EVAL: re-score a finished ckpt's best_state (no training); missing ckpt -> None ----
    if refresh_eval:
        if not (ckpt_path is not None and ckpt_path.exists()):
            return None
        st0 = torch.load(ckpt_path, weights_only=False, map_location=dev)
        if not st0.get("done"):
            return None
        model.load_state_dict(st0.get("best_state") or st0["last_state"])
        best_state = st0.get("best_state"); best_ep = st0.get("best_ep", -1)
        best_val = st0.get("best_metric", -1.0); ep = int(st0.get("epoch", -1))
        res = {"seed": seed, "best_ep": best_ep, "epochs_run": ep + 1, "secs": 0.0,
               "val": eval_split(model, xf["val"], ei["val"], va, job_exposure=job_exposure),
               "test": eval_split(model, xf["test"], ei["test"], _sup(sp, "test", dev),
                                  job_exposure=job_exposure)}
        _save(ep, done=True, last=st0.get("last_state"), res=res)
        if verbose:
            print(f"  [refresh-eval] {name} seed{seed}: re-scored best_state "
                  f"(test joint={res['test']['joint_auc']:.4f}, "
                  f"user_GAUC={res['test']['user_ranking']['GAUC']:.4f})", flush=True)
        return res

    t0 = time.time()
    ep = start_ep - 1
    for ep in range(start_ep, epochs):
        model.train(); opt.zero_grad()
        z = model.encode(xf["train"], ei["train"])
        a, c = model.decode_pairs_heads(z, tr["m"], tr["j"], tr.get("pf"))
        if loss_kind == "esmm":
            loss = esmm_loss(a, c, tr["dv"], tr["st"])
        loss.backward(); opt.step()
        v = eval_split(model, xf["val"], ei["val"], va)["joint_auc"]   # (1) validate EVERY epoch
        improved = (v == v) and (v > best_val + min_delta)
        if improved:
            best_val, best_ep, bad = v, ep, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        if verbose and (ep % 5 == 0 or improved or ep == epochs - 1):
            print(f"  ep{ep:02d} loss={loss.item():.4f} val_joint_auc={v:.4f}"
                  f"{' *' if improved else ''}", flush=True)
        if ckpt_path is not None and ep % SAVE_EVERY == 0:      # (4) periodic partial ckpt for resume
            _save(ep, done=False)
        if bad >= patience:                                    # (2) early stopping, patience=50
            if verbose:
                print(f"  early stop @ ep{ep:02d} (best {best_val:.4f} @ ep{best_ep:02d})", flush=True)
            break

    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    res = {"seed": seed, "best_ep": best_ep, "epochs_run": ep + 1,
           "secs": round(time.time() - t0, 1),
           "val": eval_split(model, xf["val"], ei["val"], va, job_exposure=job_exposure),
           "test": eval_split(model, xf["test"], ei["test"], _sup(sp, "test", dev),
                              job_exposure=job_exposure)}
    _save(ep, done=True, last=last_state, res=res)             # (4) final done=True ckpt w/ metrics
    return res


def run_hybrid(seeds=(0, 1, 2), epochs=250, device="mps", tag="hybrid_v1", verbose=True,
               skill_edges=True, art_dir=None, save=True, patience=50, only=None,
               emb_model="qwen3", refresh_eval=False, conv_type="sage", attn_heads=None):
    art = load_artifacts(art_dir=art_dir)
    sp = assign_hybrid_split(art.pairs)
    if verbose:
        summ = sp.group_by(["cohort", "hsplit", "role"]).len().sort(["cohort", "hsplit", "role"])
        print(summ)
    # champion-matched decoder (ESMM + LiRank + match feats) when the artifact carries them
    from .build import TIANCHI_MATCH_DIM, TIANCHI_MATCH_FEATURES
    has_match = all(c in art.pairs.columns for c in TIANCHI_MATCH_FEATURES)
    D = TIANCHI_MATCH_DIM if has_match else 0
    print(f"[hybrid] match features {'ON' if has_match else 'OFF'} (ext_pair_dim={D})", flush=True)

    def _mc(kind, **kw):
        # conv_type: graph aggregator (default sage; gatv2 = attention variant)
        kw.setdefault("conv_type", conv_type)
        # attn_heads: GATv2 heads (lower to fit GPU memory; None = ModelConfig default 4)
        if attn_heads is not None:
            kw.setdefault("attn_heads", attn_heads)
        # graph_mode: per-arm override (default hetero)
        gm = kw.pop("graph_mode", "hetero")
        return ModelConfig(kind=kind, graph_mode=gm, hidden_dim=128, out_dim=64,
                           num_layers=2, dropout=0.2, decoder="esmm",
                           head_calibrator=("lirank" if D > 0 else "none"), **kw)

    def _fc(qwen=True, **kw):
        # content-ON = tabular ⊕ Qwen; content-OFF = tabular only (qwen=False)
        return FeatureConfig(
            use_interaction_edges=kw.pop("use_interaction_edges", True),
            use_attribute_edges=kw.pop("use_attribute_edges", True),
            use_skill_edges=kw.pop("use_skill_edges", skill_edges),
            use_title_edges=kw.pop("use_title_edges", True),
            use_company_edges=False, use_qwen_content=qwen,
            emb_model=emb_model, ext_pair_dim=D, **kw)

    # CONTROLLED Tech-mirror family (p1/p2/p3/p8): content = base ⊕ qwen, leak-safe degree graph feats
    specs = [
        ("p1_ctrl", _mc("mlp"),          _fc(qwen=True)),                                          # content MLP
        ("p2_ctrl", _mc("sage"),         _fc(qwen=True,  tc_degree_nodes=True)),                   # content+degree coupled GNN
        ("p3_ctrl", _mc("sage"),         _fc(qwen=False, tc_degree_nodes=True, tc_ones_collapse=True)),  # ones(1)+degree GNN
        ("p8_ctrl", _mc("parallel_ref"), _fc(qwen=True,  tc_degree_nodes=True)),                   # decoupled: p1_ctrl ∥ p3_ctrl
    ]
    # MODEL-AXIS grid: p2/p3/p8 × hetero-GATv2 (attention vs the SAGE mean-aggregator)
    for sfx, gmode, mov in (("hetgat", "hetero", dict(conv_type="gatv2", attn_heads=1, gat_residual=True)),):
        specs += [
            (f"p2_{sfx}", _mc("sage",         graph_mode=gmode, **mov),
             _fc(qwen=True,  tc_degree_nodes=True)),                                      # coupled content+degree
            (f"p3_{sfx}", _mc("sage",         graph_mode=gmode, **mov),
             _fc(qwen=False, tc_degree_nodes=True, tc_ones_collapse=True)),               # content-off ones+degree
            (f"p8_{sfx}", _mc("parallel_ref", graph_mode=gmode, **mov),
             _fc(qwen=True,  tc_degree_nodes=True)),                                      # p1 ∥ p3 (parallel_ref)
        ]

    if only:                                        # run a subset by name (e.g. just the two-tower)
        keep = set(only if not isinstance(only, str) else only.split(","))
        specs = [s for s in specs if s[0] in keep]
        assert specs, f"--only matched no hybrid specs: {only}"

    ckpt_dir = (CKPT_DIR / tag) if save else None
    results = []
    for spec in specs:
        name, mc, fcfg = spec[:3]
        extra = spec[3] if len(spec) > 3 else {}      # optional per-spec train kwargs (loss_kind, ...)
        for s in seeds:
            print(f"\n=== hybrid {name} seed {s} ===", flush=True)
            r = train_hybrid(art, sp, mc, fcfg, epochs=epochs, seed=s, device=device,
                             verbose=verbose, name=name, ckpt_dir=ckpt_dir, patience=patience,
                             refresh_eval=refresh_eval, **extra)
            if r is None:                       # refresh_eval: no done ckpt for this (arm,seed) -> skip
                print(f"  [refresh-eval] {name} seed {s}: no done ckpt — skipped", flush=True)
                continue
            r["name"] = name
            results.append(r)
            v, t = r["val"], r["test"]
            print(f"  [{name} seed {s}] VAL joint={v['joint_auc']:.4f} (warm {v['joint_auc_warm']:.4f}/"
                  f"cold {v['joint_auc_cold']:.4f}) accept={v['accept_auc']:.4f} cond={v['cond_auc']:.4f}"
                  f" | TEST joint={t['joint_auc']:.4f} (warm {t['joint_auc_warm']:.4f}/"
                  f"cold {t['joint_auc_cold']:.4f}) accept={t['accept_auc']:.4f} cond={t['cond_auc']:.4f}",
                  flush=True)
    def _nan_to_none(o):     # NaN -> null so the JSON is strictly valid (degenerate-slice AUCs)
        if isinstance(o, dict):
            return {k: _nan_to_none(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_nan_to_none(v) for v in o]
        if isinstance(o, float) and o != o:
            return None
        return o
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{tag}_results.json", "w") as f:
        json.dump(_nan_to_none(results), f, indent=2)
    print(f"\nsaved -> {RESULTS_DIR / (tag + '_results.json')}")
    return results
