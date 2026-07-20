"""Graph store: load the prebuilt timestamped HeteroData and expose leak-safe PIT.

At cutoff t the visible graph = interaction edges with time < t, attribute edges of
entities born < t, plus attribute edges of the focal entities being scored.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from . import paths
from .config import FeatureConfig

NODE_TYPES = ["seeker", "job", "skill", "title", "company"]

# canonical forward relations
INTERACTION_RELS = [
    ("seeker", "considers", "job"),
    ("seeker", "accepts", "job"),
    ("job", "screens", "seeker"),
]
ATTRIBUTE_RELS = [
    ("seeker", "has_skill", "skill"),
    ("seeker", "has_title", "title"),
    ("job", "requires_skill", "skill"),
    ("job", "requires_title", "title"),
    ("job", "posted_by", "company"),
]
# real-negative outcome edges (exposed but did not pass); stored in int_* for PIT slicing
OUTCOME_NEG_RELS = [("seeker", "failed", "job")]


def _rev(rel: Tuple[str, str, str]) -> Tuple[str, str, str]:
    s, r, d = rel
    return (d, f"rev_{r}", s)


def enabled_attribute_rels(fcfg: "FeatureConfig") -> List[Tuple[str, str, str]]:
    """Attribute relations enabled by a feature config (skill/title/company toggles)."""
    rels: List[Tuple[str, str, str]] = []
    for rel in ATTRIBUTE_RELS:
        r = rel[1]
        if r in ("has_skill", "requires_skill") and not fcfg.use_skill_edges:
            continue
        if r in ("has_title", "requires_title") and not fcfg.use_title_edges:
            continue
        if r == "posted_by" and not fcfg.use_company_edges:
            continue
        rels.append(rel)
    return rels


def configured_edge_types(fcfg: FeatureConfig) -> List[Tuple[str, str, str]]:
    """Exact hetero relation set (incl. reverses) for a feature config."""
    rels: List[Tuple[str, str, str]] = []
    if getattr(fcfg, "use_interaction_edges", True):
        for rel in INTERACTION_RELS:
            rels.append(rel); rels.append(_rev(rel))
    if fcfg.use_attribute_edges:
        for rel in ATTRIBUTE_RELS:
            r = rel[1]
            if r in ("has_skill", "requires_skill") and not fcfg.use_skill_edges:
                continue
            if r in ("has_title", "requires_title") and not fcfg.use_title_edges:
                continue
            if r == "posted_by" and not fcfg.use_company_edges:
                continue
            rels.append(rel); rels.append(_rev(rel))
    return rels


def attach_outcome_neg_edges(gstore, src_seeker, dst_job, neg_ts) -> None:
    """Attach real-negative (seeker, failed, job) edges (+reverse) to gstore.int_*, time-sorted."""
    rel = OUTCOME_NEG_RELS[0]
    if rel in gstore.int_time or len(src_seeker) == 0:
        return
    t = np.asarray(neg_ts, dtype=np.int64)
    order = np.argsort(t, kind="stable")
    fwd = torch.as_tensor(
        np.stack([np.asarray(src_seeker, dtype=np.int64)[order],
                  np.asarray(dst_job, dtype=np.int64)[order]]), dtype=torch.long)
    gstore.int_fwd[rel] = fwd
    gstore.int_rev[rel] = fwd.flip(0).contiguous()
    gstore.int_time[rel] = t[order]


@dataclass
class GraphStore:
    num_nodes: Dict[str, int]
    # interaction: forward + reverse edge tensors sorted by time, plus the times
    int_fwd: Dict[Tuple[str, str, str], torch.Tensor]
    int_rev: Dict[Tuple[str, str, str], torch.Tensor]
    int_time: Dict[Tuple[str, str, str], np.ndarray]   # sorted ascending
    # attribute: forward + reverse edge tensors sorted by birth time, + src/time
    attr_fwd: Dict[Tuple[str, str, str], torch.Tensor]
    attr_rev: Dict[Tuple[str, str, str], torch.Tensor]
    attr_time: Dict[Tuple[str, str, str], np.ndarray]  # sorted ascending (= src birth)
    attr_src: Dict[Tuple[str, str, str], np.ndarray]   # src node id per edge
    # homogeneous unified-space precompute
    node_offset: Dict[str, int]
    homo_int_ei: torch.Tensor           # (2, E) unified, both directions, sorted by time
    homo_int_time: np.ndarray           # (E,) sorted ascending

    # ---- node-type helpers ----
    @property
    def n_total(self) -> int:
        return sum(self.num_nodes[t] for t in NODE_TYPES)

    def to(self, device) -> "GraphStore":
        """Return a device copy of the edge tensors (non-mutating; numpy arrays shared)."""
        return GraphStore(
            num_nodes=self.num_nodes,
            int_fwd={k: v.to(device) for k, v in self.int_fwd.items()},
            int_rev={k: v.to(device) for k, v in self.int_rev.items()},
            int_time=self.int_time,
            attr_fwd={k: v.to(device) for k, v in self.attr_fwd.items()},
            attr_rev={k: v.to(device) for k, v in self.attr_rev.items()},
            attr_time=self.attr_time, attr_src=self.attr_src,
            node_offset=self.node_offset,
            homo_int_ei=self.homo_int_ei.to(device),
            homo_int_time=self.homo_int_time,
        )

    # ---- focal lookup ----
    @staticmethod
    def _focal_bool(n: int, focal) -> Optional[np.ndarray]:
        if focal is None:
            return None
        b = np.zeros(n, dtype=bool)
        f = np.asarray(list(focal), dtype=np.int64)
        if f.size:
            b[f] = True
        return b

    def _attr_idx(self, rel, cutoff_ts: int, fm: Optional[np.ndarray],
                  fj: Optional[np.ndarray]) -> np.ndarray:
        """Indices of attribute edges visible at cutoff: born < cutoff, plus focal-source edges."""
        t = self.attr_time[rel]
        k = int(np.searchsorted(t, cutoff_ts, side="left"))   # born strictly before cutoff
        fb = fm if rel[0] == "seeker" else fj
        if fb is None or k >= len(t):
            return np.arange(k)
        suffix_src = self.attr_src[rel][k:]
        extra = k + np.nonzero(fb[suffix_src])[0]
        if extra.size == 0:
            return np.arange(k)
        return np.concatenate([np.arange(k), extra])

    # ---- PIT ----
    def hetero_pit(self, cutoff_ts: int, fcfg: FeatureConfig,
                   focal_seekers=None, focal_jobs=None) -> Dict[Tuple[str, str, str], torch.Tensor]:
        out: Dict[Tuple[str, str, str], torch.Tensor] = {}
        if getattr(fcfg, "use_interaction_edges", True):
          for rel in INTERACTION_RELS:
            k = int(np.searchsorted(self.int_time[rel], cutoff_ts, side="left"))
            out[rel] = self.int_fwd[rel][:, :k]
            out[_rev(rel)] = self.int_rev[rel][:, :k]
        if fcfg.use_attribute_edges:
            fm = self._focal_bool(self.num_nodes["seeker"], focal_seekers)
            fj = self._focal_bool(self.num_nodes["job"], focal_jobs)
            for rel in self._enabled_attr(fcfg):
                idx = self._attr_idx(rel, cutoff_ts, fm, fj)
                dev = self.attr_fwd[rel].device
                it = torch.as_tensor(idx, dtype=torch.long, device=dev)
                out[rel] = self.attr_fwd[rel].index_select(1, it)
                out[_rev(rel)] = self.attr_rev[rel].index_select(1, it)
        return out

    def _enabled_attr(self, fcfg: FeatureConfig) -> List[Tuple[str, str, str]]:
        return enabled_attribute_rels(fcfg)


def load_graph_store() -> GraphStore:
    data = torch.load(paths.HETERO_GRAPH_PT, weights_only=False)
    num_nodes = {t: int(data[t].num_nodes) for t in NODE_TYPES}

    int_fwd, int_rev, int_time = {}, {}, {}
    for rel in INTERACTION_RELS:
        ei = data[rel].edge_index.long()
        t = data[rel].time.numpy().astype(np.int64)
        order = np.argsort(t, kind="stable")
        ei_s = ei[:, order].contiguous()
        int_fwd[rel] = ei_s
        int_rev[rel] = ei_s.flip(0).contiguous()
        int_time[rel] = t[order]

    attr_fwd, attr_rev, attr_time, attr_src = {}, {}, {}, {}
    for rel in ATTRIBUTE_RELS:
        ei = data[rel].edge_index.long()
        t = data[rel].time.numpy().astype(np.int64)   # = source entity birth
        order = np.argsort(t, kind="stable")
        ei_s = ei[:, order].contiguous()
        attr_fwd[rel] = ei_s
        attr_rev[rel] = ei_s.flip(0).contiguous()
        attr_time[rel] = t[order]
        attr_src[rel] = ei_s[0].numpy()

    # homogeneous unified space: seeker | job | skill | title | company
    offset, cum = {}, 0
    for t in NODE_TYPES:
        offset[t] = cum
        cum += num_nodes[t]

    # unified interaction edges (both directions), sorted by time
    eis, ts = [], []
    for rel in INTERACTION_RELS:
        s, _, d = rel
        ei = int_fwd[rel]
        t = int_time[rel]
        ei_u = torch.stack([ei[0] + offset[s], ei[1] + offset[d]])
        eis.append(ei_u); ts.append(t)
        eis.append(ei_u.flip(0)); ts.append(t)   # reverse, same times
    homo_int_ei = torch.cat(eis, dim=1)
    homo_int_time = np.concatenate(ts)
    order = np.argsort(homo_int_time, kind="stable")
    homo_int_ei = homo_int_ei[:, order].contiguous()
    homo_int_time = homo_int_time[order]

    return GraphStore(
        num_nodes=num_nodes,
        int_fwd=int_fwd, int_rev=int_rev, int_time=int_time,
        attr_fwd=attr_fwd, attr_rev=attr_rev, attr_time=attr_time, attr_src=attr_src,
        node_offset=offset, homo_int_ei=homo_int_ei, homo_int_time=homo_int_time,
    )


def validate_graph_store(gstore: "GraphStore", nm=None, *, verbose: bool = True) -> None:
    """Structural integrity pre-flight for a loaded GraphStore (G1-G7). Raises on any violation.

    Checks node-map/num_nodes agreement, shape alignment, rev==fwd.flip, time-sort,
    index bounds, attr_src, and homo consistency (structure, not leak-safety).
    """
    fails: List[str] = []
    n_checks = 0

    def _req(ok: bool, msg: str) -> None:
        nonlocal n_checks
        n_checks += 1
        if not ok:
            fails.append(msg)

    # G1 node-map <-> num_nodes
    if nm is not None:
        for t in NODE_TYPES:
            m = getattr(nm, t, None)
            if m is not None:
                _req(len(m) == gstore.num_nodes[t],
                     f"G1 {t}: node_map size {len(m)} != num_nodes {gstore.num_nodes[t]}")

    def _check_rel(kind, rel, fwd, rev, tim, src=None):
        st, _, dt = rel
        n = int(tim.shape[0])
        shp_ok = tuple(fwd.shape) == (2, n) and tuple(rev.shape) == (2, n) \
            and (src is None or int(src.shape[0]) == n)
        _req(shp_ok, f"G2 {kind} {rel}: shapes fwd={tuple(fwd.shape)} "
                     f"rev={tuple(rev.shape)} time={n}"
                     + ("" if src is None else f" src={int(src.shape[0])}"))
        _req(n <= 1 or bool((np.diff(tim) >= 0).all()),
             f"G4 {kind} {rel}: time not ascending")
        if not shp_ok or n == 0:
            return
        fcpu = fwd.cpu()
        _req(torch.equal(rev.cpu(), fcpu.flip(0)), f"G3 {kind} {rel}: rev != fwd.flip(0)")
        if src is not None:
            _req(bool((src == fcpu[0].numpy()).all()), f"G6 {kind} {rel}: attr_src != attr_fwd[0]")
        _req(0 <= int(fcpu[0].min()) and int(fcpu[0].max()) < gstore.num_nodes[st],
             f"G5 {kind} {rel}: src idx out of [0,{gstore.num_nodes[st]})")
        _req(0 <= int(fcpu[1].min()) and int(fcpu[1].max()) < gstore.num_nodes[dt],
             f"G5 {kind} {rel}: dst idx out of [0,{gstore.num_nodes[dt]})")

    for rel in gstore.int_time:
        _check_rel("int", rel, gstore.int_fwd[rel], gstore.int_rev[rel], gstore.int_time[rel])
    for rel in gstore.attr_time:
        _check_rel("attr", rel, gstore.attr_fwd[rel], gstore.attr_rev[rel],
                   gstore.attr_time[rel], gstore.attr_src[rel])

    # G7 homogeneous unified-space interaction edges (both directions, sorted)
    hei, ht = gstore.homo_int_ei, gstore.homo_int_time
    _req(int(hei.shape[1]) == int(ht.shape[0]),
         f"G7 homo: ei cols {int(hei.shape[1])} != time {int(ht.shape[0])}")
    _req(ht.shape[0] <= 1 or bool((np.diff(ht) >= 0).all()), "G7 homo: time not ascending")
    exp = 2 * sum(int(gstore.int_time[r].shape[0]) for r in INTERACTION_RELS if r in gstore.int_time)
    _req(int(hei.shape[1]) == exp,
         f"G7 homo: edge count {int(hei.shape[1])} != 2*interaction {exp}")
    if hei.numel():
        _req(0 <= int(hei.min()) and int(hei.max()) < gstore.n_total,
             f"G7 homo: node idx out of [0,{gstore.n_total})")

    if verbose:
        print(f"   [graph-store validation] {'PASS' if not fails else 'FAIL'}: "
              f"{n_checks - len(fails)}/{n_checks} checks passed")
        for f in fails:
            print(f"      [FAIL] {f}")
    if fails:
        raise AssertionError(f"GraphStore validation failed ({len(fails)} violations): {fails}")
