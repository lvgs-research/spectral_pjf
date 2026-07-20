"""Leak-safety audits (A1-A8): user-disjoint split, inductiveness, strict-PIT.

Run before trusting any result.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import polars as pl

from .config import FeatureConfig
from .graph_build import GraphStore, INTERACTION_RELS


def _check(name: str, ok: bool, detail: str) -> Tuple[str, bool, str]:
    return (name, bool(ok), detail)


def audit_inductive(cfg, in_dims, gstore: GraphStore) -> List[Tuple[str, bool, str]]:
    """A8: model is inductive — no per-node id embedding lookup."""
    import torch.nn as nn
    from .models import build_model, GraphMeta
    from .graph_build import configured_edge_types
    meta = GraphMeta(num_nodes=gstore.num_nodes, edge_types=configured_edge_types(cfg.feature),
                     node_offset=gstore.node_offset, n_total=gstore.n_total)
    model = build_model(cfg, in_dims, meta)
    n_emb = sum(1 for m in model.modules() if isinstance(m, nn.Embedding))
    return [_check("A8 inductive (no per-node id embedding)", n_emb == 0,
                   f"{n_emb} nn.Embedding modules (expected 0 — feature-based encoders)")]


def run_audits(df: pl.DataFrame, gstore: GraphStore, fcfg: FeatureConfig,
               *, cfg=None, in_dims=None) -> List[Tuple[str, bool, str]]:
    out: List[Tuple[str, bool, str]] = []

    # A1 user-disjoint
    dup = (df.select("seeker_str", "dataset_split").unique()
             .group_by("seeker_str").len().filter(pl.col("len") > 1).height)
    out.append(_check("A1 user-disjoint split", dup == 0, f"{dup} users in >1 split"))

    # A2 inductive (val/test unseen in train)
    tr = set(df.filter(pl.col("dataset_split") == "train")["seeker_str"].to_list())
    va = set(df.filter(pl.col("dataset_split") == "val")["seeker_str"].to_list())
    te = set(df.filter(pl.col("dataset_split") == "test")["seeker_str"].to_list())
    leak = len((va | te) & tr)
    out.append(_check("A2 inductive (val/test unseen)", leak == 0,
                      f"{leak} val/test users also in train"))

    # A3/A4 audit the interaction edges (only when message-passed)
    times_all = np.concatenate([gstore.int_time[r] for r in INTERACTION_RELS])
    if getattr(fcfg, "use_interaction_edges", True):
        # A3 strict-PIT: interaction edges < cutoff
        cutoff = int(np.quantile(times_all, 0.6))
        eid = gstore.hetero_pit(cutoff, fcfg)
        bad = 0
        for r in INTERACTION_RELS:
            k = int(np.searchsorted(gstore.int_time[r], cutoff, side="left"))
            if eid[r].shape[1] != k:
                bad += 1
            if k > 0 and gstore.int_time[r][k - 1] >= cutoff:
                bad += 1
        out.append(_check("A3 strict-PIT interaction edges", bad == 0,
                          f"cutoff={cutoff}; {bad} relation violations"))

        # A4 monotone visibility
        lo, hi = int(np.quantile(times_all, 0.3)), int(np.quantile(times_all, 0.9))
        n_lo = sum(gstore.hetero_pit(lo, fcfg)[r].shape[1] for r in INTERACTION_RELS)
        n_hi = sum(gstore.hetero_pit(hi, fcfg)[r].shape[1] for r in INTERACTION_RELS)
        out.append(_check("A4 monotone visibility", n_hi >= n_lo,
                          f"edges@lo={n_lo} <= edges@hi={n_hi}"))
    else:
        out.append(_check("A3/A4 interaction edges OFF", True, "interaction edges disabled (attr-only)"))

    # A5 attribute edges birth-gated (no future-entity profile leak)
    if fcfg.use_attribute_edges:
        arels = gstore._enabled_attr(fcfg)
        def n_attr(cut, **kw):
            eg = gstore.hetero_pit(cut, fcfg, **kw)
            return sum(eg[r].shape[1] for r in arels)
        attr_min = min(int(gstore.attr_time[r].min()) for r in arels)
        attr_max = max(int(gstore.attr_time[r].max()) for r in arels)
        n_before = n_attr(attr_min - 10)
        n_mid = n_attr(int((attr_min + attr_max) / 2))
        n_after = n_attr(attr_max + 10)
        ok5 = (n_before == 0) and (n_mid > n_before) and (n_after >= n_mid)
        out.append(_check("A5 attribute edges birth-gated (no future leak)", ok5,
                          f"#attr before-births={n_before}, mid={n_mid}, after={n_after}"))

        # A6 focal entity retains its profile even if born >= cutoff
        rel = ("job", "requires_skill", "skill")
        early = attr_min - 10
        fut_jobs = np.unique(gstore.attr_src[rel][gstore.attr_time[rel] >= early + 5])
        if fut_jobs.size:
            j_future = int(fut_jobs[0])
            base = gstore.hetero_pit(early, fcfg)[rel].shape[1]
            with_focal = gstore.hetero_pit(early, fcfg, focal_jobs=[j_future])
            src_present = j_future in set(with_focal[rel][0].cpu().numpy().tolist())
            ok6 = with_focal[rel].shape[1] > base and src_present
            out.append(_check("A6 focal entity profile retained", ok6,
                              f"future job {j_future}: base_attr={base} -> with_focal={with_focal[rel].shape[1]}, src_present={src_present}"))

    # A8 inductive check — only when cfg is supplied
    if cfg is not None and in_dims is not None:
        out += audit_inductive(cfg, in_dims, gstore)
    return out


def assert_clean(df: pl.DataFrame, gstore: GraphStore, fcfg: FeatureConfig, *,
                 verbose=True, cfg=None, in_dims=None) -> None:
    results = run_audits(df, gstore, fcfg, cfg=cfg, in_dims=in_dims)
    if verbose:
        for name, ok, detail in results:
            print(f"   [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    failed = [r for r in results if not r[1]]
    if failed:
        raise AssertionError(f"Leak-safety audit failed: {[r[0] for r in failed]}")
