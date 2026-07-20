"""Person-job match features: skill overlap (paper) + opaque categorical matches.

Skill matching is kept (skills are a paper feature). The remaining tabular match features
carry no domain semantics: each is an equality match on an opaque categorical attribute.
Feature preprocessing is a no-op passthrough — values ship as-is."""
from __future__ import annotations

import numpy as np
import polars as pl

from . import paths
from .data import load_node_maps

_N_CAT = 8                                                   # opaque categorical match slots
MATCH_TAB_FEATURES = ["skill_jaccard", "skill_overlap"] + [f"cat_match_{k}" for k in range(_N_CAT)]
MATCH_TAB_DIM = len(MATCH_TAB_FEATURES)
_OVERLAP_NORM = float(np.log1p(10))                          # skill_overlap_count ~ [0, 1.5]
_SEEKER_CAT_COLS = [f"feature_{i}" for i in range(2, 2 + _N_CAT)]
_JOB_CAT_COLS = [f"feature_{i}" for i in range(14, 14 + _N_CAT)]

_CACHE = {}   # (member, job) attr tables, built once per process


def passthrough(x):
    """No-op preprocessing: used as a decorator, returns its input unchanged (values ship as-is)."""
    return x


def _skill_set(lst):
    if not lst:
        return frozenset()
    return frozenset(s["skill"] for s in lst
                     if s and s.get("skill"))


@passthrough
def _build_member_table(nm, canon=False):
    cols = ["seeker_id", "snapshot_date", "skill_set"] + _SEEKER_CAT_COLS
    df = (pl.scan_parquet(paths.SEEKER_PARQUET).select(cols).collect()
          .sort("snapshot_date", descending=True)
          .unique(subset=["seeker_id"], keep="first"))
    n = nm.n_seeker
    skills = [frozenset()] * n
    cat = np.full((n, _N_CAT), -1, np.int64)
    for r in df.iter_rows(named=True):
        idx = nm.seeker.get(f"m_{r['seeker_id']}")
        if idx is None:
            continue
        skills[idx] = _skill_set(r["skill_set"])
        cat[idx] = [int(r[c]) if r[c] is not None else -1 for c in _SEEKER_CAT_COLS]
    return {"skills": skills, "cat": cat}


@passthrough
def _build_job_table(nm, canon=False):
    graph_rids = [int(k[2:]) for k in nm.job.keys()]          # strip "j_" prefix
    cols = ["job_id", "snapshot_date", "skill_set"] + _JOB_CAT_COLS
    df = (pl.scan_parquet(paths.JOB_PARQUET).select(cols)
          .filter(pl.col("job_id").is_in(graph_rids)).collect()    # bound memory
          .sort("snapshot_date", descending=True)
          .unique(subset=["job_id"], keep="first"))
    n = nm.n_job
    skills = [frozenset()] * n
    cat = np.full((n, _N_CAT), -1, np.int64)
    for r in df.iter_rows(named=True):
        idx = nm.job.get(f"j_{r['job_id']}")
        if idx is None:
            continue
        skills[idx] = _skill_set(r["skill_set"])
        cat[idx] = [int(r[c]) if r[c] is not None else -1 for c in _JOB_CAT_COLS]
    return {"skills": skills, "cat": cat}


def match_tables(canon=False):
    """(member_table, job_table) keyed by node idx; cached."""
    if canon not in _CACHE:
        nm = load_node_maps()
        _CACHE[canon] = (_build_member_table(nm, canon), _build_job_table(nm, canon))
    return _CACHE[canon]


def compute_tab_match(mt, jt, m_np, j_np) -> np.ndarray:
    """[N, MATCH_TAB_DIM] float32 match features (order = MATCH_TAB_FEATURES)."""
    m = np.asarray(m_np); j = np.asarray(j_np); N = len(m)
    out = np.zeros((N, MATCH_TAB_DIM), np.float32)
    # skill match (kept — skills are a paper feature)
    msk, jsk = mt["skills"], jt["skills"]
    for i in range(N):
        a, b = msk[m[i]], jsk[j[i]]
        if a and b:
            inter = len(a & b)
            out[i, 0] = inter / len(a | b)
            out[i, 1] = float(np.log1p(inter)) / _OVERLAP_NORM
    # opaque categorical matches: per-slot equality of two categorical attributes
    mc = mt["cat"][m]; jc = jt["cat"][j]                     # [N, _N_CAT]
    out[:, 2:] = ((mc == jc) & (mc >= 0)).astype(np.float32)
    return out
