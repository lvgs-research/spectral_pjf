"""Exposure count cohorts and cold/warm definitions.

exposure count = cf_n_prior (prior point-in-time exposures of the job); cold job = cf_n_prior==0.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import polars as pl

from .graph_build import GraphStore

# exposure count buckets on cf_n_prior -> [0],[1-2],[3-5],[6-20],[21+]
_BUCKET_EDGES = [1, 3, 6, 21]
_BUCKET_LABELS = ["0", "1-2", "3-5", "6-20", "21+"]


def build_job_exposure_times(df: pl.DataFrame) -> Dict[int, np.ndarray]:
    """job_idx -> sorted ascending array of all exposure timestamps (df rows)."""
    job = df["job_idx"].to_numpy()
    ts = df["exposure_ts"].to_numpy()
    order = np.argsort(job, kind="stable")
    job_s, ts_s = job[order], ts[order]
    out: Dict[int, np.ndarray] = {}
    # split into per-job blocks
    uniq, starts = np.unique(job_s, return_index=True)
    starts = list(starts) + [len(job_s)]
    for i, j in enumerate(uniq):
        block = np.sort(ts_s[starts[i]:starts[i + 1]])
        out[int(j)] = block
    return out


_BIG = 1 << 33   # > max unix-second ts; packs (job, ts) into one monotone key


def cf_n_prior(job_times: Dict[int, np.ndarray], job_idx: np.ndarray,
               ts: np.ndarray) -> np.ndarray:
    """For each (job, anchor_ts): #events on job strictly before anchor_ts (excludes ties)."""
    job_idx = np.asarray(job_idx, dtype=np.int64)
    ts = np.asarray(ts, dtype=np.int64)
    if not job_times:
        return np.zeros(len(job_idx), dtype=np.int64)
    jids = np.sort(np.fromiter(job_times.keys(), dtype=np.int64, count=len(job_times)))
    parts = [job_times[int(j)].astype(np.int64) + int(j) * _BIG for j in jids]  # sorted within & across
    key = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
    hi = np.searchsorted(key, job_idx * _BIG + ts, side="left")
    lo = np.searchsorted(key, job_idx * _BIG, side="left")
    return (hi - lo).astype(np.int64)


def _bucket(counts: np.ndarray) -> np.ndarray:
    idx = np.digitize(counts, _BUCKET_EDGES, right=False)
    return np.asarray(_BUCKET_LABELS, dtype=object)[idx]


def assign_cohorts(df: pl.DataFrame, gstore: GraphStore, event: str) -> pl.DataFrame:
    """Attach cf_n_prior, cold_job, cold_user, exposure_bucket to the frame."""
    jt = build_job_exposure_times(df)
    job_idx = df["job_idx"].to_numpy()
    ts = df["exposure_ts"].to_numpy()
    counts = cf_n_prior(jt, job_idx, ts)

    train_seekers = set(df.filter(pl.col("dataset_split") == "train")["seeker_str"].to_list())
    cold_user = np.fromiter((m not in train_seekers for m in df["seeker_str"].to_list()),
                            dtype=bool, count=df.height)
    out = df.with_columns([
        pl.Series("cf_n_prior", counts),
        pl.Series("cold_job", counts == 0),
        pl.Series("cold_user", cold_user),
        pl.Series("exposure_bucket", _bucket(counts).tolist()),
    ])
    return out
