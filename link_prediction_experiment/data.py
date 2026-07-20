"""Interaction frame: load the exposure log, align to graph nodes, derive
labels, build the user-disjoint temporal split. One row per (seeker, job).
User-disjoint split (each seeker in earliest split) = leak-safe inductive."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Dict, Tuple

import numpy as np
import polars as pl

from . import paths
from .config import ExperimentConfig, SplitConfig

USER_COL = "seeker_id"
JOB_COL = "job_id"


@dataclass(frozen=True)
class NodeMaps:
    seeker: Dict[str, int]
    job: Dict[str, int]
    skill: Dict[str, int]
    title: Dict[str, int]
    company: Dict[str, int]

    @property
    def n_seeker(self) -> int: return len(self.seeker)
    @property
    def n_job(self) -> int: return len(self.job)


def load_node_maps() -> NodeMaps:
    with open(paths.NODE_MAPS_JSON) as f:
        nm = json.load(f)
    return NodeMaps(seeker=nm["seeker"], job=nm["job"], skill=nm["skill"],
                    title=nm["title"], company=nm["company"])


_CACHE_VERSION = 5  # bump on schema/label change


def _cache_path(sc: SplitConfig) -> "paths.Path":
    tag = (f"interactions_v{_CACHE_VERSION}_{sc.min_date}_{sc.max_date}"
           f"_{sc.val_pct}_{sc.test_pct}.parquet")
    return paths.CACHE_DIR / tag


def load_interactions(cfg: ExperimentConfig, *, force: bool = False) -> Tuple[pl.DataFrame, NodeMaps]:
    """Return (interaction frame, node maps). Cached by split config."""
    nm = load_node_maps()
    cpath = _cache_path(cfg.split)
    if cpath.exists() and not force:
        return pl.read_parquet(cpath), nm

    sc = cfg.split
    seeker_keys = set(nm.seeker.keys())
    job_keys = set(nm.job.keys())

    lf = pl.scan_parquet(paths.TARGET_PARQUET).select([
        USER_COL, JOB_COL, "exposure_date", "exposure_datetime",
        "pos_label", "pos_datetime", "accept_datetime",
        "neg_datetime_a", "neg_datetime_b", "neg_datetime_c",
        "snapshot_date",
    ])
    df = (
        lf
        .filter(pl.col(USER_COL).is_not_null() & pl.col(JOB_COL).is_not_null())
        .filter(pl.col("exposure_datetime").is_not_null())
        .collect()
    )
    # dedup to one row per (seeker, job): latest
    df = (
        df.sort("snapshot_date", descending=True)
          .unique(subset=[USER_COL, JOB_COL], keep="first")
    )

    # align to graph node universe
    df = df.with_columns([
        ("m_" + pl.col(USER_COL).cast(pl.Utf8)).alias("seeker_str"),
        ("j_" + pl.col(JOB_COL).cast(pl.Utf8)).alias("job_str"),
    ])
    df = df.filter(
        pl.col("seeker_str").is_in(list(seeker_keys))
        & pl.col("job_str").is_in(list(job_keys))
    )

    # date window
    dmin = date.fromisoformat(sc.min_date)
    dmax = date.fromisoformat(sc.max_date)
    df = df.filter(
        (pl.col("exposure_date") >= dmin) & (pl.col("exposure_date") <= dmax)
    )

    # labels
    df = df.with_columns([
        (pl.col("pos_label") == True).fill_null(False).cast(pl.Int8).alias("passed"),  # noqa: E712
        pl.col("exposure_datetime").dt.epoch("s").cast(pl.Int64).alias("exposure_ts"),
        # accept outcome time (clamped >= expose); null = no accept
        pl.when(pl.col("accept_datetime").is_not_null())
          .then(pl.max_horizontal([pl.col("accept_datetime"),
                                   pl.col("exposure_datetime")]).dt.epoch("s"))
          .otherwise(None).cast(pl.Int64).alias("accept_event_ts"),
        # failure outcome time (clamped >= expose); non-passed rows only
        pl.when((pl.col("pos_label") != True) &                            # noqa: E712
                pl.min_horizontal([
                    pl.when(pl.col(c).is_not_null())
                      .then(pl.max_horizontal([pl.col(c), pl.col("exposure_datetime")]))
                      .otherwise(None)
                    for c in ("neg_datetime_a", "neg_datetime_b",
                              "neg_datetime_c")]).is_not_null())
          .then(pl.min_horizontal([
                    pl.when(pl.col(c).is_not_null())
                      .then(pl.max_horizontal([pl.col(c), pl.col("exposure_datetime")]))
                      .otherwise(None)
                    for c in ("neg_datetime_a", "neg_datetime_b",
                              "neg_datetime_c")]).dt.epoch("s"))
          .otherwise(None).cast(pl.Int64).alias("neg_event_ts"),
    ])

    # integer node indices
    mem_map = nm.seeker
    job_map = nm.job
    df = df.with_columns([
        pl.col("seeker_str").replace_strict(mem_map, default=-1).alias("seeker_idx"),
        pl.col("job_str").replace_strict(job_map, default=-1).alias("job_idx"),
    ]).filter((pl.col("seeker_idx") >= 0) & (pl.col("job_idx") >= 0))

    # --- user-disjoint global-timeline split ---
    # 1) temporal qcut of exposure_date into train/val/test
    df = df.with_columns(
        pl.col("exposure_date").qcut(
            [sc.val_pct, sc.test_pct], labels=["train", "val", "test"],
            allow_duplicates=True,
        ).cast(pl.Utf8).alias("_row_split")
    )
    # 2) assign each seeker to the split of their EARLIEST exposure
    df = df.with_columns(
        pl.col("_row_split").sort_by("exposure_date").first().over("seeker_str")
          .alias("dataset_split")
    )
    # 3) drop rows whose row-split disagrees with the user's assigned split
    df = df.filter(pl.col("_row_split") == pl.col("dataset_split")).drop("_row_split")

    df = df.select([
        "seeker_str", "job_str", "seeker_idx", "job_idx",
        "exposure_date", "exposure_ts", "accept_event_ts", "neg_event_ts",
        "passed", "dataset_split",
        USER_COL, JOB_COL,
    ])
    # deterministic row order: time-sorted with (seeker_idx, job_idx) tie-break
    df = df.sort(["exposure_ts", "seeker_idx", "job_idx"])
    df.write_parquet(cpath)
    return df, nm
