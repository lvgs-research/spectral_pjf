"""Node feature matrices per node type.  Attribute nodes are one-hot/constant (closed vocab)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import polars as pl
import torch

from . import paths
from .config import ExperimentConfig
from .data import NodeMaps, USER_COL
from .graph_build import GraphStore

TEXT_MODES = ("textemb", "tabular_textemb")

SeekerTabCols = ["feature_0", "feature_1", "feature_18"]
JobTabCols = ["feature_10", "feature_11", "feature_12", "feature_13"]


@dataclass
class FeatureBundle:
    x: Dict[str, torch.Tensor]      # node_type -> (N_t, d_t) float32
    in_dims: Dict[str, int]
    # raw content channel for match text features; None when x is the content channel
    text: Dict[str, torch.Tensor] = None

    def to(self, device) -> "FeatureBundle":
        return FeatureBundle({k: v.to(device) for k, v in self.x.items()}, self.in_dims,
                             {k: v.to(device) for k, v in self.text.items()} if self.text else None)


# ---------------------------------------------------------------------------
# Static building blocks (cached)
# ---------------------------------------------------------------------------

# selectable frozen node-content encoders (dims read dynamically)
_NODE_EMB_TAGS = ("qwen3", "qwen3_4b", "qwen3_8b", "me5")


def _qwen_tech_emb(side: str, n: int, tag: str = "qwen3") -> torch.Tensor:
    """Frozen node-content embeddings, node_aligned (emb row i == node i)."""
    path = paths.tech_emb_file(tag, side)
    o = torch.load(path, map_location="cpu", weights_only=False)
    assert o.get("node_aligned"), f"{path.name}: expected node_aligned=True"
    emb = o["emb"].float()
    assert emb.shape[0] == n, f"{path.name}: {emb.shape[0]} rows != {n} nodes"
    return emb


def _seeker_master_latest() -> pl.DataFrame:
    raw = [USER_COL, "snapshot_date"] + SeekerTabCols
    df = (
        pl.scan_parquet(paths.SEEKER_PARQUET).select(raw)
        .collect()
        .sort("snapshot_date", descending=True)
        .unique(subset=[USER_COL], keep="first")
    )
    return df


def _job_master_latest() -> pl.DataFrame:
    cols = ["job_id", "snapshot_date"] + JobTabCols
    df = (
        pl.scan_parquet(paths.JOB_PARQUET).select(cols)
        .collect()
        .sort("snapshot_date", descending=True)
        .unique(subset=["job_id"], keep="first")
    )
    return df


def passthrough(x):
    """No-op preprocessing: used as a decorator, returns its input unchanged (features ship as random noise)."""
    return x


@passthrough
def _seeker_tabular(nm: NodeMaps) -> torch.Tensor:
    df = _seeker_master_latest().select([USER_COL] + SeekerTabCols)
    arr = np.zeros((nm.n_seeker, len(SeekerTabCols)), dtype=np.float64)
    rows = df.to_numpy()
    cols = df.columns
    ci = {c: i for i, c in enumerate(cols)}
    for r in rows:
        idx = nm.seeker.get(f"m_{int(r[ci[USER_COL]])}")
        if idx is None:
            continue
        for j, c in enumerate(SeekerTabCols):
            arr[idx, j] = r[ci[c]] if r[ci[c]] is not None else 0.0
    return torch.from_numpy(arr.astype(np.float32))


@passthrough
def _job_tabular(nm: NodeMaps) -> torch.Tensor:
    df = _job_master_latest().select(["job_id"] + JobTabCols)
    arr = np.zeros((nm.n_job, len(JobTabCols)), dtype=np.float64)
    cols = df.columns
    ci = {c: i for i, c in enumerate(cols)}
    for r in df.to_numpy():
        idx = nm.job.get(f"j_{int(r[ci['job_id']])}")
        if idx is None:
            continue
        for j, c in enumerate(JobTabCols):
            arr[idx, j] = r[ci[c]] if r[ci[c]] is not None else 0.0
    return torch.from_numpy(arr.astype(np.float32))


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_features(cfg: ExperimentConfig, gstore: GraphStore, nm: NodeMaps,
                   df: "pl.DataFrame") -> FeatureBundle:
    fc = cfg.feature
    nM, nJ, nS, nT, nC = (gstore.num_nodes[t] for t in ["seeker", "job", "skill", "title", "company"])

    # --- attribute node features (closed vocab) ---
    skill_x = torch.eye(nS, dtype=torch.float32)
    title_x = torch.eye(nT, dtype=torch.float32)
    # constant company feature (a degree feat would leak future exposure count)
    company_x = torch.ones(nC, 1, dtype=torch.float32)

    # frozen-encoder arm (seeker+job, node-aligned) = sole node-content source
    seeker_parts = [_qwen_tech_emb("seeker", nm.n_seeker, fc.text_emb)]
    job_parts = [_qwen_tech_emb("job", nm.n_job, fc.text_emb)]
    txt_seeker, txt_job = seeker_parts[0], job_parts[0]   # pure text emb (equal dims)
    if fc.entity == "tabular_textemb" and fc.use_tabular:
        seeker_parts.append(_seeker_tabular(nm))
        job_parts.append(_job_tabular(nm))
    seeker_x = torch.cat(seeker_parts, dim=1)
    job_x = torch.cat(job_parts, dim=1)

    text_raw = None
    # TEXT modes + match: hold match text channel at pure text emb (equal dims)
    if fc.entity in TEXT_MODES and getattr(fc, "use_match_feats", False):
        text_raw = {"seeker": txt_seeker.contiguous().float(),
                    "job": txt_job.contiguous().float()}
    if getattr(fc, "content_off", False):              # drop content: seeker/job -> constant
        # text_raw keeps the text emb, so match cosine survives content_off
        seeker_x = torch.ones(nM, 1, dtype=torch.float32)
        job_x = torch.ones(nJ, 1, dtype=torch.float32)
    x = {"seeker": seeker_x, "job": job_x, "skill": skill_x, "title": title_x, "company": company_x}
    # ensure float32 contiguous
    x = {k: v.contiguous().float() for k, v in x.items()}
    in_dims = {k: v.shape[1] for k, v in x.items()}
    return FeatureBundle(x=x, in_dims=in_dims, text=text_raw)
