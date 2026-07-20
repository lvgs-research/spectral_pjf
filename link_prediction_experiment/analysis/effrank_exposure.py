"""Leak-safe PIT targets (exposure count + per-edge-type degree) for the PC-direction analysis, both datasets."""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
import polars as pl
import torch
torch.set_num_threads(4)

from .. import paths
from .effrank_trajectory import _tech_shared, _tc_ctx

NB_DATA = paths.PKG_DIR.parent / "tables" / "data"
EMB = NB_DATA / "embeddings"
EMB_ENC = NB_DATA / "embeddings_encoders"


def _pearson(a, b):
    a = np.asarray(a, float) - np.mean(a); b = np.asarray(b, float) - np.mean(b)
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def _spearman(a, b):
    return _pearson(pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy())


def tech_exposure():
    """Tech leak-safe historical exposure degree from train+val (strictly precede test)."""
    df = _tech_shared().df
    assert df.filter(pl.col("dataset_split").is_in(["train", "val"]))["exposure_ts"].max() < \
           df.filter(pl.col("dataset_split") == "test")["exposure_ts"].min(), "train/val NOT before test!"
    hist = df.filter(pl.col("dataset_split").is_in(["train", "val"]))
    nm = int(df["seeker_idx"].max()) + 1; nj = int(df["job_idx"].max()) + 1
    return (np.bincount(hist["seeker_idx"].to_numpy(), minlength=nm),
            np.bincount(hist["job_idx"].to_numpy(), minlength=nj))


def tc_exposure():
    """Tianchi leak-safe historical degree from train-split pairs (test supervision excluded)."""
    _, sp, _ = _tc_ctx()
    tr = sp.filter(pl.col("hsplit") == "train")
    nm = int(sp["seeker_idx"].max()) + 1; nj = int(sp["job_idx"].max()) + 1
    return (np.bincount(tr["seeker_idx"].to_numpy(), minlength=nm),
            np.bincount(tr["job_idx"].to_numpy(), minlength=nj))


# Per-edge-type degree input features (leak-safe, PIT, arm/conv/seed-independent).
def _assemble(int_specs, attr_specs, nn):
    """Build per-edge-type degree targets from forward edge arrays (interaction -> in/out, attribute -> out)."""
    focal = ("seeker", "job"); SIDE = {"seeker": "seeker", "job": "job"}
    tg = {"seeker": {}, "job": {}}; keys = []; per_side = {"seeker": [], "job": []}; labels = {}
    for r, s_t, d_t, ei in int_specs:
        indeg = {t: np.zeros(nn[t]) for t in focal}
        outdeg = {t: np.zeros(nn[t]) for t in focal}
        if ei.shape[1]:
            if s_t in outdeg: outdeg[s_t] += np.bincount(ei[0], minlength=nn[s_t])
            if d_t in indeg:  indeg[d_t]  += np.bincount(ei[1], minlength=nn[d_t])
        for direc, degd, owner in (("in", indeg, d_t), ("out", outdeg, s_t)):
            key = f"{r}_{direc}"; keys.append(key); labels[key] = f"{r} {direc}-degree"
            tg["seeker"][key] = degd["seeker"]; tg["job"][key] = degd["job"]
            if owner in focal:                        # side this direction is non-degenerate on
                per_side[SIDE[owner]].append(key)
    for r, s_t, ei in attr_specs:                     # entity is the src
        d = np.bincount(ei[0], minlength=nn[s_t]).astype(float) if ei.shape[1] else np.zeros(nn[s_t])
        key = r; keys.append(key); labels[key] = f"{r} degree"
        tg["seeker"][key] = d if s_t == "seeker" else np.zeros(nn["seeker"])
        tg["job"][key]    = d if s_t == "job" else np.zeros(nn["job"])
        per_side[SIDE[s_t]].append(key)
    return tg, keys, per_side, labels


def tech_degree_feats():
    """Tech per-edge-type leak-safe degree targets (interaction over train+val < first test ts; attribute profile)."""
    from ..graph_build import INTERACTION_RELS, ATTRIBUTE_RELS
    s = _tech_shared(); g = s.gstore; nn = g.num_nodes
    tmin = s.df.filter(pl.col("dataset_split") == "test")["exposure_ts"].min()
    int_specs = []
    for rel in INTERACTION_RELS:
        s_t, r, d_t = rel
        k = int(np.searchsorted(g.int_time[rel], tmin, side="left"))   # train+val edges only
        int_specs.append((r, s_t, d_t, g.int_fwd[rel][:, :k].cpu().numpy()))
    attr_specs = [(rel[1], rel[0], g.attr_fwd[rel].cpu().numpy()) for rel in ATTRIBUTE_RELS]
    return _assemble(int_specs, attr_specs, nn)


def tc_degree_feats():
    """Tianchi per-edge-type leak-safe degree targets (accepts/satisfied over train-split; skill+title attributes)."""
    from ..tianchi_pjf.experiment import ATTR_RELS
    art, sp, _ = _tc_ctx(); nn = art.num_nodes
    tr = sp.filter(pl.col("hsplit") == "train")
    m = tr["seeker_idx"].to_numpy(); j = tr["job_idx"].to_numpy()
    dv = tr["delivered"].to_numpy() == 1; st = tr["satisfied"].to_numpy() == 1
    int_specs = [("accepts", "seeker", "job", np.stack([m[dv], j[dv]])),
                 ("satisfied", "seeker", "job", np.stack([m[st], j[st]]))]
    attr_specs = [(rel[1], rel[0], art.edges[rel].cpu().numpy()) for rel in ATTR_RELS]
    return _assemble(int_specs, attr_specs, nn)
