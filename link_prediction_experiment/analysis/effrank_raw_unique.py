"""Raw-input effective-rank baseline: eff-rank of the node features fed to model.encode (no GNN forward, seed/conv-invariant).

Run:
  python -m link_prediction_experiment.analysis.effrank_raw_unique
Writes tables/data/effrank_raw_unique.json
"""
from __future__ import annotations
import os, dataclasses
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import polars as pl
import torch
torch.set_num_threads(4)

from .. import paths
from ..config import FeatureConfig
from ..train import Trainer
from .effrank_util import per_pair_raw_emb
from .effrank_trajectory import _tech_shared, rebuild_cfg, _tc_ctx, Tech, TC

EXPS = ["p1", "p2", "p3", "p8"]


def _u(M, ids):
    _, first = np.unique(np.asarray(ids), return_index=True)
    return M[np.sort(first)]


def raw_tech(exp):
    shared = _tech_shared()
    arm, tag = Tech["sage"][exp]
    st = torch.load(paths.CKPT_DIR / tag / f"{arm}_seed0.pt", weights_only=False, map_location="cpu")
    base = rebuild_cfg(st["config"])
    cfg = dataclasses.replace(base, train=dataclasses.replace(base.train, device="cpu", seed=0))
    feats = shared.features(cfg)
    tr = Trainer(cfg, shared.gstore, feats, shared.df, verbose=False)   # weights unused (raw only)
    test = shared.df.filter(pl.col("dataset_split") == "test")
    m = test["seeker_idx"].to_numpy(); j = test["job_idx"].to_numpy(); ts = test["exposure_ts"].to_numpy().astype(np.int64)
    Xm, Xj, _ = per_pair_raw_emb(tr, {}, m, j, ts)                            # {} models -> no encode
    return m, j, Xm, Xj


def raw_tc(exp):
    from ..tianchi_pjf.experiment import make_x, _load_qwen_content, NODE_TYPES
    from ..tianchi_pjf.hybrid import build_visibility, _tc_degree_block
    art, sp, test = _tc_ctx()
    arm, tag = TC["sage"][exp]
    st = torch.load(paths.PKG_DIR / "tianchi_pjf" / "_checkpoints" / tag / f"{arm}_seed0.pt",
                    weights_only=False, map_location="cpu")
    fcfg = FeatureConfig(**st["feature_cfg"])
    m = test["seeker_idx"].to_numpy(); j = test["job_idx"].to_numpy()
    qwen = (_load_qwen_content(art.num_nodes, getattr(fcfg, "emb_model", "qwen3"))
            if getattr(fcfg, "use_qwen_content", False) else None)
    x = make_x(art, fcfg, "cpu", qwen=qwen)
    if getattr(fcfg, "tc_ones_collapse", False):
        for t in ("seeker", "job"):
            x[t] = torch.ones(x[t].size(0), 1, dtype=x[t].dtype)
    ei = build_visibility(art, sp, "test", fcfg, "cpu")
    if getattr(fcfg, "tc_degree_nodes", False):
        deg = _tc_degree_block(ei, art.num_nodes, fcfg, torch.device("cpu"))
        x = {t: torch.cat([x[t], deg[t]], dim=1) for t in NODE_TYPES}
    Xm = x["seeker"].numpy()[m]; Xj = x["job"].numpy()[j]
    return m, j, Xm, Xj
