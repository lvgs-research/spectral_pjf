"""Rank-based read of content-vs-graph modality usage + eff-rank along the decode pipeline.

Run:  python -m link_prediction_experiment.analysis.effrank_modality_decomposition --datasets tech,tianchi
"""
from __future__ import annotations

import numpy as np
import torch

from ..models import _combine
from .effrank_util import eff_rank


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def eff_rank_std(X, eps=1e-9):
    """Column-standardised effective rank (scale-invariant correlation participation ratio)."""
    X = np.asarray(X, np.float64)
    if X.shape[0] < 3 or X.shape[1] == 0:
        return 0.0
    return eff_rank(X / (X.std(0, keepdims=True) + eps))


def pre_logit(decoder, Zm, Zj, pf):
    """Per-head pre-logit hidden layer = head[:-1](concat[z_m, z_j (, match_feats)])."""
    zm = torch.as_tensor(Zm, dtype=torch.float32); zj = torch.as_tensor(Zj, dtype=torch.float32)
    comb = _combine(zm, zj, getattr(decoder, "head_combine", "concat"))
    base = [comb] if pf is None else [comb, pf.detach().cpu().float()]
    if getattr(decoder, "esmm_pair_diff", False):
        fa = torch.cat(base + [zj - zm], dim=-1); fc = torch.cat(base + [zm - zj], dim=-1)
    else:
        fa = fc = torch.cat(base, dim=-1)
    decoder.eval()
    with torch.no_grad():
        pa = decoder.head_accept[:-1](fa).numpy()
        pc = decoder.head_cond[:-1](fc).numpy()
    return pa, pc
