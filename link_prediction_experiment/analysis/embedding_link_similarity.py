"""Shared effective-rank helper + results directory for the effrank / PC analysis pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "_results"


def _eff_rank(X):
    s = np.linalg.svd(X - X.mean(0, keepdims=True), compute_uv=False)
    s = s[s > 1e-9]
    return float((s.sum() ** 2) / (s ** 2).sum()) if s.size else 0.0
