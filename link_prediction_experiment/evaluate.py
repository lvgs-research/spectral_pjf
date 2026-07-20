"""Evaluation on the real exposed pos/neg, sliced by exposure count cohort.

Reports exposed AUC and seeker/job-grouped GAUC + MAP@k / Hit@k.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from .config import ExperimentConfig


def _safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def calibration_stats(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    """Calibration of probabilities ``p`` vs binary outcomes ``y`` (Brier, ECE/MCE, calib_gap)."""
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) < 2 or y.sum() == 0 or y.sum() == len(y):
        return {"n": int(len(y)), "brier": float("nan"), "ece": float("nan"),
                "mce": float("nan"), "base_rate": (float(y.mean()) if len(y) else float("nan")),
                "mean_pred": (float(p.mean()) if len(p) else float("nan")),
                "calib_gap": float("nan"), "bins": []}
    brier = float(np.mean((p - y) ** 2))
    base = float(y.mean()); mean_pred = float(p.mean())
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins, ece, mce = [], 0.0, 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        sel = (p >= lo) & ((p < hi) if b < n_bins - 1 else (p <= hi))
        if sel.sum() == 0:
            continue
        conf = float(p[sel].mean()); acc = float(y[sel].mean()); w = int(sel.sum()) / len(y)
        gap = abs(acc - conf)
        ece += w * gap; mce = max(mce, gap)
        bins.append({"lo": float(lo), "hi": float(hi), "n": int(sel.sum()),
                     "mean_pred": conf, "frac_pos": acc})
    return {"n": int(len(y)), "brier": brier, "ece": float(ece), "mce": float(mce),
            "base_rate": base, "mean_pred": mean_pred,
            "calib_gap": float(mean_pred - base), "bins": bins}


def _by_bucket(metric, y: np.ndarray, s: np.ndarray, bucket: np.ndarray) -> Dict[str, float]:
    """Apply a (y, s) -> float metric over all rows + each exposure count bucket + cold/warm."""
    out = {"all": metric(y, s)}
    for b in ["0", "1-2", "3-5", "6-20", "21+"]:
        mask = bucket == b
        if mask.sum() >= 10:
            out[b] = metric(y[mask], s[mask])
    cold = bucket == "0"
    if cold.sum() >= 10:
        out["cold_job"] = metric(y[cold], s[cold])
    if (~cold).sum() >= 10:
        out["warm_job"] = metric(y[~cold], s[~cold])
    return out


def _auc_by_bucket(y: np.ndarray, s: np.ndarray, bucket: np.ndarray) -> Dict[str, float]:
    return _by_bucket(_safe_auc, y, s, bucket)


def _ranking_metrics(rel_sorted: np.ndarray, ks: tuple, total_rel: int) -> Dict[str, float]:
    out = {}
    for k in ks:
        topk = rel_sorted[:k]
        hits = float(topk.sum())
        out[f"P@{k}"] = hits / k
        out[f"R@{k}"] = hits / max(total_rel, 1)
        # average precision@k
        if total_rel == 0:
            out[f"MAP@{k}"] = 0.0
            continue
        precs, nh = [], 0
        for i in range(min(k, len(rel_sorted))):
            if rel_sorted[i]:
                nh += 1
                precs.append(nh / (i + 1))
        out[f"MAP@{k}"] = (sum(precs) / min(total_rel, k)) if precs else 0.0
    return out


def evaluate(trainer, df: pl.DataFrame, cfg: ExperimentConfig, split: str,
             exact: bool = False) -> dict:
    """Score the split's exposed pairs and return AUC + grouped-ranking metrics."""
    sdf = df.filter(pl.col("dataset_split") == split)
    nce = cfg.train.n_time_chunks_eval
    rng = np.random.default_rng(cfg.train.seed + 7)
    res: dict = {"split": split, "exact_pit": bool(exact)}

    # scorer: exact per-row PIT or chunked
    def sp(m, j, t):
        return trainer.score_pairs_exact(m, j, t) if exact else trainer.score_pairs(m, j, t, nce)

    # ESMM per-head probs (pa, pc) in one scoring pass
    def sph(m, j, t):
        return (trainer.score_pairs_exact(m, j, t, return_heads=True) if exact
                else trainer.score_pairs(m, j, t, nce, return_heads=True))
    is_esmm = getattr(trainer, "_esmm_arch", False)

    # exposed AUC (passed vs exposed-not-passed)
    em = sdf["seeker_idx"].to_numpy(); ej = sdf["job_idx"].to_numpy()
    ets = sdf["exposure_ts"].to_numpy(); eby = sdf["passed"].to_numpy().astype(float)
    ebk = sdf["exposure_bucket"].to_numpy()
    cap = cfg.train.max_eval_exposed
    if cap and len(em) > cap:                  # bound exact-PIT exposed forwards
        sel = rng.choice(len(em), cap, replace=False)
        em, ej, ets, eby, ebk = em[sel], ej[sel], ets[sel], eby[sel], ebk[sel]
    # ESMM served joint = P(accept)*P(pass|accept)
    if is_esmm:
        pa_e, pc_e = sph(em, ej, ets)
        se = (pa_e * pc_e).astype(np.float32)
    else:
        se = sp(em, ej, ets)
    res["auc_exposed"] = _auc_by_bucket(eby, se, ebk)

    res["exposed_ranking"] = _exposed_per_user_ranking(em, eby, se, cfg.ranking_ks)
    res["job_ranking"] = _exposed_per_job_ranking(ej, eby, se, cfg.ranking_ks)
    # seeker-side grouped ranking (== exposed_ranking.per_user_AUC by construction)
    res["user_grouped_ranking"] = _exposed_per_group_ranking(em, eby, se, cfg.ranking_ks, group_name="seekers")
    return res


def _exposed_per_user_ranking(seeker_idx: np.ndarray, passed: np.ndarray,
                              scores: np.ndarray, ks: tuple) -> dict:
    """Per-user re-ranking of their exposed set: mean per-user AUC + MAP@k/P@k/R@k."""
    order = {}
    for i, u in enumerate(seeker_idx):
        order.setdefault(int(u), []).append(i)
    aucs = []
    agg = {f"{m}@{k}": [] for m in ("MAP", "P", "R") for k in ks}
    n_users_ranked = 0
    for u, idxs in order.items():
        idxs = np.asarray(idxs)
        rel = passed[idxs].astype(int)
        if rel.sum() == 0 or len(idxs) < 2:
            continue                      # need >=1 pos and >=2 candidates
        n_users_ranked += 1
        sc = scores[idxs]
        if 0 < rel.sum() < len(rel):      # both classes -> per-user AUC defined
            aucs.append(_safe_auc(rel, sc))
        rel_sorted = rel[np.argsort(-sc, kind="stable")]
        for kk, v in _ranking_metrics(rel_sorted, ks, int(rel.sum())).items():
            agg[kk].append(v)
    out = {f"per_user_AUC": float(np.mean(aucs)) if aucs else float("nan"),
           "n_users_ranked": n_users_ranked}
    out.update({k: (float(np.mean(v)) if v else float("nan")) for k, v in agg.items()})
    return out


def _exposed_per_group_ranking(group_idx: np.ndarray, passed: np.ndarray,
                               scores: np.ndarray, ks: tuple, group_name: str = "jobs") -> dict:
    """Per-group exposed ranking: GAUC (unweighted mean of per-group AUC), MAP@k, Hit@k."""
    order: dict = {}
    for i, j in enumerate(group_idx):
        order.setdefault(int(j), []).append(i)
    aucs, aw = [], []                                  # per-group AUC + exposure weight
    agg = {f"MAP@{k}": [] for k in ks}
    agg.update({f"Hit@{k}": [] for k in ks})
    n_ranked = 0
    for j, idxs in order.items():
        idxs = np.asarray(idxs)
        rel = passed[idxs].astype(int)
        if rel.sum() == 0 or len(idxs) < 2:
            continue                                   # need >=1 pos and >=2 candidates
        n_ranked += 1
        sc = scores[idxs]
        if 0 < rel.sum() < len(rel):                   # both classes -> per-group AUC defined
            aucs.append(_safe_auc(rel, sc)); aw.append(len(idxs))
        rel_sorted = rel[np.argsort(-sc, kind="stable")]
        rm = _ranking_metrics(rel_sorted, ks, int(rel.sum()))
        for k in ks:
            agg[f"MAP@{k}"].append(rm[f"MAP@{k}"])
            agg[f"Hit@{k}"].append(1.0 if rel_sorted[:k].sum() > 0 else 0.0)
    out = {"GAUC": float(np.mean(aucs)) if aucs else float("nan"),                       # unweighted group-mean
           "GAUC_expw": float(np.average(aucs, weights=aw)) if aucs else float("nan"),   # legacy exposure-weighted
           f"n_{group_name}_ranked": n_ranked}
    out.update({k: (float(np.mean(v)) if v else float("nan")) for k, v in agg.items()})
    return out


def _exposed_per_job_ranking(job_idx: np.ndarray, passed: np.ndarray,
                             scores: np.ndarray, ks: tuple) -> dict:
    """Job-grouped ranking (item side): wrapper over _exposed_per_group_ranking."""
    return _exposed_per_group_ranking(job_idx, passed, scores, ks, group_name="jobs")
