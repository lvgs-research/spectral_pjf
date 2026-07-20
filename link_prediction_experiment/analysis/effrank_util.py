"""Shared effrank utilities: participation-ratio effective rank + per-pair raw/pre-decoder embedding extraction."""
from __future__ import annotations

import numpy as np
import torch

from .. import paths
from .embedding_link_similarity import _eff_rank


# ---------------------------------------------------------------------------
# Effective rank (participation ratio of the mean-centred spectrum)
# ---------------------------------------------------------------------------

def eff_rank(X, eps=1e-9):
    """Participation ratio of the mean-centred singular spectrum (fast Gram route for tall X, SVD fallback for wide)."""
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] < 3 or X.shape[1] == 0:
        return 0.0
    Xc = X - X.mean(0, keepdims=True)
    n, d = Xc.shape
    if d <= n and d <= 4096:                       # tall -> Gram route
        lam = np.linalg.eigvalsh(Xc.T @ Xc)
        s = np.sqrt(np.clip(lam, 0.0, None))
    else:
        s = np.linalg.svd(Xc, compute_uv=False)
    s = s[s > eps]
    return float((s.sum() ** 2) / (s ** 2).sum()) if s.size else 0.0


def _assert_effrank_matches():
    """Assert the Gram path equals the canonical SVD eff-rank on full/deficient/wide matrices."""
    rng = np.random.default_rng(0)
    for name, X in [("full", rng.standard_normal((800, 40))),
                    ("deficient", rng.standard_normal((800, 5)) @ rng.standard_normal((5, 40))),
                    ("wide", rng.standard_normal((30, 200)))]:
        a, b = eff_rank(X), _eff_rank(X)
        assert abs(a - b) <= 1e-6 * max(1.0, b), f"eff_rank mismatch [{name}]: {a} vs {b}"


_assert_effrank_matches()


# ---------------------------------------------------------------------------
# Per-pair raw inputs + pre-decoder embeddings (1-day exact PIT for hetero; static for MLP)
# ---------------------------------------------------------------------------

def per_pair_raw_emb(tr, models, m, j, ts):
    """Returns (Xm, Xj, {name: (Zm, Zj)}) in original pair order: raw encoder inputs and pre-decoder embeddings (1-day exact PIT for hetero, static for MLP)."""
    m = np.asarray(m); j = np.asarray(j)
    if tr.mode == "mlp":
        x = tr.feats.x
        xm = x["seeker"].detach().cpu().numpy(); xj = x["job"].detach().cpu().numpy()
        out = {}
        with torch.no_grad():
            for name, model in models.items():
                model.eval()
                z = model.encode(x)
                out[name] = (z["seeker"].detach().cpu().numpy()[m], z["job"].detach().cpu().numpy()[j])
        return xm[m], xj[j], out

    ts = np.asarray(ts, dtype=np.int64)
    day = (ts // 86400) * 86400                                   # left-edge daily cutoff (== eval)
    order = np.argsort(day, kind="stable")
    ms, js, ds = m[order], j[order], day[order]
    uniq, starts = np.unique(ds, return_index=True)
    bounds = list(starts) + [len(ds)]
    is_mps = tr.device.type == "mps"
    rawm, rawj = [], []
    parts = {name: {"m": [], "j": []} for name in models}
    for model in models.values():
        model.eval()
    with torch.no_grad():
        for gi in range(len(uniq)):
            a, b = bounds[gi], bounds[gi + 1]
            cutoff = int(uniq[gi]); gm, gj = ms[a:b], js[a:b]
            edges = tr._edges(cutoff, np.unique(gm), np.unique(gj))   # PIT, focal = today's entities
            aug = tr._aug_feats(edges)                                # raw node inputs (PIT-degree if enabled)
            im = torch.as_tensor(gm, dtype=torch.long, device=tr.device)
            ij = torch.as_tensor(gj, dtype=torch.long, device=tr.device)
            rawm.append(aug["seeker"][im].cpu().numpy())
            rawj.append(aug["job"][ij].cpu().numpy())
            for name, model in models.items():
                z = model.encode(aug, edges)
                parts[name]["m"].append(z["seeker"][im].cpu().numpy())
                parts[name]["j"].append(z["job"][ij].cpu().numpy())
                del z
            del edges, aug, im, ij
            if is_mps:
                torch.mps.empty_cache()
            if (gi + 1) % 25 == 0 or gi == len(uniq) - 1:
                print(f"    day {gi + 1}/{len(uniq)} (cutoff {cutoff})", flush=True)
    inv = np.empty_like(order); inv[order] = np.arange(len(m))
    Xm = np.concatenate(rawm, 0)[inv]; Xj = np.concatenate(rawj, 0)[inv]
    out = {name: (np.concatenate(p["m"], 0)[inv], np.concatenate(p["j"], 0)[inv])
           for name, p in parts.items()}
    return Xm, Xj, out
