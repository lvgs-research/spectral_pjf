r"""Decompose what PC1 of the unique-entity pre-decoder cloud encodes (vs exposure count/degree/norm/score).

Run:  python -m link_prediction_experiment.analysis.pc1_semantics
Writes tables/data/pc1_semantics.json + prints the summary tables.
"""
from __future__ import annotations
import os, json, re
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import torch
torch.set_num_threads(4)

from .. import paths
from .effrank_exposure import (EMB, EMB_ENC, tech_exposure, tc_exposure, tech_degree_feats, tc_degree_feats,
                                 _pearson, _spearman)
from .effrank_trajectory import Tech, TC, _ENC_TAGS_TECH, _ENC_TAGS_TC, ENC_SLUG

OUT = paths.PKG_DIR.parent / "tables" / "data" / "pc1_semantics.json"
SEEDS = [0, 1, 2, 3, 4]
ARMS = ["p1", "p2", "p3", "p8"]


def _variants(ds):
    """(conv, encoder, npz_path_fn, ckpt_fn) per analysis axis (conv + encoder sweep)."""
    reg = Tech if ds == "tech" else TC
    enc_tags = _ENC_TAGS_TECH if ds == "tech" else _ENC_TAGS_TC
    ckroot = (lambda tag: paths.CKPT_DIR / tag) if ds == "tech" else \
             (lambda tag: paths.PKG_DIR / "tianchi_pjf" / "_checkpoints" / tag)
    out = []
    for conv in ("sage", "gatv2"):
        def npz(arm, sd, conv=conv):
            return EMB / ds / f"{conv}__{arm}__seed{sd}.npz"
        def ck(arm, sd, conv=conv):
            arms = reg[conv] if arm in reg[conv] else reg["sage"]   # gatv2 p1 == sage p1
            name, tag = arms[arm]
            return ckroot(tag) / f"{name}_seed{sd}.pt"
        out.append((conv, "qwen3-0.6b", npz, ck))
    for enc in ("me5", "qwen3_4b", "qwen3_8b"):
        slug = ENC_SLUG[enc]
        def npz(arm, sd, slug=slug):
            return EMB_ENC / ds / f"sage__{slug}__{arm}__seed{sd}.npz"
        def ck(arm, sd, enc=enc):
            name, tag = enc_tags[enc][arm]
            return ckroot(tag) / f"{name}_seed{sd}.pt"
        out.append(("sage", slug, npz, ck))
    return out


# --------------------------------------------------------------------------- helpers
def _svd_unique(Z, ids):
    """Dedup (first occurrence) -> centered SVD; returns (uids, U[:,:3], V[:,:3], mean, M, s)."""
    uids, first = np.unique(np.asarray(ids), return_index=True)
    M = np.asarray(Z, np.float64)[first]
    mu = M.mean(0)
    U, s, Vt = np.linalg.svd(M - mu, full_matrices=False)
    return uids, U[:, :3], Vt[:3].T, mu, M, s


def _ols_r2(y, X):
    """R^2 of y ~ [1, X] (columns standardized; degenerate columns dropped)."""
    y = np.asarray(y, np.float64)
    X = np.asarray(X, np.float64)
    if X.ndim == 1:
        X = X[:, None]
    keep = X.std(0) > 1e-12
    if not keep.any() or y.std() < 1e-12:
        return 0.0
    Xs = (X[:, keep] - X[:, keep].mean(0)) / X[:, keep].std(0)
    A = np.column_stack([np.ones(len(y)), Xs])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    return float(1.0 - resid.var() / y.var())


def _partial_r(y, x, C):
    """|partial Pearson| of y~x controlling for columns C (residualize both on [1,C])."""
    C = np.asarray(C, np.float64)
    if C.ndim == 1:
        C = C[:, None]
    keep = C.std(0) > 1e-12
    A = np.column_stack([np.ones(len(y))] + ([C[:, keep]] if keep.any() else []))
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    return abs(_pearson(ry, rx)) if ry.std() > 1e-12 and rx.std() > 1e-12 else 0.0


_LAST = re.compile(r"^(?:model\.)?decoder\.head_(accept|cond)\.(\d+)\.weight$")


def _last_linears(state):
    """(W_a, b_a, W_c, b_c) of the decoder heads' LAST Linear from a checkpoint state dict."""
    idx = {"accept": -1, "cond": -1}
    for k in state:
        m = _LAST.match(k)
        if m:
            idx[m.group(1)] = max(idx[m.group(1)], int(m.group(2)))
    if idx["accept"] < 0 or idx["cond"] < 0:
        return None
    g = lambda h, i, p: state[[k for k in state if k.endswith(f"decoder.head_{h}.{i}.{p}")][0]]
    return tuple(np.asarray(g(h, idx[h], p).cpu(), np.float64)
                 for h in ("accept", "cond") for p in ("weight", "bias"))


def _mean_score(npz, ckpt_state, side_idx):
    """Per-entity mean joint logit log sig(a)+log sig(c) from cached pre-logits x last Linear."""
    ll = _last_linears(ckpt_state)
    if ll is None:
        return None
    Wa, ba, Wc, bc = ll
    pa, pc = npz["prelogit_accept"].astype(np.float64), npz["prelogit_pass"].astype(np.float64)
    if pa.shape[1] != Wa.shape[1] or pc.shape[1] != Wc.shape[1]:
        return None
    a = pa @ Wa.ravel() + float(np.ravel(ba)[0])
    c = pc @ Wc.ravel() + float(np.ravel(bc)[0])
    s = np.log1p(np.exp(-np.abs(a))) * 0 - np.logaddexp(0, -a) - np.logaddexp(0, -c)  # logsig(a)+logsig(c)
    ids = np.asarray(npz[side_idx])
    uids, inv = np.unique(ids, return_inverse=True)
    tot = np.bincount(inv, weights=s); cnt = np.bincount(inv)
    return uids, tot / cnt


def analyze_run(npz, ckpt_state, side, emb_key, idx_key, exposure, deg, per_side_keys):
    uids, U3, V3, mu, M, s = _svd_unique(npz[emb_key], npz[idx_key])
    out = {"n_unique": int(len(uids))}
    # PC1 energy: variance fraction of the top eigenvalues
    lam = s ** 2
    out["pc1_energy"] = float(lam[0] / lam.sum())                 # PC1 share of variance
    out["top3_energy"] = float(lam[:3].sum() / lam.sum())          # cumulative top-3 share
    # top-variance axis vs DC axis; PC1 vs norm
    mu_n = mu / (np.linalg.norm(mu) + 1e-12)
    out["cos_v1_mean"] = abs(float(V3[:, 0] @ mu_n))
    norm = np.linalg.norm(M, axis=1)
    # targets (log1p degrees; norm & score as-is)
    T = {"exposure": np.log1p(exposure[uids])}
    int_keys = [k for k in per_side_keys if k.endswith("_in") or k.endswith("_out")]
    attr_keys = [k for k in per_side_keys if not (k.endswith("_in") or k.endswith("_out")) and k != "exposure count"]
    for k in int_keys + attr_keys:
        T[k] = np.log1p(deg[side][k][uids] if isinstance(deg, dict) and side in deg else deg[k][uids])
    T["attr_total"] = np.log1p(sum(np.expm1(T[k]) for k in attr_keys)) if attr_keys else np.zeros(len(uids))
    T["int_total"] = np.log1p(sum(np.expm1(T[k]) for k in int_keys)) if int_keys else np.zeros(len(uids))
    T["norm"] = norm
    ms = _mean_score(npz, ckpt_state, idx_key)
    if ms is not None:
        s_uids, s_val = ms
        assert np.array_equal(s_uids, uids)
        T["score"] = s_val
    # per-PC correlations
    corr = {}
    for k, v in T.items():
        if np.std(v) < 1e-12:
            corr[k] = {"r": [0.0] * 3, "rho": [0.0] * 3}
            continue
        corr[k] = {"r":   [abs(_pearson(U3[:, j], v)) for j in range(U3.shape[1])],
                   "rho": [abs(_spearman(U3[:, j], v)) for j in range(U3.shape[1])]}
    out["corr"] = corr
    # variance decomposition of u1
    u1 = U3[:, 0]
    Xint = np.column_stack([T[k] for k in int_keys]) if int_keys else np.zeros((len(uids), 0))
    Xattr = np.column_stack([T[k] for k in attr_keys]) if attr_keys else np.zeros((len(uids), 0))
    out["r2"] = {"int_set": _ols_r2(u1, Xint) if Xint.shape[1] else 0.0,
                 "attr_set": _ols_r2(u1, Xattr) if Xattr.shape[1] else 0.0,
                 "both": _ols_r2(u1, np.column_stack([Xint, Xattr])) if (Xint.shape[1] + Xattr.shape[1]) else 0.0,
                 "exposure": _ols_r2(u1, T["exposure"])}
    out["partial"] = {
        "exposure_given_attr": _partial_r(u1, T["exposure"], Xattr) if Xattr.shape[1] else abs(_pearson(u1, T["exposure"])),
        "attr_given_int": _partial_r(u1, T["attr_total"], Xint) if Xint.shape[1] else abs(_pearson(u1, T["attr_total"])),
    }
    return out


# --------------------------------------------------------------------------- raw baselines
def raw_baseline():
    """u1 stats of the RAW p1 content inputs (learned-vs-inherited reference)."""
    res = {}
    # ---- Tech
    import dataclasses, polars as pl
    from .effrank_trajectory import _tech_shared, rebuild_cfg
    st = torch.load(paths.CKPT_DIR / "paired_tech_v2" / "p1_mlp_seed0.pt", weights_only=False, map_location="cpu")
    shared = _tech_shared()
    cfg = dataclasses.replace(rebuild_cfg(st["config"]),
                              train=dataclasses.replace(rebuild_cfg(st["config"]).train, device="cpu"))
    X = shared.features(cfg).x
    test = shared.df.filter(pl.col("dataset_split") == "test")
    m_exp, j_exp = tech_exposure()
    tg, keys, per_side, _ = tech_degree_feats()
    for side, t, idx in (("seeker", "seeker", "seeker_idx"), ("job", "job", "job_idx")):
        ids = test[idx].to_numpy()
        uids, first = np.unique(ids, return_index=True)
        M = X[t].numpy().astype(np.float64)[uids]
        U, s, Vt = np.linalg.svd(M - M.mean(0), full_matrices=False)
        exposure = (m_exp if side == "seeker" else j_exp)[uids]
        stats = {"exposure": abs(_pearson(U[:, 0], np.log1p(exposure)))}
        for k in per_side[side]:
            stats[k] = abs(_pearson(U[:, 0], np.log1p(tg[side][k][uids])))
        res[f"tech_{side}"] = stats
    # ---- TC
    from .effrank_trajectory import _tc_ctx
    from ..tianchi_pjf.experiment import make_x, _load_qwen_content
    from ..config import FeatureConfig, ModelConfig
    art, sp, test = _tc_ctx()
    stt = torch.load(paths.PKG_DIR / "tianchi_pjf" / "_checkpoints" / "paired_tc_ctrl" / "p1_ctrl_seed0.pt",
                     weights_only=False, map_location="cpu")
    fcfg = FeatureConfig(**stt["feature_cfg"])
    qwen = _load_qwen_content(art.num_nodes, getattr(fcfg, "emb_model", "qwen3")) if getattr(fcfg, "use_qwen_content", False) else None
    x = make_x(art, fcfg, "cpu", qwen=qwen)
    m_exp, j_exp = tc_exposure()
    tg, keys, per_side, _ = tc_degree_feats()
    for side, t, col in (("seeker", "seeker", "seeker_idx"), ("job", "job", "job_idx")):
        ids = test[col].to_numpy()
        uids = np.unique(ids)
        M = x[t].numpy().astype(np.float64)[uids]
        U, s, Vt = np.linalg.svd(M - M.mean(0), full_matrices=False)
        exposure = (m_exp if side == "seeker" else j_exp)[uids]
        stats = {"exposure": abs(_pearson(U[:, 0], np.log1p(exposure)))}
        for k in per_side[side]:
            stats[k] = abs(_pearson(U[:, 0], np.log1p(tg[side][k][uids])))
        res[f"tianchi_{side}"] = stats
    return res


# --------------------------------------------------------------------------- driver
def main():
    exposures = {"tech": tech_exposure(), "tianchi": tc_exposure()}
    degs = {"tech": tech_degree_feats(), "tianchi": tc_degree_feats()}
    records = []
    for ds in ("tech", "tianchi"):
        m_exp, j_exp = exposures[ds]
        tg, keys, per_side, labels = degs[ds]
        for conv, enc, npz_fn, ck_fn in _variants(ds):
            n0 = len(records)
            for arm in ARMS:
                for sd in SEEDS:
                    fp = npz_fn(arm, sd)
                    if not fp.exists():
                        continue
                    d = np.load(fp)
                    ckp = ck_fn(arm, sd)
                    state = (torch.load(ckp, weights_only=False, map_location="cpu")["best_state"]
                             if ckp.exists() else {})            # missing ckpt -> score target skipped
                    for side, ek, ik, exposure in (("seeker", "predec_seeker", "seeker_idx", m_exp),
                                              ("job", "predec_job", "job_idx", j_exp)):
                        rec = analyze_run(d, state, side, ek, ik, exposure, tg, per_side[side])
                        rec.update(dataset=ds, arm=arm, seed=sd, side=side, conv=conv, encoder=enc)
                        records.append(rec)
            print(f"  {ds} {conv}/{enc}: {len(records) - n0} recs", flush=True)
    raw = raw_baseline()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"records": records, "raw_baseline": raw,
               "doc": "u1..u3 = top principal coords of unique-entity pre-decoder cloud; targets log1p; "
                      "score = mean per-entity joint logit from cached pre-logits x ckpt last Linear; "
                      "axes: conv in {sage,gatv2} (0.6b; gatv2 p1 == sage p1 copy), encoder in "
                      "{qwen3-0.6b,me5,qwen3-4b,qwen3-8b} (sage)"},
              open(OUT, "w"), indent=1)
    print(f"wrote {len(records)} records -> {OUT.relative_to(paths.PKG_DIR.parent)}")

    # ------- compact summary (baseline sage/0.6b)
    def agg(ds, arm, side, fn):
        vals = [fn(r) for r in records if r["dataset"] == ds and r["arm"] == arm and r["side"] == side
                and r["conv"] == "sage" and r["encoder"] == "qwen3-0.6b"]
        return np.mean(vals) if vals else float("nan")
    print("\n===== PC1 (sage/0.6b): R2 int-set / score |r| / exposure |r| =====")
    for ds in ("tech", "tianchi"):
        for side in ("seeker", "job"):
            row = " ".join(f"{arm}:{agg(ds, arm, side, lambda r: r['r2']['int_set']):.2f}/"
                           f"{agg(ds, arm, side, lambda r: r['corr'].get('score', {'r':[0]})['r'][0]):.2f}/"
                           f"{agg(ds, arm, side, lambda r: r['corr']['exposure']['r'][0]):.2f}" for arm in ARMS)
            print(f"  {ds:8s} {side:6s}: {row}")


if __name__ == "__main__":
    main()
