"""Spectral analysis over cached embeddings using Roy & Vetterli (2007) effective rank.

Run:  python link_prediction_experiment/analysis/effrank_spectral_roy.py   (writes tables/data/effrank_spectral_roy.json)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

SUITE = Path(__file__).resolve().parents[2]
DATA = SUITE / "tables" / "data"
EMB = DATA / "embeddings"
EMB_ENC = DATA / "embeddings_encoders"
OUT = DATA / "effrank_spectral_roy.json"
OUT_ENC = DATA / "effrank_spectral_roy_enc.json"
SEEDS = [0, 1, 2, 3, 4]
# content-encoder axis: qwen3-0.6b baseline + me5/4b/8b
ENC_AXIS = [("qwen3-0.6b", EMB,     "sage__{exp}__seed{s}.npz"),
            ("me5",        EMB_ENC, "sage__me5__{exp}__seed{s}.npz"),
            ("qwen3-4b",   EMB_ENC, "sage__qwen3-4b__{exp}__seed{s}.npz"),
            ("qwen3-8b",   EMB_ENC, "sage__qwen3-8b__{exp}__seed{s}.npz")]
ENC_EXPS = ["p1", "p2", "p3", "p8"]


def roy_erank(X, eps=1e-9):
    """Roy & Vetterli effective rank = exp(Shannon entropy of the L1-normalized mean-centred singular spectrum)."""
    X = np.asarray(X, np.float64)
    if X.shape[0] < 3 or X.shape[1] == 0:
        return 0.0
    Xc = X - X.mean(0, keepdims=True)
    n, d = Xc.shape
    if d <= n and d <= 4096:
        lam = np.linalg.eigvalsh(Xc.T @ Xc)
        s = np.sqrt(np.clip(lam, 0.0, None))
    else:
        s = np.linalg.svd(Xc, compute_uv=False)
    s = s[s > eps]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))


def roy_erank_std(X, eps=1e-9):
    """Column-standardized Roy effective rank (correlation spectrum)."""
    X = np.asarray(X, np.float64)
    if X.shape[0] < 3 or X.shape[1] == 0:
        return 0.0
    return roy_erank(X / (X.std(0, keepdims=True) + eps))


def cov_roy_erank(X, eps=1e-12):
    """Roy effective rank of the covariance matrix (dimensional-collapse diagnostic; in [1, dim])."""
    X = np.asarray(X, np.float64)
    if X.shape[0] < 3 or X.shape[1] == 0:
        return 0.0
    Xc = X - X.mean(0, keepdims=True)
    n, d = Xc.shape
    if d <= n and d <= 4096:
        lam = np.linalg.eigvalsh(Xc.T @ Xc)                  # ∝ covariance eigenvalues
    else:
        lam = np.linalg.svd(Xc, compute_uv=False) ** 2
    lam = lam[lam > eps]
    if lam.size == 0:
        return 0.0
    p = lam / lam.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def cov_roy_erank_std(X, eps=1e-9):
    """Same collapse diagnostic on the CORRELATION matrix (columns standardized first)."""
    X = np.asarray(X, np.float64)
    if X.shape[0] < 3 or X.shape[1] == 0:
        return 0.0
    return cov_roy_erank(X / (X.std(0, keepdims=True) + eps))


def energy_concentration(X, eps=1e-12):
    """Top-1 / top-3 covariance-energy fractions (e1, e3) of the mean-centred spectrum."""
    X = np.asarray(X, np.float64)
    if X.shape[0] < 3 or X.shape[1] == 0:
        return 0.0, 0.0
    Xc = X - X.mean(0, keepdims=True)
    n, d = Xc.shape
    if d <= n and d <= 4096:
        lam = np.linalg.eigvalsh(Xc.T @ Xc)
    else:
        lam = np.linalg.svd(Xc, compute_uv=False) ** 2
    lam = np.sort(lam[lam > eps])[::-1]
    tot = lam.sum()
    if tot <= 0:
        return 0.0, 0.0
    return float(lam[:1].sum() / tot), float(lam[:3].sum() / tot)


def _er(X):
    e1, e3 = energy_concentration(X)
    return {"eff": roy_erank(X), "std": roy_erank_std(X),
            "cov": cov_roy_erank(X), "cov_std": cov_roy_erank_std(X),  # covariance-matrix Roy erank (collapse)
            "e1": e1, "e3": e3,                                        # top-1 / top-3 covariance energy fraction
            "dim": int(X.shape[1]), "n": int(X.shape[0])}


def _unique_rows(M, ids):
    _, first = np.unique(np.asarray(ids), return_index=True)
    return M[np.sort(first)]


def analyze_file(fp):
    d = np.load(fp)
    Zm, Zj = d["predec_seeker"], d["predec_job"]
    pa, pc = d["prelogit_accept"], d["prelogit_pass"]
    mi, ji = d["seeker_idx"], d["job_idx"]
    return {
        "capacity": {
            "joint_predec":      _er(np.concatenate([Zm, Zj], axis=1)),
            "seeker_predec":     _er(Zm),              # per-side collapse
            "job_predec":        _er(Zj),              # per-side collapse
            "seeker2job_accept": _er(pa),
            "job2seeker_pass":   _er(pc),
            "joint_prod":        _er(pa * pc),                       # accept (x) pass (CTCVR)
            "joint_concat":      _er(np.concatenate([pa, pc], axis=1)),
        },
        "personalization": {
            "unique_seeker": _er(_unique_rows(Zm, mi)),
            "unique_job":    _er(_unique_rows(Zj, ji)),
        },
        "n_edges": int(len(mi)),
    }


def main():
    if not EMB.exists():
        print(f"MISSING {EMB} — run effrank_trajectory.py --dump-embeddings first", file=sys.stderr)
        return 1
    records = []
    for ds in ("tech", "tianchi"):
        for fp in sorted((EMB / ds).glob("*.npz")):
            conv, exp, sd = fp.stem.split("__")
            rec = analyze_file(fp)
            rec.update(dataset=ds, conv=conv, experiment=exp, seed=int(sd.replace("seed", "")))
            records.append(rec)
            cap, per = rec["capacity"], rec["personalization"]
            print(f"  {ds:8s} {conv:5s} {exp} s{rec['seed']} | ROY predec={cap['joint_predec']['eff']:.2f} "
                  f"accept={cap['seeker2job_accept']['eff']:.2f} pass={cap['job2seeker_pass']['eff']:.2f} "
                  f"joint(a*p)={cap['joint_prod']['eff']:.2f} | uniqS={per['unique_seeker']['eff']:.2f} "
                  f"uniqJ={per['unique_job']['eff']:.2f}", flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    json.dump({"seeds": SEEDS, "rank_measure": "roy_effective_rank_exp_shannon_entropy", "records": records},
              open(OUT, "w"), indent=1)
    print(f"\nwrote {len(records)} records -> {OUT.relative_to(SUITE)}")
    return 0


def run_encoders():
    """Encoder-axis Roy effective ranks (SAGE × {qwen3-0.6b,me5,qwen3-4b,qwen3-8b} × p1/p2/p3/p8, 5 seeds, both datasets) -> OUT_ENC."""
    records = []
    for ds in ("tech", "tianchi"):
        for enc, base, tmpl in ENC_AXIS:
            for exp in ENC_EXPS:
                for s in SEEDS:
                    fp = base / ds / tmpl.format(exp=exp, s=s)
                    if not fp.exists():
                        print(f"  [miss] {fp.relative_to(SUITE)}", file=sys.stderr); continue
                    rec = analyze_file(fp)
                    rec.update(dataset=ds, conv="sage", experiment=exp, seed=s, encoder=enc)
                    records.append(rec)
                    cap = rec["capacity"]
                    print(f"  {ds:8s} {enc:10s} {exp} s{s} | ROY predec={cap['joint_predec']['eff']:.2f} "
                          f"cov={cap['joint_predec']['cov']:.2f}", flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    json.dump({"seeds": SEEDS, "rank_measure": "roy_effective_rank_exp_shannon_entropy",
               "encoder_axis": [e for e, _, _ in ENC_AXIS], "conv": "sage", "records": records},
              open(OUT_ENC, "w"), indent=1)
    print(f"\nwrote {len(records)} encoder-axis records -> {OUT_ENC.relative_to(SUITE)}")
    return 0


if __name__ == "__main__":
    sys.exit(run_encoders() if "--encoders" in sys.argv else main())
