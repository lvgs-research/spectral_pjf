"""Roy effective rank of the raw input; same raw-only extraction as effrank_raw_unique.py but with the Roy exp(Shannon entropy) functional.

Run:
  python -m link_prediction_experiment.analysis.effrank_raw_unique_roy
Writes tables/data/effrank_raw_unique_roy.json
"""
from __future__ import annotations
import os, json
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import torch
torch.set_num_threads(4)

from .. import paths
from .effrank_raw_unique import raw_tech, raw_tc, _u, EXPS
from .effrank_spectral_roy import roy_erank, roy_erank_std

OUT = paths.PKG_DIR.parent / "tables" / "data" / "effrank_raw_unique_roy.json"


def _er(X):
    return {"eff": roy_erank(X), "std": roy_erank_std(X), "dim": int(X.shape[1]), "n": int(X.shape[0])}


def main():
    out = {}
    for ds, fn in [("tech", raw_tech), ("tianchi", raw_tc)]:
        for exp in EXPS:
            m, j, Xm, Xj = fn(exp)
            out[f"{ds}__{exp}"] = {"raw_edge": _er(np.concatenate([Xm, Xj], axis=1)),
                                   "raw_unique_seeker": _er(_u(Xm, m)),
                                   "raw_unique_job": _er(_u(Xj, j))}
            r = out[f"{ds}__{exp}"]
            print(f"  {ds:8s} {exp} | ROY raw_edge eff={r['raw_edge']['eff']:.2f} "
                  f"uniqS={r['raw_unique_seeker']['eff']:.2f} (n={r['raw_unique_seeker']['n']}) "
                  f"uniqJ={r['raw_unique_job']['eff']:.2f} (n={r['raw_unique_job']['n']})", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {len(out)} raw records -> {OUT.relative_to(paths.PKG_DIR.parent)}")


if __name__ == "__main__":
    main()
