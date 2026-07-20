"""Run each config over multiple seeds and report mean±std.

Outputs to _results/ (json/csv/md) and _checkpoints/.
"""
from __future__ import annotations

import dataclasses
import json
import time
from typing import Dict, List

import numpy as np
import polars as pl
import torch

from . import paths
from .audit import assert_clean
from .cohorts import assign_cohorts
from .config import ExperimentConfig
from .data import load_interactions
from .features import build_features
from .graph_build import load_graph_store, attach_outcome_neg_edges, validate_graph_store
from .train import Trainer, get_device
from .evaluate import evaluate


class Shared:
    def __init__(self, cfg0: ExperimentConfig, verbose=True):
        t = time.time()
        self.df, self.nm = load_interactions(cfg0)
        self.gstore = load_graph_store()
        # structural pre-flight on the pristine artifact (before outcome-neg edges)
        validate_graph_store(self.gstore, self.nm, verbose=verbose)
        self.df = assign_cohorts(self.df, self.gstore, cfg0.cohort_prior_event)
        # real-negative (seeker, failed, job) edges, PIT-gated
        if "neg_event_ts" in self.df.columns:
            neg = self.df.filter((pl.col("passed") == 0) & pl.col("neg_event_ts").is_not_null())
            if neg.height:
                attach_outcome_neg_edges(self.gstore, neg["seeker_idx"].to_numpy(),
                                         neg["job_idx"].to_numpy(), neg["neg_event_ts"].to_numpy())
        self._feat_cache: Dict[tuple, object] = {}
        if verbose:
            print(f"[shared] loaded df={self.df.height} rows, graph nodes={self.gstore.num_nodes} "
                  f"in {time.time()-t:.1f}s")

    def features(self, cfg: ExperimentConfig):
        fc = cfg.feature
        key = (fc.entity, fc.text_emb if fc.entity == "tabular_textemb" else "-",
               fc.use_tabular if fc.entity == "tabular_textemb" else False,
               getattr(fc, "degree_feats", "none"),
               getattr(fc, "content_off", False),
               getattr(fc, "use_match_feats", False))
        if key not in self._feat_cache:
            self._feat_cache[key] = build_features(cfg, self.gstore, self.nm, self.df)
        return self._feat_cache[key]


def _agg(items: list):
    """Recursively aggregate metric values across seeds (numeric leaf -> {mean,std,n})."""
    first = items[0]
    if isinstance(first, dict):
        keys = set().union(*[set(d.keys()) for d in items if isinstance(d, dict)])
        return {k: _agg([d[k] for d in items if isinstance(d, dict) and k in d]) for k in keys}
    if isinstance(first, bool):
        return first
    if isinstance(first, (int, float)):
        arr = np.array([v for v in items if isinstance(v, (int, float))], dtype=float)
        if arr.size == 0 or np.all(np.isnan(arr)):
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr)),
                "n": int(np.sum(~np.isnan(arr)))}
    return first


def run_config(cfg: ExperimentConfig, shared: Shared, *, tag: str, verbose=True) -> dict:
    t0 = time.time()
    if verbose:
        print(f"\n=== {cfg.name}  (seeds={list(cfg.seeds)}) ===")
    feats = shared.features(cfg)
    # leak-safety + inductiveness audit
    assert_clean(shared.df, shared.gstore, cfg.feature, verbose=verbose,
                 cfg=cfg, in_dims=feats.in_dims)

    per_seed = []
    for s in cfg.seeds:
        cfg_s = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, seed=s))
        ckpt_path = paths.CKPT_DIR / tag / f"{cfg.name}_seed{s}.pt"
        # resume: skip a (config,seed) already finished
        if ckpt_path.exists():
            st = torch.load(ckpt_path, weights_only=False)
            if st.get("done") and st.get("per_seed_metrics") is not None:
                if verbose:
                    print(f" -- seed {s}: already done, reusing saved metrics --")
                per_seed.append(st["per_seed_metrics"]); continue
        if verbose:
            print(f" -- seed {s} --")
        trainer = Trainer(cfg_s, shared.gstore, feats, shared.df, verbose=verbose)
        trainer.fit(ckpt_path if cfg.save_checkpoint else None)
        trainer.fit_posthoc_calibration()   # per-head platt/isotonic on val (leak-safe)
        # eval here is CHUNKED; exact per-row PIT test deferred to exact_test_pass
        sm = {
            "seed": s, "best_epoch": trainer.best_epoch, "best_val": trainer.best_val,
            "val": evaluate(trainer, shared.df, cfg_s, "val", exact=False),
            "test": evaluate(trainer, shared.df, cfg_s, "test", exact=False),
        }
        if cfg.save_checkpoint:
            trainer.save_final(ckpt_path, sm)
        per_seed.append(sm)

    agg = {"val": _agg([x["val"] for x in per_seed]),
           "test": _agg([x["test"] for x in per_seed]),
           "best_epoch": _agg([x["best_epoch"] for x in per_seed])}
    out = {"name": cfg.name, "config": cfg.to_dict(), "device": str(get_device(cfg.train.device)),
           "in_dims": feats.in_dims, "seeds": list(cfg.seeds),
           "per_seed": per_seed, "agg": agg, "seconds": round(time.time() - t0, 1)}
    if verbose:
        _print_brief(out)
    return out


def run_matrix(configs: List[ExperimentConfig], *, tag: str = "run", verbose=True) -> List[dict]:
    shared = Shared(configs[0], verbose=verbose)
    results = []
    # Phase 1: train every config/seed with chunked eval
    for cfg in configs:
        try:
            results.append(run_config(cfg, shared, tag=tag, verbose=verbose))
        except Exception as e:  # keep going; record failure
            import traceback
            traceback.print_exc()
            results.append({"name": cfg.name, "error": repr(e)})
        _save(results, tag)
        _write_long(results, tag)
    # Phase 2: exact per-row PIT test, once, after all training
    if any(getattr(c, "exact_test_pit", False) for c in configs):
        exact_test_pass(configs, shared, results, tag=tag, verbose=verbose)
    _write_summary(results, tag)
    return results


def exact_test_pass(configs, shared, results, *, tag, verbose=True):
    """Recompute TEST metrics with exact per-row PIT on the trained checkpoints (resumable)."""
    by_name = {r.get("name"): r for r in results if "agg" in r}
    for cfg in configs:
        if not getattr(cfg, "exact_test_pit", False):
            continue
        r = by_name.get(cfg.name)
        if r is None:
            continue
        if verbose:
            print(f"\n=== exact per-row PIT test: {cfg.name} ===")
        feats = shared.features(cfg)
        for sm in r["per_seed"]:
            if sm.get("test_exact") is not None:
                continue
            s = sm["seed"]
            cfg_s = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, seed=s))
            ckpt_path = paths.CKPT_DIR / tag / f"{cfg.name}_seed{s}.pt"
            if not ckpt_path.exists():
                continue
            st = torch.load(ckpt_path, weights_only=False)
            trainer = Trainer(cfg_s, shared.gstore, feats, shared.df, verbose=False)
            trainer.model.load_state_dict(st["best_state"])
            trainer._cal = st.get("posthoc_cal")   # restore val-fit calibrators
            sm["test_exact"] = evaluate(trainer, shared.df, cfg_s, "test", exact=True)
            st["per_seed_metrics"] = sm           # persist so a re-run skips it
            torch.save(st, ckpt_path)
            if verbose:
                te = sm["test_exact"]["auc_exposed"]
                print(f"   seed {s}: exact exp_all={te.get('all'):.4f} cold={te.get('cold_job', float('nan'))}")
        ev = [sm["test_exact"] for sm in r["per_seed"] if sm.get("test_exact") is not None]
        if ev:
            r["agg"]["test_exact"] = _agg(ev)
        _save(results, tag); _write_long(results, tag)


def _gm(d, *path):
    """Mean of an aggregated metric at a nested path (None if absent)."""
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    if isinstance(cur, dict) and "mean" in cur:
        return cur["mean"]
    return cur


def _print_brief(m: dict) -> None:
    t = m["agg"].get("test_exact") or m["agg"]["test"]
    print(f"   [test mean/{len(m['seeds'])} seeds] "
          f"exp_all={_fmt(_gm(t,'auc_exposed','all'))} "
          f"exp_cold={_fmt(_gm(t,'auc_exposed','cold_job'))} "
          f"user_GAUC={_fmt(_gm(t,'user_grouped_ranking','GAUC'))} "
          f"job_GAUC={_fmt(_gm(t,'job_ranking','GAUC'))} "
          f"best_ep={_fmt(_gm(m['agg'],'best_epoch'))} ({m.get('seconds')}s)")


def _summary_row(m: dict) -> dict:
    t = m["agg"].get("test_exact") or m["agg"]["test"]
    g = lambda *p: _gm(t, *p)
    return {
        "name": m["name"],
        "test_pit": "exact" if m["agg"].get("test_exact") else "chunked",
        # exposed real pos/neg AUC (all + cold-job slice)
        "exp_all": g("auc_exposed", "all"),
        "exp_cold_job": g("auc_exposed", "cold_job"),
        "exp_peruser_AUC": g("exposed_ranking", "per_user_AUC"),
        # seeker-side grouped ranking (== exp_peruser_AUC by construction)
        "user_GAUC": g("user_grouped_ranking", "GAUC"),
        # job-grouped ranking (item side)
        "job_GAUC": g("job_ranking", "GAUC"),
        "job_MAP@10": g("job_ranking", "MAP@10"),
        "job_Hit@10": g("job_ranking", "Hit@10"),
        "best_epoch": _gm(m["agg"], "best_epoch"),
        "seconds": m.get("seconds"),
    }


def _save(results: List[dict], tag: str) -> None:
    with open(paths.RESULTS_DIR / f"{tag}_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


def _flatten(obj, prefix="") -> list:
    rows = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            rows += _flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        rows.append((prefix, float(obj)))
    return rows


def _write_long(results: List[dict], tag: str) -> None:
    """Tidy per-seed rows: (config, seed, split, metric, value)."""
    rows = []
    for m in results:
        if "error" in m:
            continue
        for sm in m.get("per_seed", []):
            seed = sm["seed"]
            rows.append({"config": m["name"], "seed": seed, "split": "train",
                         "metric": "best_epoch", "value": float(sm.get("best_epoch", 0))})
            for split in ("val", "test"):
                for metric, val in _flatten(sm.get(split, {})):
                    rows.append({"config": m["name"], "seed": seed, "split": split,
                                 "metric": metric, "value": val})
    if rows:
        pl.DataFrame(rows).write_csv(paths.RESULTS_DIR / f"{tag}_long.csv")


def _fmt(v) -> str:
    if isinstance(v, float):
        return "" if v != v else f"{v:.4f}"   # NaN -> blank
    return "" if v is None else str(v)


def _write_summary(results: List[dict], tag: str) -> None:
    rows = [_summary_row(m) for m in results if "error" not in m and "agg" in m]
    if not rows:
        return
    tbl = pl.DataFrame(rows)
    tbl.write_csv(paths.RESULTS_DIR / f"{tag}_summary.csv")
    _write_md(tbl, paths.RESULTS_DIR / f"{tag}_summary.md", tag)
    print(f"\n[summary] wrote {paths.RESULTS_DIR}/{tag}_summary.csv")
    print(tbl)


def _write_md(tbl: pl.DataFrame, path, tag: str) -> None:
    cols = tbl.columns
    lines = [f"# Link-prediction benchmark: {tag} (seed mean)", "",
             "| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for row in tbl.iter_rows(named=True):
        lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
