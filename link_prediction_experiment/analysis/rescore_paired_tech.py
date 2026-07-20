"""Re-score the Tech paired-encoder checkpoints with the current evaluate.py (no training).

Writes one shard JSON per (tag,arm,seed) into _results/rescore_shards_tech/; --merge assembles per-tag.
Env: RESCORE_DEVICE, TORCH_THREADS, SHARD_ID/SHARD_COUNT, RESCORE_TAGS, RESCORE_ARMS/SEEDS.
"""
import os, sys, json, time, dataclasses
from pathlib import Path
import numpy as np

REPO = str(Path(__file__).resolve().parents[2])  # suite root
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("TORCH_THREADS", "2"))
import torch
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))

from link_prediction_experiment.config import (
    ExperimentConfig, FeatureConfig, ModelConfig, SplitConfig, TrainConfig)
from link_prediction_experiment.runner import Shared, _agg
from link_prediction_experiment.train import Trainer, get_device
from link_prediction_experiment.evaluate import evaluate
from link_prediction_experiment import paths

DEVICE = os.environ.get("RESCORE_DEVICE", "cpu")
TAGS = os.environ.get("RESCORE_TAGS", "paired_tech_v2,paired_tech_ctrl").split(",")
ENC = {"paired_tech_v2": "qwen3", "paired_tech_ctrl": "qwen3", "paired_tech_me5": "me5",
       "paired_tech_ctrl_me5": "me5", "paired_tech_ctrl_qwen3_4b": "qwen3_4b",
       "paired_tech_ctrl_qwen3_8b": "qwen3_8b"}
ARM_FILT = set(x for x in os.environ.get("RESCORE_ARMS", "").split(",") if x)
SEED_FILT = set(int(x) for x in os.environ.get("RESCORE_SEEDS", "").split(",") if x != "")
SHARD_ID = int(os.environ.get("SHARD_ID", "0"))
SHARD_COUNT = int(os.environ.get("SHARD_COUNT", "1"))
CKPT = paths.CKPT_DIR
RES = paths.RESULTS_DIR
SHARD_DIR = RES / "rescore_shards_tech"


def rebuild_cfg(cd: dict) -> ExperimentConfig:
    cd = dict(cd); subs = {}
    cd.pop("negative", None)                # legacy field, no longer part of the config
    for key, cls in (("feature", FeatureConfig),
                     ("model", ModelConfig), ("split", SplitConfig), ("train", TrainConfig)):
        names = {f.name for f in dataclasses.fields(cls)}
        subs[key] = cls(**{k: v for k, v in dict(cd.pop(key)).items() if k in names})
    return ExperimentConfig(name=cd.pop("name"), **subs, **cd)


def dig(d, *p):
    for k in p:
        d = d.get(k, {}) if isinstance(d, dict) else {}
    return d


def enumerate_ckpts():
    out = []
    for tag in TAGS:
        for cp in sorted((CKPT / tag).glob("*.pt")):
            arm, seed = cp.stem.rsplit("_seed", 1); seed = int(seed)
            if ARM_FILT and arm not in ARM_FILT:
                continue
            if SEED_FILT and seed not in SEED_FILT:
                continue
            out.append((tag, arm, seed, cp))
    out.sort(key=lambda x: (x[0], x[1], x[2]))
    return out


def run_shard():
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    allck = enumerate_ckpts()
    mine = [x for i, x in enumerate(allck) if i % SHARD_COUNT == SHARD_ID]
    print(f"[shard {SHARD_ID}/{SHARD_COUNT}] device={DEVICE} threads={torch.get_num_threads()} "
          f"assigned {len(mine)}/{len(allck)} ckpts", flush=True)
    first = rebuild_cfg(torch.load(allck[0][3], weights_only=False, map_location="cpu")["config"])
    first = dataclasses.replace(first, train=dataclasses.replace(first.train, device=DEVICE))
    shared = Shared(first, verbose=(SHARD_ID == 0))
    for tag, arm, seed, cp in mine:
        outp = SHARD_DIR / f"{tag}__{arm}__seed{seed}.json"
        if outp.exists():
            print(f"  [skip existing] {outp.name}", flush=True); continue
        t0 = time.time()
        st = torch.load(cp, weights_only=False, map_location="cpu")
        cfg = rebuild_cfg(st["config"])
        cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, device=DEVICE, seed=seed))
        feats = shared.features(cfg)
        assert feats.in_dims == st["in_dims"], f"in_dims mismatch {arm} s{seed}"
        tr = Trainer(cfg, shared.gstore, feats, shared.df, verbose=False)
        tr.model.load_state_dict(st["best_state"]); tr._cal = st.get("posthoc_cal")
        te = evaluate(tr, shared.df, cfg, "test", exact=True)
        rec = {"tag": tag, "arm": arm, "enc": ENC.get(tag, "qwen3"), "seed": seed,
               "best_epoch": st.get("best_epoch"), "best_val": st.get("best_val"),
               "in_dims": feats.in_dims, "config": cfg.to_dict(), "test_exact": te}
        json.dump(rec, open(outp, "w"), default=str)
        old = dig(st.get("per_seed_metrics", {}), "test_exact")
        nea, oea = dig(te, "auc_exposed", "all"), dig(old, "auc_exposed", "all")
        ngw, og = dig(te, "job_ranking", "GAUC_expw"), dig(old, "job_ranking", "GAUC")
        ug, pua = dig(te, "user_grouped_ranking", "GAUC"), dig(te, "exposed_ranking", "per_user_AUC")
        dea = abs(nea - oea) if isinstance(oea, (int, float)) else float("nan")
        dgw = abs(ngw - og) if isinstance(og, (int, float)) else float("nan")
        print(f"  [{tag}/{arm} s{seed}] {time.time()-t0:.0f}s exp_all Δ={dea:.2e} "
              f"jobGAUC unw={dig(te,'job_ranking','GAUC'):.5f} expwΔ={dgw:.2e} "
              f"user_gr==per_user={abs(ug-pua)<1e-9}", flush=True)


def merge():
    recs = [json.load(open(SHARD_DIR / p)) for p in os.listdir(SHARD_DIR) if p.endswith(".json")]
    by_tag = {}
    for r in recs:
        by_tag.setdefault(r["tag"], {}).setdefault(r["arm"], []).append(r)
    for tag, arms in by_tag.items():
        records = []
        for arm, seedrecs in arms.items():
            seedrecs.sort(key=lambda r: r["seed"])
            per_seed = [{"seed": r["seed"], "best_epoch": r["best_epoch"], "best_val": r["best_val"],
                         "test_exact": r["test_exact"]} for r in seedrecs]
            agg = {"test_exact": _agg([x["test_exact"] for x in per_seed]),
                   "best_epoch": _agg([x["best_epoch"] for x in per_seed])}
            records.append({"name": arm, "config": seedrecs[0]["config"], "device": DEVICE,
                            "in_dims": seedrecs[0]["in_dims"], "seeds": [r["seed"] for r in seedrecs],
                            "per_seed": per_seed, "agg": agg})
        records.sort(key=lambda r: r["name"])
        out = RES / f"{tag}_results.RESCORED.json"
        json.dump(records, open(out, "w"), indent=2, default=str)
        print(f"[merge] wrote {out} ({len(records)} arms, {sum(len(a) for a in arms.values())} seed-recs)")


if __name__ == "__main__":
    if "--merge" in sys.argv:
        merge()
    else:
        run_shard()
