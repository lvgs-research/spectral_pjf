"""CLI entry point for the Tech experiments (content-placement family p1/p2/p3/p8).

  python -m link_prediction_experiment.run --mode paired --seeds 0,1,2,3,4 --tag paired_tech_v2
"""
from __future__ import annotations

import argparse
import dataclasses

from .config import (ExperimentConfig, paired_encoder_test,
                     controlled_p8_test, model_grid_test)
from .runner import run_matrix

# --mode -> preset builder
_MODES = {
    "paired": paired_encoder_test,
    "ctrlp8": controlled_p8_test,
    "modelgrid": model_grid_test,
}


def _override(cfg: ExperimentConfig, args) -> ExperimentConfig:
    t = cfg.train
    repl = {}
    if args.epochs is not None: repl["epochs"] = args.epochs
    if args.max_train_pos is not None: repl["max_train_pos"] = args.max_train_pos
    if args.max_eval_pos is not None: repl["max_eval_pos"] = args.max_eval_pos
    if args.max_eval_exposed is not None: repl["max_eval_exposed"] = args.max_eval_exposed
    if args.device is not None: repl["device"] = args.device
    if args.chunks_train is not None: repl["n_time_chunks_train"] = args.chunks_train
    if args.chunks_eval is not None: repl["n_time_chunks_eval"] = args.chunks_eval
    if repl:
        cfg = dataclasses.replace(cfg, train=dataclasses.replace(t, **repl))
    if args.seeds is not None:
        cfg = dataclasses.replace(cfg, seeds=tuple(int(s) for s in args.seeds.split(",")))
    if getattr(args, "text_emb", None):        # encoder-axis sweep
        cfg = dataclasses.replace(cfg, feature=dataclasses.replace(cfg.feature, text_emb=args.text_emb))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=sorted(_MODES), default="paired")
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of config names to keep from the chosen --mode")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-train-pos", dest="max_train_pos", type=int, default=None)
    ap.add_argument("--max-eval-pos", dest="max_eval_pos", type=int, default=None)
    ap.add_argument("--max-eval-exposed", dest="max_eval_exposed", type=int, default=None)
    ap.add_argument("--chunks-train", dest="chunks_train", type=int, default=None)
    ap.add_argument("--chunks-eval", dest="chunks_eval", type=int, default=None)
    ap.add_argument("--seeds", default=None, help="comma-separated, e.g. 0,1,2,3,4")
    ap.add_argument("--device", default=None)
    ap.add_argument("--text-emb", dest="text_emb", default=None,
                    help="override feature.text_emb across all arms "
                         "(encoder axis: qwen3 / qwen3_4b / qwen3_8b / me5)")
    args = ap.parse_args()

    cfgs = _MODES[args.mode]()

    if args.only:
        keep = set(args.only.split(","))
        cfgs = [c for c in cfgs if c.name in keep]
        if not cfgs:
            raise SystemExit(f"--only matched no configs in --mode {args.mode}")
        missing = keep - {c.name for c in cfgs}
        if missing:
            print(f"[warn] --only names not in --mode {args.mode}: {sorted(missing)}")
    cfgs = [_override(c, args) for c in cfgs]
    tag = args.tag or args.mode
    run_matrix(cfgs, tag=tag)


if __name__ == "__main__":
    main()
