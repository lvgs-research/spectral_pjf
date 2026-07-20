"""CLI: run the Tianchi PJF link-prediction benchmark (hybrid flow).

  python -m link_prediction_experiment.tianchi_pjf.run --mode hybrid --seeds 5 --tag paired_tc_ctrl
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hybrid"], default="hybrid",
                    help="hybrid is the canonical flow: own warm/cold split + incremental "
                         "visibility + reciprocal ESMM")
    ap.add_argument("--only", default=None, help="comma-separated subset of config names to run")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--artifact-dir", dest="artifact_dir", default=None,
                    help="artifact dir (default data/tianchi_prepared; use "
                         "data/tianchi_prepared_bv for the boundary-validated + match-feats build)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--emb-model", dest="emb_model", default="qwen3",
                    choices=["qwen3", "qwen3_4b", "qwen3_8b", "me5"],
                    help="content encoder for the hybrid arms (encoder variant sweep)")
    ap.add_argument("--conv-type", dest="conv_type", default="sage",
                    choices=["sage", "gatv2"],
                    help="graph aggregator for the hybrid graph arms (default sage; "
                         "gatv2 = the GATv2 paired-encoder variant)")
    ap.add_argument("--attn-heads", dest="attn_heads", type=int, default=None,
                    help="GATv2 attention heads (default None = ModelConfig's 4). Lower to 2/1 to "
                         "fit the larger Tianchi graph in GPU memory (heads=4 OOMs a 24GB card).")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--save-ckpt", dest="save_ckpt", action="store_true",
                    help="save best+last model checkpoints to _checkpoints/<tag>/ (checkpoints "
                         "are always written in --mode hybrid; flag kept for CLI compatibility)")
    args = ap.parse_args()

    # CANONICAL flow: own split + incremental visibility + reciprocal ESMM
    from .hybrid import run_hybrid
    run_hybrid(seeds=tuple(range(args.seeds)), epochs=args.epochs or 250,
               device=args.device, tag=args.tag or "hybrid_v1", art_dir=args.artifact_dir,
               patience=args.patience or 50, only=args.only,
               emb_model=args.emb_model,
               conv_type=args.conv_type,
               attn_heads=args.attn_heads)   # saves + resumes/skips by default


if __name__ == "__main__":
    main()
