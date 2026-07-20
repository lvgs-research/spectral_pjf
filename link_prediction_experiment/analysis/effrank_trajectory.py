"""Effective-rank trajectory from raw features to the pass/fail pre-logit.

Run:
  python -m link_prediction_experiment.analysis.effrank_trajectory --dataset tech --shard-count 3 --shard-id 0
  python -m link_prediction_experiment.analysis.effrank_trajectory --dataset tianchi
  python -m link_prediction_experiment.analysis.effrank_trajectory --merge
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("TORCH_THREADS", "4"))
import numpy as np
import polars as pl
import torch
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))

from .. import paths
from ..config import (ExperimentConfig, FeatureConfig, ModelConfig,
                      SplitConfig, TrainConfig)
from ..runner import Shared
from ..train import Trainer
from .effrank_util import eff_rank, per_pair_raw_emb
from .effrank_modality_decomposition import eff_rank_std, pre_logit
from .embedding_link_similarity import RESULTS_DIR

SHARD_DIR = RESULTS_DIR / "effrank_traj_shards"
NB_DATA = paths.PKG_DIR.parent / "tables" / "data"                # suite-level analysis-data dir
EMB_DIR = NB_DATA / "embeddings"                                  # saved intermediate features
EMB_ENC_DIR = NB_DATA / "embeddings_encoders"                     # encoder-axis (me5/4b/8b)
SEEDS = [0, 1, 2, 3, 4]

# encoder axis: p1/p2/p3 in the ctrl123_<enc> tag, p8 in ctrl_<enc>.
_ENC_TAGS_TECH = {enc: {"p1": ("p1_mlp", f"paired_tech_ctrl123_{enc}"),
                       "p2": ("p2_sage_con", f"paired_tech_ctrl123_{enc}"),
                       "p3": ("p3_sage_coff", f"paired_tech_ctrl123_{enc}"),
                       "p8": ("p8_parallel", f"paired_tech_ctrl_{enc}")}
                 for enc in ("me5", "qwen3_4b", "qwen3_8b")}
_ENC_TAGS_TC = {enc: {"p1": ("p1_ctrl", f"paired_tc_ctrl123_{enc}"),
                      "p2": ("p2_ctrl", f"paired_tc_ctrl123_{enc}"),
                      "p3": ("p3_ctrl", f"paired_tc_ctrl123_{enc}"),
                      "p8": ("p8_ctrl", f"paired_tc_ctrl_{enc}")}
                for enc in ("me5", "qwen3_4b", "qwen3_8b")}
ENC_SLUG = {"me5": "me5", "qwen3_4b": "qwen3-4b", "qwen3_8b": "qwen3-8b"}  # file-token -> notebook slug

# experiment key -> (arm name, tag) per dataset & conv. p1 GATv2 omitted (MLP has no conv; merge copies SAGE p1).
Tech = {
    "sage":  {"p1": ("p1_mlp", "paired_tech_v2"), "p2": ("p2_sage_con", "paired_tech_v2"),
              "p3": ("p3_sage_coff", "paired_tech_v2"), "p8": ("p8_parallel", "paired_tech_ctrl")},
    "gatv2": {"p2": ("p2_hetgat", "paired_tech_modelgrid"), "p3": ("p3_hetgat", "paired_tech_modelgrid"),
              "p8": ("p8_hetgat", "paired_tech_modelgrid")},
}
TC = {
    "sage":  {"p1": ("p1_ctrl", "paired_tc_ctrl"), "p2": ("p2_ctrl", "paired_tc_ctrl"),
              "p3": ("p3_ctrl", "paired_tc_ctrl"), "p8": ("p8_ctrl", "paired_tc_ctrl")},
    "gatv2": {"p2": ("p2_hetgat", "paired_tc_modelgrid"), "p3": ("p3_hetgat", "paired_tc_modelgrid"),
              "p8": ("p8_hetgat", "paired_tc_modelgrid")},
}


def rebuild_cfg(cd: dict) -> ExperimentConfig:
    cd = dict(cd); subs = {}
    cd.pop("negative", None)                # legacy field, no longer part of the config
    for key, cls in (("feature", FeatureConfig),
                     ("model", ModelConfig), ("split", SplitConfig), ("train", TrainConfig)):
        names = {f.name for f in dataclasses.fields(cls)}
        subs[key] = cls(**{k: v for k, v in dict(cd.pop(key)).items() if k in names})
    return ExperimentConfig(name=cd.pop("name"), **subs, **cd)


# --------------------------------------------------------------------------- rank per stage
def ranks(M, y):
    """centred + column-standardised eff-rank of matrix M over all/pos/neg rows."""
    M = np.asarray(M, np.float64); y = np.asarray(y).astype(int)
    out = {"dim": int(M.shape[1])}
    for grp, mask in [("all", np.ones(len(y), bool)), ("pos", y == 1), ("neg", y == 0)]:
        Mg = M[mask]
        out[grp] = {"eff": eff_rank(Mg), "std": eff_rank_std(Mg), "n": int(mask.sum())}
    return out


def stages_from(Xm, Xj, Zm, Zj, pa, pc, y):
    edge = lambda A, B: np.concatenate([A, B], axis=1)
    return {
        "raw":             ranks(edge(Xm, Xj), y),
        "predecoder":      ranks(edge(Zm, Zj), y),
        "prelogit_accept": ranks(pa, y),
        "prelogit_pass":   ranks(pc, y),
        "prelogit_concat": ranks(np.concatenate([pa, pc], axis=1), y),
    }


# --------------------------------------------------------------------------- Tech extraction
def _tech_shared():
    if not hasattr(_tech_shared, "s"):
        # any controlled arm seeds Shared; features rebuilt per-cfg below
        st = torch.load(paths.CKPT_DIR / "paired_tech_v2" / "p1_mlp_seed0.pt",
                        weights_only=False, map_location="cpu")
        c0 = dataclasses.replace(rebuild_cfg(st["config"]),
                                 train=dataclasses.replace(rebuild_cfg(st["config"]).train, device="cpu"))
        _tech_shared.s = Shared(c0, verbose=True)
    return _tech_shared.s


def extract_tech(exp, arm, tag, seed):
    shared = _tech_shared()
    st = torch.load(paths.CKPT_DIR / tag / f"{arm}_seed{seed}.pt", weights_only=False, map_location="cpu")
    cfg = dataclasses.replace(rebuild_cfg(st["config"]),
                              train=dataclasses.replace(rebuild_cfg(st["config"]).train, device="cpu", seed=seed))
    feats = shared.features(cfg)
    assert feats.in_dims == st["in_dims"], f"in_dims mismatch {arm} s{seed}: {feats.in_dims} vs {st['in_dims']}"
    tr = Trainer(cfg, shared.gstore, feats, shared.df, verbose=False)
    tr.model.load_state_dict(st["best_state"]); tr._cal = st.get("posthoc_cal")
    test = shared.df.filter(pl.col("dataset_split") == "test")
    m = test["seeker_idx"].to_numpy(); j = test["job_idx"].to_numpy()
    ts = test["exposure_ts"].to_numpy().astype(np.int64); y = test["passed"].to_numpy().astype(int)
    Xm, Xj, emb = per_pair_raw_emb(tr, {"trained": tr.model}, m, j, ts)
    Zm, Zj = emb["trained"]
    pf = tr._pair_feats(m, j, ts)
    pa, pc = pre_logit(tr.model.decoder, Zm, Zj, pf)
    return m, j, Xm, Xj, Zm, Zj, pa, pc, y


# --------------------------------------------------------------------------- TC extraction
def _tc_ctx():
    if not hasattr(_tc_ctx, "c"):
        from ..tianchi_pjf.experiment import load_artifacts
        from ..tianchi_pjf.hybrid import assign_hybrid_split
        art = load_artifacts(art_dir=str(paths.PKG_DIR / "data" / "tianchi_prepared_bv"))
        sp = assign_hybrid_split(art.pairs)
        test = sp.filter((pl.col("hsplit") == "test") & (pl.col("role") == "supervision"))
        _tc_ctx.c = (art, sp, test)
    return _tc_ctx.c


def extract_tc(exp, arm, tag, seed):
    from ..tianchi_pjf.experiment import make_x, _load_qwen_content, _CfgShim, _pf, NODE_TYPES
    from ..tianchi_pjf.hybrid import build_visibility, _hybrid_meta, _tc_degree_block
    from ..models import build_model
    art, sp, test = _tc_ctx()
    m = test["seeker_idx"].to_numpy(); j = test["job_idx"].to_numpy()
    y = ((test["delivered"].to_numpy() == 1) & (test["satisfied"].to_numpy() == 1)).astype(int)
    st = torch.load(paths.PKG_DIR / "tianchi_pjf" / "_checkpoints" / tag / f"{arm}_seed{seed}.pt",
                    weights_only=False, map_location="cpu")
    mc = ModelConfig(**st["model_cfg"]); fcfg = FeatureConfig(**st["feature_cfg"])
    meta = _hybrid_meta(art, fcfg)
    qwen = (_load_qwen_content(art.num_nodes, getattr(fcfg, "emb_model", "qwen3"))
            if getattr(fcfg, "use_qwen_content", False) else None)
    x = make_x(art, fcfg, "cpu", qwen=qwen)
    if getattr(fcfg, "tc_ones_collapse", False):
        for t in ("seeker", "job"):
            x[t] = torch.ones(x[t].size(0), 1, dtype=x[t].dtype)
    content_in = {t: x[t].shape[1] for t in NODE_TYPES}
    ei = build_visibility(art, sp, "test", fcfg, "cpu")
    if getattr(fcfg, "tc_degree_nodes", False):
        deg = _tc_degree_block(ei, art.num_nodes, fcfg, torch.device("cpu"))
        x = {t: torch.cat([x[t], deg[t]], dim=1) for t in NODE_TYPES}
    cdims = content_in if mc.kind == "parallel_ref" else None
    model = build_model(_CfgShim(mc, fcfg), {t: x[t].shape[1] for t in x}, meta, content_dims=cdims).to("cpu")
    model.load_state_dict(st["best_state"]); model.eval()
    Xm = x["seeker"].detach().cpu().numpy()[m]; Xj = x["job"].detach().cpu().numpy()[j]
    with torch.no_grad():
        z = model.encode(x, ei)
    Zm = z["seeker"].numpy()[m]; Zj = z["job"].numpy()[j]
    pf = _pf(test, "cpu")
    pa, pc = pre_logit(model.decoder, Zm, Zj, pf)
    return m, j, Xm, Xj, Zm, Zj, pa, pc, y


# --------------------------------------------------------------------------- driver
def work_items(ds):
    reg = Tech if ds == "tech" else TC
    items = []
    for conv, arms in reg.items():
        for exp, (arm, tag) in arms.items():
            for s in SEEDS:
                items.append((ds, conv, exp, arm, tag, s))
    items.sort(key=lambda t: (t[1], t[2], t[5]))
    return items


def dump_npz(ds, conv, exp, seed, m, j, y, Zm, Zj, pa, pc):
    """Save pre-decoder embeddings + both head pre-logits as one .npz per (dataset, conv, experiment, seed), row-aligned to test edges."""
    outdir = EMB_DIR / ds
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outdir / f"{conv}__{exp}__seed{seed}.npz",
        seeker_idx=np.asarray(m, np.int32), job_idx=np.asarray(j, np.int32),
        y=np.asarray(y, np.int8),
        predec_seeker=np.asarray(Zm, np.float32), predec_job=np.asarray(Zj, np.float32),
        prelogit_accept=np.asarray(pa, np.float32), prelogit_pass=np.asarray(pc, np.float32))


def run(ds, shard_id, shard_count, dump=False):
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    items = work_items(ds)
    mine = [x for i, x in enumerate(items) if i % shard_count == shard_id]
    print(f"[shard {shard_id}/{shard_count}] {ds}: {len(mine)}/{len(items)} items "
          f"threads={torch.get_num_threads()} dump={dump}", flush=True)
    extract = extract_tech if ds == "tech" else extract_tc
    for (ds_, conv, exp, arm, tag, seed) in mine:
        rank_p = SHARD_DIR / f"{ds}__{conv}__{exp}__seed{seed}.json"
        emb_p = EMB_DIR / ds / f"{conv}__{exp}__seed{seed}.npz"
        need_rank = not rank_p.exists()
        need_dump = dump and not emb_p.exists()
        if not need_rank and not need_dump:
            print(f"  [skip] {conv}/{exp} s{seed}", flush=True); continue
        import time as _t; t0 = _t.time()
        m, j, Xm, Xj, Zm, Zj, pa, pc, y = extract(exp, arm, tag, seed)
        msg = []
        if need_rank:
            stg = stages_from(Xm, Xj, Zm, Zj, pa, pc, y)
            rec = {"dataset": ds, "conv": conv, "experiment": exp, "arm": arm, "tag": tag,
                   "seed": int(seed), "n": int(len(y)), "n_pos": int((y == 1).sum()), "stages": stg}
            json.dump(rec, open(rank_p, "w"), indent=1)
            pp_ = stg["prelogit_pass"]["all"]
            msg.append(f"rank(pass eff={pp_['eff']:.2f}/std={pp_['std']:.2f})")
        if need_dump:
            dump_npz(ds, conv, exp, seed, m, j, y, Zm, Zj, pa, pc)
            msg.append(f"dump predec={Zm.shape[1]}+{Zj.shape[1]} accept={pa.shape[1]} pass={pc.shape[1]} (n={len(y)})")
        print(f"  {conv:5s} {exp} s{seed} {(_t.time()-t0):.0f}s | " + " | ".join(msg), flush=True)


def finalize_dump():
    """Copy p1 SAGE embeddings -> p1 GATv2 (MLP has no conv) and write the embeddings README."""
    import shutil
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for ds in ("tech", "tianchi"):
        for s in SEEDS:
            src = EMB_DIR / ds / f"sage__p1__seed{s}.npz"
            dst = EMB_DIR / ds / f"gatv2__p1__seed{s}.npz"
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst); copied += 1
    (EMB_DIR / "README.md").write_text(
        "# Saved intermediate features (effective-rank trajectory)\n\n"
        "One `.npz` per `<conv>__<experiment>__seed<s>.npz` under `tech/` and `tianchi/`, row-aligned to\n"
        "the pos+neg TEST edges (Tech 1-day exact PIT; Tianchi static hybrid). Produced by\n"
        "`analysis/effrank_trajectory.py --dump-embeddings`. RAW inputs are NOT saved (too large).\n\n"
        "Keys per file:\n"
        "- `seeker_idx`, `job_idx` (int32), `y` (int8: Tech passed / TC joint deliver&satisfy)\n"
        "- `predec_seeker`, `predec_job` (float32) — pre-decoder node embeddings `model.encode(...)`\n"
        "- `prelogit_accept` (float32) — accept-head pre-logit `head_accept[:-1](concat[z_m,z_j,match])`\n"
        "- `prelogit_pass`   (float32) — PASS/FAIL (cond) head pre-logit `head_cond[:-1](...)`\n\n"
        "Load: `d = np.load('tech/sage__p8__seed0.npz'); d['predec_seeker'], d['prelogit_pass'], ...`\n"
        "`gatv2__p1__*` are copies of `sage__p1__*` (a content MLP has no graph convolution).\n")
    n = len(list(EMB_DIR.rglob("*.npz")))
    print(f"[finalize] copied {copied} p1-GATv2 files; {n} .npz total under {EMB_DIR}")


def dump_npz_enc(ds, enc, exp, seed, m, j, y, Zm, Zj, pa, pc):
    outdir = EMB_ENC_DIR / ds
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outdir / f"sage__{ENC_SLUG[enc]}__{exp}__seed{seed}.npz",
        seeker_idx=np.asarray(m, np.int32), job_idx=np.asarray(j, np.int32), y=np.asarray(y, np.int8),
        predec_seeker=np.asarray(Zm, np.float32), predec_job=np.asarray(Zj, np.float32),
        prelogit_accept=np.asarray(pa, np.float32), prelogit_pass=np.asarray(pc, np.float32))


def run_enc(ds, shard_id, shard_count):
    """Dump pre-decoder + both head pre-logits for the encoder-axis families -> data/embeddings_encoders/<ds>/."""
    EMB_ENC_DIR.mkdir(parents=True, exist_ok=True)
    reg = _ENC_TAGS_TECH if ds == "tech" else _ENC_TAGS_TC
    items = [(enc, exp, arm, tag, s) for enc, arms in reg.items()
             for exp, (arm, tag) in arms.items() for s in SEEDS]
    items.sort(key=lambda t: (t[0], t[1], t[4]))
    mine = [x for i, x in enumerate(items) if i % shard_count == shard_id]
    print(f"[enc shard {shard_id}/{shard_count}] {ds}: {len(mine)}/{len(items)} items", flush=True)
    extract = extract_tech if ds == "tech" else extract_tc
    import time as _t
    for (enc, exp, arm, tag, seed) in mine:
        outp = EMB_ENC_DIR / ds / f"sage__{ENC_SLUG[enc]}__{exp}__seed{seed}.npz"
        if outp.exists():
            print(f"  [skip] {outp.name}", flush=True); continue
        t0 = _t.time()
        m, j, Xm, Xj, Zm, Zj, pa, pc, y = extract(exp, arm, tag, seed)
        dump_npz_enc(ds, enc, exp, seed, m, j, y, Zm, Zj, pa, pc)
        print(f"  {ENC_SLUG[enc]:9s} {exp} s{seed} {(_t.time()-t0):.0f}s -> {outp.name} "
              f"(predec {Zm.shape[1]}+{Zj.shape[1]}, n={len(y)})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["tech", "tianchi"], default="tech")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--dump-embeddings", action="store_true",
                    help="also save pre-decoder + both head pre-logits as .npz under tables/data/embeddings/")
    ap.add_argument("--dump-finalize", action="store_true",
                    help="copy p1 SAGE embeddings -> p1 GATv2 + write embeddings README")
    ap.add_argument("--encoders", action="store_true",
                    help="dump the ENCODER-AXIS families (me5/qwen3_4b/qwen3_8b) -> data/embeddings_encoders/")
    args = ap.parse_args()
    if args.dump_finalize:
        finalize_dump(); return
    ds = "tianchi" if args.dataset == "tianchi" else "tech"
    if args.encoders:
        run_enc(ds, args.shard_id, args.shard_count); return
    run(ds, args.shard_id, args.shard_count, dump=args.dump_embeddings)


if __name__ == "__main__":
    main()
