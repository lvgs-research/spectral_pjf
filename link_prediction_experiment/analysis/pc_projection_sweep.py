r"""PC-projection ablations (remove PC1 / PC2 / top-2) over all arms/datasets/conv/encoders,
reusing cached pre-decoder embeddings + the trained decoder (no re-encode).

Run: python -m link_prediction_experiment.analysis.pc_projection_sweep -> tables/data/pc_projection.json
"""
from __future__ import annotations
import os, json, dataclasses
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import polars as pl
import torch
torch.set_num_threads(4)
from sklearn.metrics import roc_auc_score

from .. import paths
from .pc1_ablation import joint_score, auc, gauc, top_rvs, proj_out_multi
from .effrank_exposure import tech_exposure, tc_exposure
from .effrank_trajectory import rebuild_cfg, _tech_shared, _tc_ctx
from .pc1_semantics import _variants
from ..train import Trainer

OUT = paths.PKG_DIR.parent / "tables" / "data" / "pc_projection.json"
SEEDS = [0, 1, 2, 3, 4]   # full 5-seed family
ARMS = ["p1", "p2", "p3", "p8"]
PC_SETS = [("PC1", (0,)), ("PC2", (1,)), ("PC1+2", (0, 1))]


# --------------------------------------------------------------------------- decoder + pair feats
_TECH_TR_CACHE = {}   # features cached in Shared (state differs per seed)


def tech_dec_pf(ck_path):
    """Trainer from checkpoint cfg (no encode) -> (decoder, pf, m, j, y)."""
    shared = _tech_shared()
    st = torch.load(ck_path, weights_only=False, map_location="cpu")
    cfg = rebuild_cfg(st["config"])
    cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, device="cpu"))
    feats = shared.features(cfg)
    tr = Trainer(cfg, shared.gstore, feats, shared.df, verbose=False)
    tr.model.load_state_dict(st["best_state"])
    test = shared.df.filter(pl.col("dataset_split") == "test")
    m = test["seeker_idx"].to_numpy(); j = test["job_idx"].to_numpy()
    ts = test["exposure_ts"].to_numpy().astype(np.int64)
    y = test["passed"].to_numpy().astype(int)
    pf = tr._pair_feats(m, j, ts)
    return tr.model.decoder, pf, m, j, y


_TC_QWEN_CACHE = {}
_TC_X_CACHE = {}


def tc_dec_pf(ck_path):
    """build_model from checkpoint cfgs (no encode) -> (decoder, pf, m, j, y)."""
    from ..tianchi_pjf.experiment import make_x, _load_qwen_content, _CfgShim, _pf, NODE_TYPES
    from ..tianchi_pjf.hybrid import build_visibility, _hybrid_meta, _tc_degree_block
    from ..models import build_model
    from ..config import ModelConfig, FeatureConfig
    art, sp, test = _tc_ctx()
    st = torch.load(ck_path, weights_only=False, map_location="cpu")
    mc = ModelConfig(**st["model_cfg"]); fcfg = FeatureConfig(**st["feature_cfg"])
    meta = _hybrid_meta(art, fcfg)
    xkey = (getattr(fcfg, "emb_model", "qwen3"), getattr(fcfg, "use_qwen_content", False),
            getattr(fcfg, "tc_ones_collapse", False), getattr(fcfg, "tc_degree_nodes", False))
    if xkey not in _TC_X_CACHE:
        enc = getattr(fcfg, "emb_model", "qwen3")
        if fcfg.use_qwen_content and enc not in _TC_QWEN_CACHE:
            _TC_QWEN_CACHE.clear()                     # keep at most one encoder's tensors in memory
            _TC_QWEN_CACHE[enc] = _load_qwen_content(art.num_nodes, enc)
        x = make_x(art, fcfg, "cpu", qwen=_TC_QWEN_CACHE.get(enc) if fcfg.use_qwen_content else None)
        if getattr(fcfg, "tc_ones_collapse", False):
            for t in ("seeker", "job"):
                x[t] = torch.ones(x[t].size(0), 1, dtype=x[t].dtype)
        content_in = {t: x[t].shape[1] for t in NODE_TYPES}
        ei = build_visibility(art, sp, "test", fcfg, "cpu")
        if getattr(fcfg, "tc_degree_nodes", False):
            deg = _tc_degree_block(ei, art.num_nodes, fcfg, torch.device("cpu"))
            x = {t: torch.cat([x[t], deg[t]], dim=1) for t in NODE_TYPES}
        _TC_X_CACHE.clear()
        _TC_X_CACHE[xkey] = ({t: x[t].shape[1] for t in x}, content_in)
    in_dims, content_in = _TC_X_CACHE[xkey]
    cdims = content_in if mc.kind == "parallel_ref" else None
    model = build_model(_CfgShim(mc, fcfg), in_dims, meta, content_dims=cdims).to("cpu")
    model.load_state_dict(st["best_state"]); model.eval()
    m = test["seeker_idx"].to_numpy(); j = test["job_idx"].to_numpy()
    y = ((test["delivered"].to_numpy() == 1) & (test["satisfied"].to_numpy() == 1)).astype(int)
    pf = _pf(test, "cpu")
    return model.decoder, pf, m, j, y


# --------------------------------------------------------------------------- metrics
def make_metrics(m, j, y, j_exp):
    jcold = j_exp[j] == 0; jwarm = ~jcold
    cold_groups = np.unique(j[jcold]); warm_groups = np.unique(j[jwarm])

    def gauc_sub(groupset, s):
        aucs = []
        for g in groupset:
            idx = j == g; yy = y[idx]
            if 0 < yy.sum() < len(yy):
                aucs.append(roc_auc_score(yy, s[idx]))
        return float(np.mean(aucs)) if aucs else float("nan")

    def metrics(s):
        return {"pooled": auc(y, s), "seekerGAUC": gauc(m, y, s)[0], "jobGAUC": gauc(j, y, s)[0],
                "pooled_jobWARM": auc(y[jwarm], s[jwarm]), "pooled_jobCOLD": auc(y[jcold], s[jcold]),
                "jobGAUC_warm": gauc_sub(warm_groups, s), "jobGAUC_cold": gauc_sub(cold_groups, s)}
    return metrics


# --------------------------------------------------------------------------- driver
def main():
    exposures = {"tech": tech_exposure(), "tianchi": tc_exposure()}
    records = []
    for ds in ("tech", "tianchi"):
        m_exp, j_exp = exposures[ds]
        for conv, enc, npz_fn, ck_fn in _variants(ds):
            for arm in ARMS:
                for sd in SEEDS:
                    fp = npz_fn(arm, sd); ckp = ck_fn(arm, sd)
                    if not fp.exists() or not ckp.exists():
                        print(f"  [miss] {ds} {conv}/{enc} {arm} s{sd}", flush=True); continue
                    d = np.load(fp)
                    dec, pf, m, j, y = (tech_dec_pf if ds == "tech" else tc_dec_pf)(ckp)
                    assert np.array_equal(d["seeker_idx"], m) and np.array_equal(d["job_idx"], j), \
                        f"npz/test-frame misalignment: {fp.name}"
                    Zm, Zj = d["predec_seeker"].astype(np.float64), d["predec_job"].astype(np.float64)
                    _, fm = np.unique(m, return_index=True); _, fj = np.unique(j, return_index=True)
                    metrics = make_metrics(m, j, y, j_exp)
                    rec = {"dataset": ds, "conv": conv, "encoder": enc, "arm": arm, "seed": sd,
                           "base": metrics(joint_score(dec, Zm, Zj, pf)), "removed": {}}
                    for name, idxs in PC_SETS:
                        Vm = top_rvs(Zm[fm], idxs); Vj = top_rvs(Zj[fj], idxs)
                        s1 = joint_score(dec, proj_out_multi(Zm, Vm), proj_out_multi(Zj, Vj), pf)
                        rec["removed"][name] = metrics(s1)
                    records.append(rec)
                    print(f"  {ds:8s} {conv:5s}/{enc:10s} {arm} s{sd} | base={rec['base']['pooled']:.4f} "
                          f"-PC1={rec['removed']['PC1']['pooled']:.4f} -PC2={rec['removed']['PC2']['pooled']:.4f} "
                          f"-PC1+2={rec['removed']['PC1+2']['pooled']:.4f}", flush=True)
                # free Tech feature tensors between arms
                if ds == "tech":
                    try:
                        _tech_shared()._feat_cache.clear()
                    except AttributeError:
                        pass
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"seeds": SEEDS, "pc_sets": [n for n, _ in PC_SETS],
               "doc": "cached-embedding PC-projection ablations; scores = trained decoder on projected "
                      "pair-level pre-decoder embeddings (logsig(a)+logsig(c)); no re-encode",
               "records": records}, open(OUT, "w"), indent=1)
    print(f"\nwrote {len(records)} records -> {OUT.relative_to(paths.PKG_DIR.parent)}")


if __name__ == "__main__":
    main()
