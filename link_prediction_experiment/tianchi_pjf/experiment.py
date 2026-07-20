"""Inductive PJF link-prediction experiment on the Tianchi graph.

Predicts the (seeker -> job) SATISFIED link over EXPOSED pairs; user-disjoint split, SATISFIED
edge never in the message-passing graph (leak-safe). Reuses link_prediction_experiment/models.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
import torch

from ..config import FeatureConfig
from ..models import GraphMeta

PKG_DIR = Path(__file__).resolve().parent
OUT_DIR = PKG_DIR.parent / "data" / "tianchi_prepared"
CKPT_DIR = PKG_DIR / "_checkpoints"        # trained-model checkpoints (best + last states)


def save_ckpt(path, *, name, seed, model_cfg, feature_cfg, in_dims,
              best_state, last_state, best_ep, best_metric, select_metric=None,
              done=True, extra=None) -> None:
    """Persist a Tianchi checkpoint: best-val + last-epoch state_dicts plus config to reconstruct."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    st = {
        "best_state": best_state if best_state is not None else last_state,
        "last_state": last_state,
        "best_ep": best_ep, "best_val_metric": best_metric,
        "in_dims": in_dims, "name": name, "seed": seed,
        "select_metric": select_metric,
        "model_cfg": model_cfg, "feature_cfg": feature_cfg,
        "done": done,
    }
    if extra:
        st.update(extra)
    torch.save(st, path)
    if done:
        print(f"  [ckpt] saved best+last -> {path}", flush=True)

NODE_TYPES = ["seeker", "job", "skill", "title"]
INTERACTION_RELS = [("seeker", "considers", "job"), ("seeker", "accepts", "job")]
ATTR_RELS = [("seeker", "has_skill", "skill"), ("seeker", "has_title", "title"),
             ("job", "requires_skill", "skill"), ("job", "requires_title", "title")]

# split label for pairs-absent seekers: never scored, excluded from every MP mask
# (else their profile edges would leak into the training graph)
UNSCORED_SPLIT = "none"


def _rev(rel):
    s, r, d = rel
    return (d, f"rev_{r}", s)


# ---------------------------------------------------------------------------
# artifact loading
# ---------------------------------------------------------------------------
@dataclass
class Artifacts:
    num_nodes: Dict[str, int]
    x: Dict[str, torch.Tensor]
    edges: Dict[Tuple[str, str, str], torch.Tensor]     # forward edge_index, cpu
    pairs: pl.DataFrame
    seeker_split: np.ndarray                            # per-seeker idx -> split str


def load_artifacts(*, art_dir=None, validate: bool = True) -> Artifacts:
    out = Path(art_dir) if art_dir is not None else OUT_DIR
    g = torch.load(out / "graph.pt", weights_only=False)
    edges = {tuple(k.split("__")): v for k, v in g["edges"].items()}
    pairs = pl.read_parquet(out / "pairs.parquet")
    n_seeker = g["num_nodes"]["seeker"]
    # pairs-absent seekers -> UNSCORED_SPLIT (excluded from every MP graph)
    ms = np.array([UNSCORED_SPLIT] * n_seeker, dtype=object)
    msdf = pairs.select(["seeker_idx", "split"]).unique()
    for mi, sp in zip(msdf["seeker_idx"].to_list(), msdf["split"].to_list()):
        ms[mi] = sp
    art = Artifacts(num_nodes=g["num_nodes"], x=g["x"], edges=edges, pairs=pairs, seeker_split=ms)
    if validate:
        validate_artifacts(art, node_maps_path=out / "node_maps.json", verbose=True)
    return art


def validate_artifacts(art: Artifacts, *, node_maps_path: Path = None,
                       verbose: bool = True) -> None:
    """Structural + leak-safety pre-flight for the loaded Tianchi Artifacts (raises on violation)."""
    fails: List[str] = []
    n_checks = 0

    def _req(ok: bool, msg: str) -> None:
        nonlocal n_checks
        n_checks += 1
        if not ok:
            fails.append(msg)

    nn = art.num_nodes
    EXPECTED_RELS = set(INTERACTION_RELS + ATTR_RELS)
    SPLITS = {"train", "val", "test"}

    # T1 node-map <-> num_nodes (a mismatched pair silently mis-indexes everything)
    p = Path(node_maps_path) if node_maps_path is not None else (OUT_DIR / "node_maps.json")
    if p.exists():
        try:
            with open(p) as f:
                maps = json.load(f)
        except Exception as e:                       # unreadable -> hard fail
            maps = None
            _req(False, f"T1 node_maps.json unreadable: {e!r}")
        if maps is not None:
            for t in NODE_TYPES:
                _req(t in maps and len(maps[t]) == nn.get(t),
                     f"T1 {t}: node_map size {len(maps.get(t, []))} != num_nodes {nn.get(t)}")
    elif verbose:
        print(f"   [tianchi validation] note: {p.name} absent -> T1 skipped")

    # T2 node-feature matrices (baked into graph.pt here, unlike Tech)
    for t in NODE_TYPES:
        if t not in art.x:
            _req(False, f"T2 {t}: missing node-feature matrix")
            continue
        xt = art.x[t]
        rows_ok = xt.dim() == 2 and int(xt.shape[0]) == nn.get(t)
        _req(rows_ok, f"T2 {t}: x shape {tuple(xt.shape)} rows != num_nodes {nn.get(t)}")
        _req(torch.is_floating_point(xt), f"T2 {t}: x dtype {xt.dtype} not floating")
        if torch.is_floating_point(xt):
            _req(bool(torch.isfinite(xt).all()), f"T2 {t}: x has non-finite values (NaN/Inf)")
        if xt.dim() == 2 and t not in ("seeker", "job"):   # skill/title node feats are square identity
            n = int(nn.get(t, -1))
            _req(int(xt.shape[1]) == n
                 and torch.equal(xt.cpu(), torch.eye(n, dtype=xt.dtype)),
                 f"T2 {t}: x is not the {n}x{n} identity (shape {tuple(xt.shape)})")

    # T3 edges: shape / dtype / index bounds / duplicate-free
    for rel, ei in art.edges.items():
        shp_ok = ei.dim() == 2 and ei.shape[0] == 2
        _req(shp_ok, f"T3 {rel}: edge_index shape {tuple(ei.shape)} not (2,E)")
        _req(ei.dtype == torch.long, f"T3 {rel}: dtype {ei.dtype} != long")
        if shp_ok and rel in EXPECTED_RELS and ei.shape[1] > 0:
            st, _, dt = rel
            ec = ei.cpu()
            _req(0 <= int(ec[0].min()) and int(ec[0].max()) < nn.get(st, 0),
                 f"T3 {rel}: src idx out of [0,{nn.get(st)})")
            _req(0 <= int(ec[1].min()) and int(ec[1].max()) < nn.get(dt, 0),
                 f"T3 {rel}: dst idx out of [0,{nn.get(dt)})")
            _req(int(torch.unique(ec, dim=1).shape[1]) == int(ec.shape[1]),
                 f"T3 {rel}: duplicate (src,dst) edges present")

    # T4 relation coverage: missing rel OR leaked target (satisfied) edge
    got = set(art.edges.keys())
    _req(got == EXPECTED_RELS,
         f"T4 edges rel set != expected (missing={EXPECTED_RELS - got}, extra={got - EXPECTED_RELS})")

    # T5 supervision pairs frame
    pr = art.pairs
    need = {"seeker_idx", "job_idx", "browsed", "delivered", "satisfied", "split", "job_exposure"}
    cols_ok = need <= set(pr.columns)
    _req(cols_ok, f"T5 pairs missing cols: {need - set(pr.columns)}")
    if cols_ok and pr.height:
        mi, ji = pr["seeker_idx"].to_numpy(), pr["job_idx"].to_numpy()
        _req(0 <= int(mi.min()) and int(mi.max()) < nn.get("seeker", 0),
             f"T5 pairs.seeker_idx out of [0,{nn.get('seeker')})")
        _req(0 <= int(ji.min()) and int(ji.max()) < nn.get("job", 0),
             f"T5 pairs.job_idx out of [0,{nn.get('job')})")
        bad_split = set(pr["split"].unique().to_list()) - SPLITS
        _req(not bad_split, f"T5 pairs.split unexpected labels {bad_split}")
        dup = (pr.select("seeker_idx", "split").unique()
                 .group_by("seeker_idx").len().filter(pl.col("len") > 1).height)
        _req(dup == 0, f"T5 seeker-disjoint split violated: {dup} seekers in >1 split")
        for c in ("browsed", "delivered", "satisfied"):
            extra = set(pr[c].unique().to_list()) - {0, 1}
            _req(not extra, f"T5 pairs.{c} not binary: {extra}")
        # unique (seeker,job): a duplicated row double-counts in loss/eval
        n_uniq = pr.select("seeker_idx", "job_idx").n_unique()
        _req(n_uniq == pr.height, f"T5 duplicate (seeker,job) pairs: {pr.height - n_uniq}")
        # job_exposure = #TRAIN interactions per job (cold-start key); recompute to catch val/test leak
        _req(int(pr["job_exposure"].min()) >= 0, "T5 pairs.job_exposure has negative values")
        recomputed = (pr.filter((pl.col("split") == "train")
                                & ((pl.col("browsed") == 1) | (pl.col("delivered") == 1)))
                        .group_by("job_idx").len().rename({"len": "_jp"}))
        jp_bad = (pr.join(recomputed, on="job_idx", how="left")
                    .with_columns(pl.col("_jp").fill_null(0))
                    .filter(pl.col("job_exposure") != pl.col("_jp")).height)
        _req(jp_bad == 0, f"T5 pairs.job_exposure != recomputed train-interaction count ({jp_bad} rows)")

    # T6 seeker_split vector (derived in load_artifacts; must align + be labelled)
    ms = art.seeker_split
    n_absent = (int(nn.get("seeker", 0)) - int(pr["seeker_idx"].n_unique())) if pr.height else 0
    _req(len(ms) == nn.get("seeker"),
         f"T6 seeker_split len {len(ms)} != #seekers {nn.get('seeker')}")
    bad_ms = set(np.unique(ms).tolist()) - (SPLITS | {UNSCORED_SPLIT})
    _req(not bad_ms, f"T6 seeker_split unexpected labels {bad_ms}")
    if cols_ok and pr.height:
        # agree with pairs' split for seekers present in pairs
        msdf = pr.select("seeker_idx", "split").unique()
        disagree = sum(1 for m_i, sp in zip(msdf["seeker_idx"].to_list(), msdf["split"].to_list())
                       if 0 <= int(m_i) < len(ms) and ms[int(m_i)] != sp)
        _req(disagree == 0, f"T6 seeker_split disagrees with pairs split for {disagree} seeker(s)")
        # pairs-absent seekers must be exactly the UNSCORED_SPLIT set (excluded from MP)
        n_unscored = int((ms == UNSCORED_SPLIT).sum())
        _req(n_unscored == n_absent,
             f"T6 unscored-seeker count {n_unscored} != pairs-absent count {n_absent}")

    # T7 edge <-> pairs count consistency: considers == #browsed, accepts == #delivered
    considers, accepts = ("seeker", "considers", "job"), ("seeker", "accepts", "job")
    if cols_ok and pr.height and {considers, accepts} <= set(art.edges):
        _req(int(art.edges[considers].shape[1]) == int(pr["browsed"].sum()),
             "T7 considers-edge count != pairs.browsed sum (graph/pairs version skew)")
        _req(int(art.edges[accepts].shape[1]) == int(pr["delivered"].sum()),
             "T7 accepts-edge count != pairs.delivered sum (graph/pairs version skew)")

    # informational: seekers absent from pairs are excluded from all MP graphs.
    if n_absent and verbose:
        print(f"   [tianchi validation] note: {n_absent} seeker(s) absent from pairs "
              f"-> marked '{UNSCORED_SPLIT}', excluded from all message-passing graphs (not scored)")

    if verbose:
        print(f"   [tianchi artifact validation] {'PASS' if not fails else 'FAIL'}: "
              f"{n_checks - len(fails)}/{n_checks} checks passed")
        for f in fails:
            print(f"      [FAIL] {f}")
    if fails:
        raise AssertionError(
            f"Tianchi Artifacts validation failed ({len(fails)} violations): {fails}")


# ---------------------------------------------------------------------------
# graph assembly
# ---------------------------------------------------------------------------
def configured_rels(fcfg: FeatureConfig) -> List[Tuple[str, str, str]]:
    rels = []
    if fcfg.use_interaction_edges:
        for r in INTERACTION_RELS:
            rels += [r, _rev(r)]
    if fcfg.use_attribute_edges:
        for r in ATTR_RELS:
            name = r[1]
            if name in ("has_skill", "requires_skill") and not fcfg.use_skill_edges:
                continue
            if name in ("has_title", "requires_title") and not fcfg.use_title_edges:
                continue
            rels += [r, _rev(r)]
    return rels


def _load_qwen_content(num_nodes: Dict[str, int], tag: str = "qwen3"):
    """Frozen node-content vectors (node-aligned) for seeker/job; tag selects the encoder."""
    out = {}
    for t, fname in (("seeker", f"{tag}_seeker_emb.pt"), ("job", f"{tag}_job_emb.pt")):
        o = torch.load(OUT_DIR / fname, weights_only=False, map_location="cpu")
        assert o.get("node_aligned"), f"{fname}: expected node_aligned=True"
        emb = o["emb"].float()
        assert emb.shape[0] == num_nodes[t], (
            f"{fname}: {emb.shape[0]} rows != {num_nodes[t]} {t} nodes")
        out[t] = emb
    return out


def make_x(art: Artifacts, fcfg: FeatureConfig, device, qwen=None) -> Dict[str, torch.Tensor]:
    x = {t: art.x[t].clone() for t in NODE_TYPES}
    if qwen is not None:                                   # append Qwen node-content
        for t in ("seeker", "job"):
            x[t] = torch.cat([x[t], qwen[t]], dim=1)
    return {t: v.to(device) for t, v in x.items()}


def make_meta(art: Artifacts, fcfg: FeatureConfig) -> GraphMeta:
    offset, cum = {}, 0
    for t in NODE_TYPES:
        offset[t] = cum
        cum += art.num_nodes[t]
    return GraphMeta(num_nodes=art.num_nodes, edge_types=configured_rels(fcfg),
                     node_offset=offset, n_total=cum)


class _CfgShim:
    """Minimal cfg with .model/.feature for models.build_model."""
    def __init__(self, model, feature):
        self.model = model
        self.feature = feature


# ---------------------------------------------------------------------------
# supervision-frame helpers
# ---------------------------------------------------------------------------
def _pf(frame: pl.DataFrame, device) -> torch.Tensor:
    """TIANCHI_MATCH_FEATURES columns of a pair frame -> [N, dim] tensor, or None if absent."""
    from .build import TIANCHI_MATCH_FEATURES
    cols = [c for c in TIANCHI_MATCH_FEATURES if c in frame.columns]
    if not cols:
        return None
    arr = frame.select(cols).to_numpy().astype(np.float32)
    return torch.from_numpy(arr).to(device)
