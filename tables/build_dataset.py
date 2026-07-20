"""Consolidate the 5-seed main-paper results into a self-contained copy under tables/data/,
so table generation reads only from here.

Run: python tables/build_dataset.py  (re-runnable & idempotent)
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                       # <suite>/tables/
PKG = HERE.parent / "link_prediction_experiment"              # the experiment package
RES = PKG / "_results"
TC_RES = PKG / "tianchi_pjf" / "_results"
SHARDS = RES / "rescore_shards_tech"
OUT = HERE / "data"
SEEDS = [0, 1, 2, 3, 4]

# ------------------------------------------------------------------ helpers
def flatten(d, prefix=""):
    """Nested metric dict -> {dotted_leaf: float} (scalars only)."""
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            nk = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(flatten(v, nk))
            elif isinstance(v, bool):
                continue
            elif isinstance(v, (int, float)):
                out[nk] = float(v)
    return out


def _write(ds, experiment, conv, conv_slug, encoder, enc_slug, placement, protocol,
           metric_field, source, per_seed):
    """per_seed: list of {seed, metrics:{flat}} — must be exactly the 5 SEEDS."""
    seeds_present = sorted(r["seed"] for r in per_seed)
    rec = {
        "dataset": ds, "experiment": experiment, "conv": conv, "encoder": encoder,
        "content_placement": placement, "protocol": protocol,
        "metric_field": metric_field, "source": source,
        "seeds": seeds_present, "n_seeds": len(seeds_present),
        "per_seed": sorted(per_seed, key=lambda r: r["seed"]),
    }
    dest = OUT / ds / f"{experiment}__{conv_slug}__{enc_slug}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(dest, "w"), indent=2)
    ok = seeds_present == SEEDS
    print(f"  {'OK ' if ok else '!! '}{ds}/{experiment:3s} {conv:12s} {encoder:12s} "
          f"seeds={seeds_present} n_metrics={len(per_seed[0]['metrics'])} -> {dest.relative_to(HERE)}")
    if not ok:
        print(f"     WARNING: expected seeds {SEEDS}, got {seeds_present}", file=sys.stderr)
    return ok


PLACEMENT = {
    "p1": "content MLP (no graph)",
    "p2": "coupled: content INSIDE message passing",
    "p3": "graph-only GNN (content-off)",
    "p8": "decoupled: content-MLP || graph-GNN (strict p1||p3), late fusion",
}

# ------------------------------------------------------------------ loaders
def from_shards(tag, arm):
    per_seed = []
    for s in SEEDS:
        p = SHARDS / f"{tag}__{arm}__seed{s}.json"
        if not p.exists():
            print(f"     MISSING shard {p.name}", file=sys.stderr); continue
        per_seed.append({"seed": s, "metrics": flatten(json.load(open(p))["test_exact"])})
    return per_seed


def from_perseed_json(path, arm=None, field="test_exact"):
    """Tech-style result JSON (list of arm entries); arm=None -> first entry, else match by name."""
    d = json.load(open(path))
    entry = d[0] if arm is None else next(e for e in d if e.get("name") == arm)
    return [{"seed": ps["seed"], "metrics": flatten(ps[field])} for ps in entry["per_seed"]]


def from_rows(path, arm, field="test"):
    """Tianchi-style result JSON: flat list of (arm,seed) rows."""
    d = json.load(open(path))
    return [{"seed": r["seed"], "metrics": flatten(r[field])}
            for r in d if r.get("name") == arm]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_ok = True
    SAGE, GAT = ("HeteroSAGE", "sage"), ("HeteroGATv2", "gatv2")
    QW, QWs = "Qwen3-0.6B", "qwen3-0.6b"
    ep = "1-day exact PIT"
    hy = "incremental-visibility hybrid"
    tech_p1_sage = None   # GATv2 p1 == SAGE p1 (conv-invariant content MLP)

    # ---- Tech controlled family, HeteroSAGE (re-scored shards) ----
    print("== Tech controlled family — HeteroSAGE (Qwen3-0.6B; re-scored, exact PIT) ==")
    for exp, arm, tag in [("p1", "p1_mlp", "paired_tech_v2"), ("p2", "p2_sage_con", "paired_tech_v2"),
                          ("p3", "p3_sage_coff", "paired_tech_v2"), ("p8", "p8_parallel", "paired_tech_ctrl")]:
        ps = from_shards(tag, arm)
        if exp == "p1":
            tech_p1_sage = ps
        if not ps:
            all_ok = False; continue
        all_ok &= _write("tech", exp, *SAGE, QW, QWs, PLACEMENT[exp], ep, "test_exact",
                         f"rescore_shards_tech/{tag}__{arm}__seed{{0..4}}.json "
                         f"(re-scored via analysis/rescore_paired_tech.py, current evaluate.py)", ps)

    # ---- Tech controlled family, HeteroGATv2 (model-grid hetgat) ----
    print("== Tech controlled family — HeteroGATv2 (Qwen3-0.6B; model-grid, exact PIT) ==")
    mg = RES / "paired_tech_modelgrid_results.json"
    # p1 GATv2 == p1 SAGE (content MLP has no graph conv)
    if tech_p1_sage:
        all_ok &= _write("tech", "p1", *GAT, QW, QWs, PLACEMENT["p1"], ep, "test_exact",
                         "== HeteroSAGE p1 (content MLP has no graph conv -> conv-invariant)", tech_p1_sage)
    for exp, arm in [("p2", "p2_hetgat"), ("p3", "p3_hetgat"), ("p8", "p8_hetgat")]:
        ps = from_perseed_json(mg, arm=arm, field="test_exact")
        all_ok &= _write("tech", exp, *GAT, QW, QWs, PLACEMENT[exp], ep, "test_exact",
                         f"_results/paired_tech_modelgrid_results.json (arm {arm})", ps)

    # ---- Tech controlled family, HeteroSAGE, encoder axis (mE5 / Qwen3-4B / Qwen3-8B) ----
    print("== Tech controlled family — HeteroSAGE, encoder axis me5/qwen3_4b/qwen3_8b (exact PIT) ==")
    for enc, slug in [("mE5-large", "me5"), ("Qwen3-4B", "qwen3-4b"), ("Qwen3-8B", "qwen3-8b")]:
        ftok = slug.replace("-", "_")
        c123 = RES / f"paired_tech_ctrl123_{ftok}_results.json"       # p1/p2/p3
        for exp, arm in [("p1", "p1_mlp"), ("p2", "p2_sage_con"), ("p3", "p3_sage_coff")]:
            all_ok &= _write("tech", exp, *SAGE, enc, slug, PLACEMENT[exp], ep, "test_exact",
                             f"_results/{c123.name} (arm {arm})",
                             from_perseed_json(c123, arm=arm, field="test_exact"))
        cp8 = RES / f"paired_tech_ctrl_{ftok}_results.json"           # p8 (single arm)
        all_ok &= _write("tech", "p8", *SAGE, enc, slug, PLACEMENT["p8"], ep, "test_exact",
                         f"_results/{cp8.name}", from_perseed_json(cp8, field="test_exact"))

    # ---- Tianchi controlled family, HeteroSAGE ----
    print("== Tianchi controlled family — HeteroSAGE (Qwen3-0.6B; hybrid) ==")
    tcp = TC_RES / "paired_tc_ctrl_results.json"
    tc_p1_sage = None
    for exp, arm in [("p1", "p1_ctrl"), ("p2", "p2_ctrl"), ("p3", "p3_ctrl"), ("p8", "p8_ctrl")]:
        ps = from_rows(tcp, arm)
        if exp == "p1":
            tc_p1_sage = ps
        all_ok &= _write("tianchi", exp, *SAGE, QW, QWs, PLACEMENT[exp], hy, "test",
                         f"tianchi_pjf/_results/{tcp.name} (arm {arm})", ps)

    # ---- Tianchi controlled family, HeteroGATv2 (model-grid hetgat) ----
    print("== Tianchi controlled family — HeteroGATv2 (Qwen3-0.6B; model-grid, hybrid) ==")
    tmg = TC_RES / "paired_tc_modelgrid_results.json"
    if tc_p1_sage:
        all_ok &= _write("tianchi", "p1", *GAT, QW, QWs, PLACEMENT["p1"], hy, "test",
                         "== HeteroSAGE p1 (content MLP has no graph conv -> conv-invariant)", tc_p1_sage)
    for exp, arm in [("p2", "p2_hetgat"), ("p3", "p3_hetgat"), ("p8", "p8_hetgat")]:
        ps = from_rows(tmg, arm)
        all_ok &= _write("tianchi", exp, *GAT, QW, QWs, PLACEMENT[exp], hy, "test",
                         f"tianchi_pjf/_results/{tmg.name} (arm {arm})", ps)

    # ---- Tianchi controlled family, HeteroSAGE, encoder axis (mE5 / Qwen3-4B / Qwen3-8B) ----
    print("== Tianchi controlled family — HeteroSAGE, encoder axis me5/qwen3_4b/qwen3_8b (hybrid) ==")
    for enc, slug in [("mE5-large", "me5"), ("Qwen3-4B", "qwen3-4b"), ("Qwen3-8B", "qwen3-8b")]:
        ftok = slug.replace("-", "_")
        c123 = TC_RES / f"paired_tc_ctrl123_{ftok}_results.json"     # p1/p2/p3
        for exp, arm in [("p1", "p1_ctrl"), ("p2", "p2_ctrl"), ("p3", "p3_ctrl")]:
            all_ok &= _write("tianchi", exp, *SAGE, enc, slug, PLACEMENT[exp], hy, "test",
                             f"tianchi_pjf/_results/{c123.name} (arm {arm})", from_rows(c123, arm))
        cp8 = TC_RES / f"paired_tc_ctrl_{ftok}_results.json"         # p8
        all_ok &= _write("tianchi", "p8", *SAGE, enc, slug, PLACEMENT["p8"], hy, "test",
                         f"tianchi_pjf/_results/{cp8.name}", from_rows(cp8, "p8_ctrl"))

    manifest = sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*.json")
                      if p.name != "MANIFEST.json")
    json.dump(manifest, open(OUT / "MANIFEST.json", "w"), indent=2)
    print(f"\n{'ALL 5-SEED OK' if all_ok else 'INCOMPLETE (see warnings)'} — "
          f"{len(manifest)} experiment files under {OUT.relative_to(PKG.parent)}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
