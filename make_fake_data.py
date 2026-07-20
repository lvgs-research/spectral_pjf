#!/usr/bin/env python3
"""Generate a fully synthetic dataset (all values fake) to exercise the pipeline.

Usage: python make_fake_data.py [--encoders] [--seed N] [--emb-dim D] ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch

PKG = Path(__file__).resolve().parent / "link_prediction_experiment"
DATA = PKG / "data"

DAY = 86400
WINDOW_START = datetime(2022, 1, 1)
WINDOW_END = datetime(2025, 5, 1)          # keep inside the [2022-01-01, 2025-06-01] split window
SPAN_DAYS = (WINDOW_END - WINDOW_START).days


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def _emb_dict(n: int, dim: int, rng: np.random.Generator) -> dict:
    """A node-aligned frozen-content embedding artifact (row i == node index i)."""
    return {"emb": torch.from_numpy(_l2(rng.standard_normal((n, dim)))),
            "node_aligned": True, "dim": int(dim)}


def _save_emb(path: Path, n: int, dim: int, rng: np.random.Generator) -> None:
    torch.save(_emb_dict(n, dim, rng), path)


def _dt(days_from_start: float) -> datetime:
    return WINDOW_START + timedelta(days=float(days_from_start))


# --------------------------------------------------------------------------- #
# TECH dataset
# --------------------------------------------------------------------------- #
def gen_tech(args, rng: np.random.Generator) -> None:
    nM, nJ = args.n_seekers, args.n_jobs
    nS, nT, nC = args.n_skills, args.n_titles, args.n_companies
    out_a = DATA / "tech"
    out_g = DATA / "tech_graph"
    out_p = DATA / "tech_prepared"
    for d in (out_a, out_g, out_p):
        d.mkdir(parents=True, exist_ok=True)

    # --- synthetic vocabularies (meaningless labels) ---
    skills = [f"Skill{i:03d}" for i in range(nS)]
    titles = [f"Title{i:02d}" for i in range(nT)]

    # --- ids (random but distinct integers) ---
    seeker_ids = rng.choice(np.arange(100000, 999999), size=nM, replace=False).tolist()
    job_ids = rng.choice(np.arange(100000, 999999), size=nJ, replace=False).tolist()
    company_ids = rng.choice(np.arange(1000, 9999), size=nC, replace=False).tolist()

    # --- node maps (contiguous 0..n-1 per type) ---
    node_maps = {
        "seeker": {f"m_{sid}": i for i, sid in enumerate(seeker_ids)},
        "job": {f"j_{jid}": i for i, jid in enumerate(job_ids)},
        "skill": {f"s_{s}": i for i, s in enumerate(skills)},
        "title": {f"t_{t}": i for i, t in enumerate(titles)},
        "company": {f"c_{c}": i for i, c in enumerate(company_ids)},
    }

    # --- assign each seeker a temporal "era" (70/20/10 train/val/test by seeker) ---
    # Birth = era - offset, so a seeker is born before its first exposure.
    order = rng.permutation(nM)
    era = np.empty(nM)                                            # activity-window center (days)
    n_tr, n_va = int(nM * 0.70), int(nM * 0.20)
    for rank, mi in enumerate(order):
        if rank < n_tr:
            era[mi] = rng.uniform(0.08 * SPAN_DAYS, 0.55 * SPAN_DAYS)
        elif rank < n_tr + n_va:
            era[mi] = rng.uniform(0.60 * SPAN_DAYS, 0.75 * SPAN_DAYS)
        else:
            era[mi] = rng.uniform(0.80 * SPAN_DAYS, 0.95 * SPAN_DAYS)
    mem_birth = np.maximum(era - rng.uniform(30, 300, size=nM), 0.0)   # born before activity
    # jobs are born spread across the early/mid span (before they are exposed)
    job_birth = rng.uniform(0, SPAN_DAYS * 0.7, size=nJ)

    # --- interaction rows (each seeker: a cluster of exposures in its era) ---
    rows = []
    per_seeker = args.rows_per_seeker
    for mi in range(nM):
        sid = seeker_ids[mi]
        base = era[mi]
        k = rng.integers(max(3, per_seeker - 3), per_seeker + 4)
        # pick jobs that already exist by this seeker's activity time (job born earlier)
        eligible = np.where(job_birth < base - 5)[0]
        if eligible.size < k:
            eligible = np.argsort(job_birth)[: max(k, 10)]
        chosen = rng.choice(eligible, size=min(k, eligible.size), replace=False)
        for ji in chosen:
            jid = job_ids[ji]
            sday = base + rng.uniform(0, 45)                     # clustered within ~45d
            sday = min(sday, SPAN_DAYS - 2)
            s_dt = _dt(sday)
            passed = rng.random() < 0.38
            accepted = passed or (rng.random() < 0.45)           # accept precedes pass
            acc_dt = s_dt + timedelta(days=float(rng.uniform(1, 8))) if accepted else None
            if passed:
                sp = True
                pass_dt = (acc_dt or s_dt) + timedelta(days=float(rng.uniform(1, 10)))
                neg_dt = None
            else:
                sp = False
                pass_dt = None
                neg_dt = (acc_dt or s_dt) + timedelta(days=float(rng.uniform(1, 12)))
            rows.append({
                "seeker_id": sid, "job_id": jid,
                "exposure_date": s_dt.date(), "exposure_datetime": s_dt,
                "pos_label": sp,
                "pos_datetime": pass_dt,
                "accept_datetime": acc_dt,
                "neg_datetime_a": neg_dt,
                "neg_datetime_b": None,
                "neg_datetime_c": None,
                "snapshot_date": WINDOW_END.date(),
            })
    inter = pl.DataFrame(rows, schema_overrides={
        "pos_label": pl.Boolean,
        "exposure_datetime": pl.Datetime("us"),
        "pos_datetime": pl.Datetime("us"),
        "accept_datetime": pl.Datetime("us"),
        "neg_datetime_a": pl.Datetime("us"),
        "neg_datetime_b": pl.Datetime("us"),
        "neg_datetime_c": pl.Datetime("us"),
    })
    inter.write_parquet(out_a / "interactions.parquet")

    # --- per-seeker / per-job attribute sets (skills/titles) ---
    mem_skills = [rng.choice(nS, size=int(rng.integers(2, 6)), replace=False).tolist() for _ in range(nM)]
    mem_titles = [rng.choice(nT, size=int(rng.integers(1, 3)), replace=False).tolist() for _ in range(nM)]
    job_skills = [rng.choice(nS, size=int(rng.integers(2, 7)), replace=False).tolist() for _ in range(nJ)]
    job_titles = [rng.choice(nT, size=int(rng.integers(1, 3)), replace=False).tolist() for _ in range(nJ)]
    job_company = rng.integers(0, nC, size=nJ)

    # --- seekers.parquet ---
    def _norm_skills(idxs):
        return [{"skill": skills[s]} for s in idxs]

    seeker_rows = []
    for mi in range(nM):
        seeker_rows.append({
            "seeker_id": seeker_ids[mi],
            "snapshot_date": WINDOW_END.date(),
            "profile_text_emb": rng.standard_normal(768).astype(np.float64).tolist(),
            # raw seeker profile text (placeholder; input to embed_text_tech.py)
            "profile_text": "Candidate profile. Skills: "
                            + ", ".join(skills[s] for s in mem_skills[mi]) + ".",
            "skill_set": _norm_skills(mem_skills[mi]),
            # tabular node features: opaque numeric noise (identity preprocessing)
            "feature_0": float(rng.standard_normal()),
            "feature_1": float(rng.standard_normal()),
            "feature_18": float(rng.standard_normal()),
            # opaque categorical attributes for the match features
            **{f"feature_{2 + k}": int(rng.integers(0, 5)) for k in range(8)},
        })
    pl.DataFrame(seeker_rows, schema_overrides={
        "profile_text_emb": pl.List(pl.Float64),
        "snapshot_date": pl.Date,
    }).write_parquet(out_a / "seekers.parquet")

    # --- jobs.parquet ---
    job_rows = []
    for ji in range(nJ):
        job_rows.append({
            "job_id": job_ids[ji],
            "snapshot_date": WINDOW_END.date(),
            "job_text_emb": rng.standard_normal(768).astype(np.float64).tolist(),
            # raw job text (placeholder; input to embed_text_tech.py)
            "job_title": "Hiring: " + (skills[job_skills[ji][0]] if job_skills[ji] else "generalist"),
            "job_description": "Requirements include "
                               + ", ".join(skills[s] for s in job_skills[ji]) + ".",
            "job_feature": "Attributes: " + ", ".join(skills[s] for s in job_skills[ji][:2]) + ".",
            "skill_set": _norm_skills(job_skills[ji]),
            # tabular node features: opaque numeric noise (identity preprocessing)
            "feature_10": float(rng.standard_normal()),
            "feature_11": float(rng.standard_normal()),
            "feature_12": float(rng.standard_normal()),
            "feature_13": float(rng.standard_normal()),
            # opaque categorical attributes for the match features
            **{f"feature_{14 + k}": int(rng.integers(0, 5)) for k in range(8)},
        })
    pl.DataFrame(job_rows, schema_overrides={
        "job_text_emb": pl.List(pl.Float64),
        "snapshot_date": pl.Date,
    }).write_parquet(out_a / "jobs.parquet")


    # --- skills vocabulary ---
    pl.DataFrame({"skill_name": skills}).write_parquet(out_a / "skills_vocab.parquet")

    # --- node maps json ---
    with open(out_g / "node_maps.json", "w") as f:
        json.dump(node_maps, f)

    # --- hetero graph (PyG HeteroData) ---
    from torch_geometric.data import HeteroData
    g = HeteroData()
    g["seeker"].num_nodes = nM
    g["job"].num_nodes = nJ
    g["skill"].num_nodes = nS
    g["title"].num_nodes = nT
    g["company"].num_nodes = nC

    # interaction edges derived from the interaction frame (event times AFTER expose)
    m_idx = np.array([node_maps["seeker"][f"m_{r}"] for r in inter["seeker_id"].to_list()])
    j_idx = np.array([node_maps["job"][f"j_{r}"] for r in inter["job_id"].to_list()])
    exp_ts = np.array([int(dt.timestamp()) for dt in inter["exposure_datetime"].to_list()], dtype=np.int64)

    def _ts_of(col):
        return np.array([int(dt.timestamp()) if dt is not None else -1
                         for dt in inter[col].to_list()], dtype=np.int64)
    acc_ts = _ts_of("accept_datetime")
    pass_ts = _ts_of("pos_datetime")

    # considers: every exposure, event = expose + 1 day (post-expose)
    cons_t = exp_ts + DAY
    _add_rel(g, ("seeker", "considers", "job"), m_idx, j_idx, cons_t)
    # accepts: rows with an accept event
    am = acc_ts >= 0
    _add_rel(g, ("seeker", "accepts", "job"), m_idx[am], j_idx[am], acc_ts[am])
    # screens: job screens seeker, on the pass event
    pm = pass_ts >= 0
    _add_rel(g, ("job", "screens", "seeker"), j_idx[pm], m_idx[pm], pass_ts[pm])

    # attribute edges (time = source-entity birth, as unix seconds)
    mem_birth_ts = (WINDOW_START.timestamp() + mem_birth * DAY).astype(np.int64)
    job_birth_ts = (WINDOW_START.timestamp() + job_birth * DAY).astype(np.int64)
    _add_attr(g, ("seeker", "has_skill", "skill"), mem_skills, mem_birth_ts)
    _add_attr(g, ("seeker", "has_title", "title"), mem_titles, mem_birth_ts)
    _add_attr(g, ("job", "requires_skill", "skill"), job_skills, job_birth_ts)
    _add_attr(g, ("job", "requires_title", "title"), job_titles, job_birth_ts)
    _add_attr(g, ("job", "posted_by", "company"), [[c] for c in job_company], job_birth_ts)

    torch.save(g, out_g / "hetero_graph.pt")

    # --- frozen node-content embeddings ---
    _save_emb(out_p / "qwen3_seeker_emb.pt", nM, args.emb_dim, rng)
    _save_emb(out_p / "qwen3_job_emb.pt", nJ, args.emb_dim, rng)
    if args.encoders:
        for tag in ("me5", "qwen3_4b", "qwen3_8b"):
            _save_emb(out_p / f"{tag}_seeker_emb.pt", nM, args.emb_dim, rng)
            _save_emb(out_p / f"{tag}_job_emb.pt", nJ, args.emb_dim, rng)

    print(f"[tech] {nM} seekers / {nJ} jobs / {nS} skills / {nT} titles / {nC} companies; "
          f"{inter.height} interaction rows")


def _add_rel(g, rel, src, dst, t):
    src = np.asarray(src, dtype=np.int64); dst = np.asarray(dst, dtype=np.int64)
    t = np.asarray(t, dtype=np.int64)
    if src.size == 0:                       # keep the relation present but empty
        g[rel].edge_index = torch.zeros((2, 0), dtype=torch.long)
        g[rel].time = torch.zeros((0,), dtype=torch.long)
        return
    g[rel].edge_index = torch.from_numpy(np.stack([src, dst])).long()
    g[rel].time = torch.from_numpy(t).long()


def _add_attr(g, rel, per_src_targets, src_birth_ts):
    src, dst, t = [], [], []
    for s, targets in enumerate(per_src_targets):
        for d in targets:
            src.append(s); dst.append(int(d)); t.append(int(src_birth_ts[s]))
    _add_rel(g, rel, np.array(src), np.array(dst), np.array(t))


# --------------------------------------------------------------------------- #
# TIANCHI dataset
# --------------------------------------------------------------------------- #
def _salary_code(rng) -> str:
    lo = int(rng.integers(1, 30)) * 1000
    hi = lo + int(rng.integers(1, 30)) * 1000
    return f"{lo:05d}{hi:05d}"


def gen_tianchi(args, rng: np.random.Generator) -> None:
    """Emit fake raw Tianchi tables, then run the real graph builder on them."""
    raw = DATA / "tianchi"
    prep = DATA / "tianchi_prepared"
    prep_bv = DATA / "tianchi_prepared_bv"
    raw.mkdir(parents=True, exist_ok=True)

    # 1) fake raw tables in the builder's input schema
    _write_tianchi_raw(raw, args.tc_seekers, args.tc_jobs, args.tc_skills,
                       args.tc_titles, args.tc_pairs, rng)

    # 2) run the real builder (boundary_validate=False = jieba-free substring skill-miner;
    #    the _bv variant coincides on fake ASCII text)
    from link_prediction_experiment.tianchi_pjf import build as tc_build
    tc_build.main(boundary_validate=False, out_dir=prep)
    tc_build.main(boundary_validate=False, out_dir=prep_bv)

    # 3) node-aligned frozen content embeddings, sized to the built node maps
    nm = json.load(open(prep / "node_maps.json"))
    nM_b, nJ_b = len(nm["seeker"]), len(nm["job"])
    _save_emb(prep / "qwen3_seeker_emb.pt", nM_b, args.emb_dim, rng)
    _save_emb(prep / "qwen3_job_emb.pt", nJ_b, args.emb_dim, rng)
    if args.encoders:
        for tag in ("me5", "qwen3_4b", "qwen3_8b"):
            _save_emb(prep / f"{tag}_seeker_emb.pt", nM_b, args.emb_dim, rng)
            _save_emb(prep / f"{tag}_job_emb.pt", nJ_b, args.emb_dim, rng)

    # 4) validate the built artifacts (experiment.validate_artifacts T1-T7)
    from link_prediction_experiment.tianchi_pjf.experiment import load_artifacts
    for d in (prep, prep_bv):
        load_artifacts(art_dir=d, validate=True)
    print(f"[tianchi] built via tianchi_pjf/build.py from raw tables: "
          f"{nM_b} seekers / {nJ_b} jobs / {len(nm['skill'])} skills / {len(nm['title'])} titles")


# Chinese ordinal vocab the builder decodes (degree / min-experience codes).
_TC_DEGREES = ["\u9ad8\u4e2d", "\u5927\u4e13", "\u672c\u79d1", "\u7855\u58eb"]
_TC_MINYEARS = ["-1", "1", "103", "3", "305", "5"]


def _write_tianchi_raw(raw, nM, nJ, nS, nT, n_pairs, rng):
    """Fake raw Tianchi tables in the column layout the builder reads."""
    def _w(path, header, rows):
        with open(path, "w") as f:
            f.write("\t".join(header) + "\n")
            for r in rows:
                f.write("\t".join(str(c) for c in r) + "\n")

    skills = [f"skl{i:02d}" for i in range(nS)]        # <= 8 chars (SKILL_MAX_LEN)
    titles = [f"ttl{i:02d}" for i in range(nT)]
    user_ids = [f"u{i:05d}" for i in range(nM)]
    jd_ids = [f"j{i:05d}" for i in range(nJ)]
    cities = [str(c) for c in range(1, 21)]

    u_rows = []
    for u in user_ids:
        exp = "|".join(rng.choice(skills, size=int(rng.integers(3, 7)), replace=False))
        des_t = ",".join(rng.choice(titles, size=int(rng.integers(1, 4)), replace=False))
        des_c = ",".join(rng.choice(cities, size=int(rng.integers(1, 4)), replace=False))
        u_rows.append([
            u, str(rng.choice(cities)), des_c, str(int(rng.integers(1, 10))),
            des_t, _salary_code(rng), str(rng.choice(titles)), str(int(rng.integers(1, 10))),
            _salary_code(rng), str(rng.choice(_TC_DEGREES)), str(int(rng.integers(22, 46))),
            str(int(rng.integers(2005, 2020))), exp,
        ])
    _w(raw / "table1_user.txt",
       ["user_id", "live_city_id", "desire_jd_city_id", "desire_jd_industry_id",
        "desire_jd_type_id", "desire_jd_salary_id", "cur_jd_type", "cur_industry_id",
        "cur_salary_id", "cur_degree_id", "birthday", "start_work_date", "experience"],
       u_rows)

    j_rows = []
    for j in jd_ids:
        sub = str(rng.choice(titles))
        desc = " ".join(rng.choice(skills, size=int(rng.integers(3, 6)), replace=False)) + " " + sub
        lo = int(rng.integers(5, 20)) * 1000
        hi = lo + int(rng.integers(3, 20)) * 1000
        j_rows.append([j, f"Position {sub}", sub, str(rng.choice(cities)), str(lo), str(hi),
                       str(int(rng.integers(1, 6))), str(int(rng.integers(0, 2))),
                       str(rng.choice(_TC_DEGREES)), str(rng.choice(_TC_MINYEARS)), "\\N", desc])
    # jd_title: used by tianchi_pjf.embed_text, not the graph builder; mirrors table2_jd layout.
    _w(raw / "table2_jd.txt",
       ["jd_no", "jd_title", "jd_sub_type", "city", "min_salary", "max_salary", "require_nums",
        "is_travel", "min_edu_level", "min_years", "key", "job_description"],
       j_rows)

    seen, a_rows = set(), []
    for ui, u in enumerate(user_ids):                  # every user gets >= 1 action
        for jj in rng.choice(nJ, size=int(rng.integers(1, 6)), replace=False):
            jj = int(jj)
            if (ui, jj) in seen:
                continue
            seen.add((ui, jj))
            d = 1 if rng.random() < 0.5 else 0
            sat = 1 if (d and rng.random() < 0.4) else 0
            a_rows.append([u, jd_ids[jj], "1", str(d), str(sat)])
    while len(a_rows) < n_pairs:                        # top up to the target volume
        ui = int(rng.integers(0, nM)); jj = int(rng.integers(0, nJ))
        if (ui, jj) in seen:
            continue
        seen.add((ui, jj))
        d = 1 if rng.random() < 0.5 else 0
        sat = 1 if (d and rng.random() < 0.4) else 0
        a_rows.append([user_ids[ui], jd_ids[jj], "1", str(d), str(sat)])
    _w(raw / "table3_action.txt",
       ["user_id", "jd_no", "browsed", "delivered", "satisfied"], a_rows)


# --------------------------------------------------------------------------- #
def _wipe_stale_cache():
    """Delete runtime artifacts derived from the data being regenerated (gitignored, safe)."""
    import shutil
    tables_data = PKG.parent / "tables" / "data"
    for d in (PKG / "_cache",
              PKG / "_results", PKG / "_checkpoints",
              PKG / "tianchi_pjf" / "_results", PKG / "tianchi_pjf" / "_checkpoints",
              tables_data / "embeddings", tables_data / "embeddings_encoders"):
        if d.exists():
            shutil.rmtree(d)
            print(f"[wipe] removed stale derived artifacts: {d.relative_to(PKG.parent)}")


def main():
    ap = argparse.ArgumentParser(description="Generate a fully synthetic dataset for the suite.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emb-dim", dest="emb_dim", type=int, default=64,
                    help="frozen node-content embedding dim (all encoder tags)")
    ap.add_argument("--encoders", action="store_true",
                    help="also emit me5 / qwen3_4b / qwen3_8b embeddings (encoder-axis sweep)")
    # Tech (tech) scale
    ap.add_argument("--n-seekers", dest="n_seekers", type=int, default=300)
    ap.add_argument("--n-jobs", dest="n_jobs", type=int, default=500)
    ap.add_argument("--n-skills", dest="n_skills", type=int, default=40)
    ap.add_argument("--n-titles", dest="n_titles", type=int, default=15)
    ap.add_argument("--n-companies", dest="n_companies", type=int, default=60)
    ap.add_argument("--rows-per-seeker", dest="rows_per_seeker", type=int, default=12)
    # Tianchi scale
    ap.add_argument("--tc-seekers", dest="tc_seekers", type=int, default=400)
    ap.add_argument("--tc-jobs", dest="tc_jobs", type=int, default=800)
    ap.add_argument("--tc-skills", dest="tc_skills", type=int, default=60)
    ap.add_argument("--tc-titles", dest="tc_titles", type=int, default=20)
    ap.add_argument("--tc-pairs", dest="tc_pairs", type=int, default=15000)
    ap.add_argument("--out", default=str(DATA))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    _wipe_stale_cache()
    print("Generating SYNTHETIC data — every value is fake (see module docstring).")
    gen_tech(args, rng)
    gen_tianchi(args, rng)
    print("\nDone. Artifacts under", DATA)


if __name__ == "__main__":
    main()
