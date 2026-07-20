"""Build the Tianchi PJF heterogeneous graph artifact.

The SATISFIED (label) edge is never put in the graph; vocab is fit on the TRAIN split only.
Outputs -> data/tianchi_prepared/: node_maps.json, graph.pt, pairs.parquet.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR.parent / "data"
TIANCHI = DATA_DIR / "tianchi"
OUT_DIR = DATA_DIR / "tianchi_prepared"
OUT_DIR.mkdir(exist_ok=True)

# --- vocab / feature knobs -------------------------------------------------
SKILL_MIN_USERS = 10      # keep experience terms used by >= this many seekers
SKILL_MIN_LEN = 2         # Chinese chars
SKILL_MAX_LEN = 8
# denoise + bound job-side mined skills (else ~8.7M edges -> OOM)
SKILL_PER_JOB_CAP = 15    # keep <= this many (rarest) mined skills per job
SKILL_MAX_DF_FRAC = 0.35  # drop mined skills present in > this fraction of jobs
TITLE_MIN_FREQ = 3        # keep title values with >= this many (seeker or job) uses
DESC_CAP = 400            # cap job_description length for skill mining
SPLIT_SEED = 20260702
VAL_FRAC, TEST_FRAC = 0.20, 0.10   # 70/20/10 train/val/test, aligned with Tech


def _norm(s: str) -> str:
    return s.strip() if isinstance(s, str) else s


def load_tables():
    users = pl.read_csv(TIANCHI / "table1_user.txt", separator="\t", quote_char=None,
                        infer_schema_length=0)
    jds = pl.read_csv(TIANCHI / "table2_jd.txt", separator="\t", quote_char=None,
                      infer_schema_length=0, truncate_ragged_lines=True)
    act = pl.read_csv(TIANCHI / "table3_action.txt", separator="\t", quote_char=None,
                      infer_schema_length=0)
    act = act.with_columns([pl.col(c).cast(pl.Int64) for c in ("browsed", "delivered", "satisfied")])
    act = act.group_by(["user_id", "jd_no"]).agg([
        pl.col("browsed").max(), pl.col("delivered").max(), pl.col("satisfied").max()])
    return users, jds, act


def build_node_maps(users, jds, act, split_map):
    # seekers = all seekers that appear in actions
    seeker_keys = sorted(set(act["user_id"].unique().to_list()))
    seeker_map = {k: i for i, k in enumerate(seeker_keys)}

    # jobs = ENGAGED (browsed∪delivered) jds present in the JD master
    master_jobs = set(jds["jd_no"].unique().to_list())
    engaged = act.filter((pl.col("browsed") == 1) | (pl.col("delivered") == 1))
    job_keys = sorted(set(engaged["jd_no"].unique().to_list()) & master_jobs)
    job_map = {k: i for i, k in enumerate(job_keys)}

    # ---- vocab FIT on TRAIN split only (leak-safe: val/test-only terms discarded at transform) ----
    train_users = [k for k, v in split_map.items() if v == "train"]
    u_tr = users.filter(pl.col("user_id").is_in(train_users))
    train_job_keys = sorted(
        set(engaged.filter(pl.col("user_id").is_in(train_users))["jd_no"].unique().to_list())
        & master_jobs)
    j_tr = jds.filter(pl.col("jd_no").is_in(train_job_keys))
    train_job_mask = np.zeros(len(job_map), dtype=bool)
    for k in train_job_keys:
        train_job_mask[job_map[k]] = True

    # ---- title vocab: TRAIN seeker cur_jd_type ∪ TRAIN-engaged job jd_sub_type ----
    # seeker title = CURRENT occupation only (desired titles dropped)
    u_curtitles = u_tr.select(pl.col("cur_jd_type").str.strip_chars().alias("t"))["t"].drop_nulls()
    j_titles = j_tr.select(pl.col("jd_sub_type").str.strip_chars().alias("t"))["t"].drop_nulls()
    tcount = {}
    for series in (u_curtitles, j_titles):
        for t in series.to_list():
            if t and t not in ("-", "null", "\\N"):
                tcount[t] = tcount.get(t, 0) + 1
    title_keys = sorted([t for t, c in tcount.items() if c >= TITLE_MIN_FREQ])
    title_map = {k: i for i, k in enumerate(title_keys)}

    # ---- skill vocab: TRAIN seeker experience terms (freq >= SKILL_MIN_USERS) ----
    exp = (u_tr.select(pl.col("experience").str.split("|").explode()
                       .str.strip_chars().alias("kw"))["kw"].drop_nulls())
    scount = {}
    for kw in exp.to_list():
        if kw and SKILL_MIN_LEN <= len(kw) <= SKILL_MAX_LEN:
            scount[kw] = scount.get(kw, 0) + 1
    skill_keys = sorted([k for k, c in scount.items() if c >= SKILL_MIN_USERS])
    skill_map = {k: i for i, k in enumerate(skill_keys)}

    return ({"seeker": seeker_map, "job": job_map, "skill": skill_map, "title": title_map},
            train_job_mask)


# feature hashing for the content bag (shared skill+title space)


DEGREE_ORD = {"初中": 0, "中技": 1, "中专": 1, "高中": 1, "大专": 2, "本科": 3,
              "MBA": 4, "硕士": 4, "博士": 5, "其他": 2}
EDU_ORD = {"初中": 0, "中技": 1, "中专": 1, "高中": 1, "大专": 2, "本科": 3, "硕士": 4, "博士": 5}
MINYEARS_ORD = {"-1": 0, "1": 0, "103": 1, "3": 1, "305": 2, "5": 2, "510": 3, "1099": 4}
# actual year lower-bounds for exp_meets (distinct from the compressed MINYEARS_ORD)
MINYEARS_LOWER = {"-1": 0, "0": 0, "1": 0, "103": 1, "3": 1, "305": 3, "5": 5, "510": 5, "1099": 10}

# --- person-job MATCH pair features (leak-free static; order is canonical) ---
TIANCHI_MATCH_FEATURES = [
    "desire_city_match",   # job.city in seeker desired cities
    "live_city_match",     # job.city == seeker live_city
    "salary_in_range",     # desired band overlaps job [min,max]
    "salary_missing",      # desired band negotiable/dirty OR job band unspecified
    "salary_gap",          # log1p(desired_mid) - log1p(job_mid)  (signed)
    "cur_salary_gap",      # log1p(job_mid) - log1p(cur_mid)  (is this a raise?)
    "cur_salary_missing",  # cur band negotiable/dirty
    "title_match",         # job.jd_sub_type in seeker (desired titles U cur_jd_type)
    "skill_jaccard",       # |m_skills & j_skills| / |m_skills | j_skills|
    "skill_overlap_count", # log1p(|m_skills & j_skills|)
    "edu_meets",           # seeker degree >= job min_edu
    "edu_gap",             # (seeker_deg - job_min_edu) / 5
    "exp_meets",           # tenure >= job min_years_lower
    "exp_gap",             # (tenure - min_years_lower) / 10
]
TIANCHI_MATCH_DIM = len(TIANCHI_MATCH_FEATURES)


def decode_salary_band(code):
    """Zhaopin salary code -> (lo, hi) monthly-yuan band, or None if negotiable/dirty."""
    if code is None:
        return None
    s = str(code).strip()
    if s in ("", "-", "null", "\\N") or not s.isdigit():
        return None
    if len(s) == 9:
        s = "0" + s
    elif len(s) == 11:
        s = "0" + s
    if len(s) == 10:
        lo, hi = int(s[:5]), int(s[5:])
    elif len(s) == 12:
        lo, hi = int(s[:6]), int(s[6:])
    else:
        return None
    if lo == 0 and hi == 0:            # 0000000000 negotiable
        return None
    if hi == 99999 or hi <= lo:        # open-ended top band
        hi = None
    return (lo, hi)


def _band_mid(band):
    """Representative monthly value of a decoded band (open-ended -> the lower bound)."""
    if band is None:
        return None
    lo, hi = band
    return lo if hi is None else (lo + hi) / 2.0


def build_seeker_features(users, seeker_map):
    u = users.filter(pl.col("user_id").is_in(list(seeker_map.keys())))
    # order by seeker idx
    u = u.with_columns(pl.col("user_id").replace_strict(seeker_map).alias("_idx")).sort("_idx")

    def _f(col):
        return pl.col(col).cast(pl.Float64, strict=False)

    age = _f("birthday").fill_null(30.0)
    tenure = (2019 - _f("start_work_date")).clip(0, 45).fill_null(3.0)
    degree = pl.col("cur_degree_id").str.strip_chars().replace_strict(DEGREE_ORD, default=2).cast(pl.Float64)
    n_skills = pl.col("experience").str.split("|").list.len().fill_null(0).cast(pl.Float64)
    n_dcity = (pl.col("desire_jd_city_id").str.count_matches(r"\d+")).fill_null(0).cast(pl.Float64)
    tab = u.select([
        (age / 40.0).alias("age"),
        (tenure / 20.0).alias("tenure"),
        (degree / 5.0).alias("degree"),
        (n_skills.log1p() / 5.0).alias("n_skills"),
        (n_dcity / 3.0).alias("n_dcity"),
    ]).to_numpy().astype(np.float32)

    # --- node tabular: salary preferences, desire counts, career-change flag ---
    ds_id = u["desire_jd_salary_id"].to_list(); cs_id = u["cur_salary_id"].to_list()
    dt_l = u["desire_jd_type_id"].to_list(); ct_l = u["cur_jd_type"].to_list()
    di_l = u["desire_jd_industry_id"].to_list()
    def _split(s):
        return [x.strip() for x in (s.split(",") if s else []) if x.strip() not in ("", "-", "null")]
    extra = np.zeros((len(ds_id), 6), dtype=np.float32)
    for i in range(len(ds_id)):
        dmid = _band_mid(decode_salary_band(ds_id[i]))
        cmid = _band_mid(decode_salary_band(cs_id[i]))
        extra[i, 0] = np.log1p(dmid) / 12.0 if dmid else 0.0        # desired_salary (band midpoint)
        extra[i, 1] = np.log1p(cmid) / 12.0 if cmid else 0.0        # current_salary
        dts = _split(dt_l[i]); dind = _split(di_l[i])
        extra[i, 2] = min(len(dts), 3) / 3.0                        # n_desired_titles
        extra[i, 3] = min(len(dind), 3) / 3.0                       # n_desired_industries
        extra[i, 4] = 1.0 if dind else 0.0                          # has_industry_pref
        cur = _norm(ct_l[i])                                        # is_career_changer:
        extra[i, 5] = 1.0 if (cur and cur not in {_norm(x) for x in dts}) else 0.0  # cur occ not desired
    tab = np.concatenate([tab, extra], axis=1)
    return tab


def _mine_job_skills(descs, keys, skill_by_len, boundary_validate=False):
    """n-gram intersection of job text with the skill vocab (boundary_validate: jieba-aligned)."""
    if not boundary_validate:
        lens = sorted(skill_by_len.keys())
        out = []
        for desc, key in zip(descs, keys):
            found = set()
            text = ""
            if isinstance(desc, str):
                text += desc[:DESC_CAP]
            if isinstance(key, str) and key not in ("null", "\\N", ""):
                text += " " + key
            n = len(text)
            for L in lens:
                S = skill_by_len[L]
                for i in range(0, n - L + 1):
                    sub = text[i:i + L]
                    if sub in S:
                        found.add(sub)
            out.append(sorted(found))
        return out

    import re
    import ahocorasick
    import jieba

    vocab = set().union(*skill_by_len.values())
    A = ahocorasick.Automaton()
    for w in vocab:
        A.add_word(w, w)
    A.make_automaton()
    for w in vocab:
        jieba.add_word(w, freq=10_000_000)
    jieba.initialize()

    latin = re.compile(r"[0-9A-Za-z]+")
    out = []
    for desc, key in zip(descs, keys):
        text = ""
        if isinstance(desc, str):
            text += desc[:DESC_CAP]
        if isinstance(key, str) and key not in ("null", "\\N", ""):
            text += " " + key
        if not text:
            out.append([])
            continue
        # candidate spans: exact substring matches
        spans = [(end - len(w) + 1, end + 1, w) for end, w in A.iter(text)]
        if not spans:
            out.append([])
            continue
        # valid cut positions = jieba boundaries; cuts inside a Latin/digit run removed
        cuts = {0, len(text)}
        for _tok, s, e in jieba.tokenize(text, HMM=False):
            cuts.add(s)
            cuts.add(e)
        for m in latin.finditer(text):
            a, b = m.start(), m.end()
            cuts -= set(range(a + 1, b))
            cuts.add(a)
            cuts.add(b)
        # a term counts if ANY of its occurrences is boundary-aligned
        found = {w for s, e, w in spans if s in cuts and e in cuts}
        out.append(sorted(found))
    return out


def build_job_features(jds, job_map, skill_map, train_job_mask,
                       boundary_validate=False):
    j = jds.filter(pl.col("jd_no").is_in(list(job_map.keys())))
    j = j.with_columns(pl.col("jd_no").replace_strict(job_map).alias("_idx")).sort("_idx")

    def _f(col):
        return pl.col(col).cast(pl.Float64, strict=False)

    min_sal = _f("min_salary").fill_null(0.0)
    max_sal = _f("max_salary").fill_null(0.0)
    req = _f("require_nums").fill_null(1.0)
    minyr = pl.col("min_years").str.strip_chars().replace_strict(MINYEARS_ORD, default=0).cast(pl.Float64)
    edu = pl.col("min_edu_level").str.strip_chars().replace_strict(EDU_ORD, default=2).cast(pl.Float64)
    travel = _f("is_travel").fill_null(0.0)
    tab = j.select([
        (min_sal.log1p() / 12.0).alias("min_sal"),
        (max_sal.log1p() / 12.0).alias("max_sal"),
        (((min_sal + max_sal) / 2.0).log1p() / 12.0).alias("mid_sal"),
        (req.log1p() / 4.0).alias("req"),
        (minyr / 4.0).alias("minyr"),
        (edu / 5.0).alias("edu"),
        (travel / 2.0).alias("travel"),
    ]).to_numpy().astype(np.float32)

    # requires_title from jd_sub_type; requires_skill from mined text
    subtypes = j["jd_sub_type"].to_list()
    skill_by_len = {}
    for s in skill_map:
        skill_by_len.setdefault(len(s), set()).add(s)
    t0 = time.time()
    mined = _mine_job_skills(j["job_description"].to_list(), j["key"].to_list(), skill_by_len,
                             boundary_validate=boundary_validate)
    raw_edges = sum(len(m) for m in mined)
    # denoise + cap: drop near-ubiquitous skills, keep rarest-per-job; df fit on TRAIN jobs only
    df = {}
    assert len(mined) == len(train_job_mask), "mined rows must align 1:1 with job idx"
    for m, is_tr in zip(mined, train_job_mask):
        if not is_tr:
            continue
        for s in m:
            df[s] = df.get(s, 0) + 1
    n_jobs = len(mined)
    n_train_jobs = int(train_job_mask.sum())
    max_df = SKILL_MAX_DF_FRAC * n_train_jobs
    generic = {s for s, c in df.items() if c > max_df}
    mined2 = []
    for m in mined:
        keep = sorted((s for s in m if s not in generic), key=lambda s: df.get(s, 0))[:SKILL_PER_JOB_CAP]
        mined2.append(sorted(keep))
    mined = mined2
    miner = "boundary-validated (jieba)" if boundary_validate else "substring"
    print(f"  mined job skills [{miner}] in {time.time()-t0:.1f}s  (jobs={n_jobs}, df fit on "
          f"{n_train_jobs} train-engaged, raw_edges={raw_edges} -> "
          f"capped={sum(len(m) for m in mined)}, dropped {len(generic)} generic terms)")

    return tab, subtypes, mined


def build_edges(users, jds, act, maps):
    seeker_map, job_map = maps["seeker"], maps["job"]
    skill_map, title_map = maps["skill"], maps["title"]
    edges = {}

    # ---- interaction edges (restricted to jobs in the job universe) ----
    a = act.filter(pl.col("jd_no").is_in(list(job_map.keys())))
    a = a.with_columns([
        pl.col("user_id").replace_strict(seeker_map).alias("m"),
        pl.col("jd_no").replace_strict(job_map).alias("j"),
    ])
    br = a.filter(pl.col("browsed") == 1)
    de = a.filter(pl.col("delivered") == 1)
    edges[("seeker", "considers", "job")] = np.stack([br["m"].to_numpy(), br["j"].to_numpy()])
    edges[("seeker", "accepts", "job")] = np.stack([de["m"].to_numpy(), de["j"].to_numpy()])

    # ---- has_skill / has_title (seeker side) ----
    u = users.filter(pl.col("user_id").is_in(list(seeker_map.keys())))
    hs_m, hs_s, ht_m, ht_t = [], [], [], []
    for uid, exp, ct in zip(u["user_id"].to_list(),
                            u.select(pl.col("experience").str.split("|")).to_series().to_list(),
                            u["cur_jd_type"].to_list()):
        mi = seeker_map[uid]
        for t in set(_norm(x) for x in (exp or [])):
            if t in skill_map:
                hs_m.append(mi); hs_s.append(skill_map[t])
        tt = {_norm(ct)}   # seeker title = CURRENT occupation only; desired titles dropped
        for t in tt:
            if t in title_map:
                ht_m.append(mi); ht_t.append(title_map[t])
    edges[("seeker", "has_skill", "skill")] = np.array([hs_m, hs_s], dtype=np.int64)
    edges[("seeker", "has_title", "title")] = np.array([ht_m, ht_t], dtype=np.int64)
    return edges, a


def build_job_attr_edges(job_map, title_map, skill_map, subtypes, mined):
    rt_j, rt_t, rs_j, rs_s = [], [], [], []
    # subtypes/mined are ordered by job idx (built from the sorted job frame)
    for ji, (st, msk) in enumerate(zip(subtypes, mined)):
        st = _norm(st)
        if st in title_map:
            rt_j.append(ji); rt_t.append(title_map[st])
        for s in msk:
            if s in skill_map:
                rs_j.append(ji); rs_s.append(skill_map[s])
    out = {}
    out[("job", "requires_title", "title")] = np.array([rt_j, rt_t], dtype=np.int64)
    out[("job", "requires_skill", "skill")] = np.array([rs_j, rs_s], dtype=np.int64)
    return out


def build_pairs(a, seeker_map, split_map):
    """Supervised pair frame + job exposure count (train-interaction count)."""
    df = a.select(["m", "j", "browsed", "delivered", "satisfied"]).rename(
        {"m": "seeker_idx", "j": "job_idx"})
    # seeker split
    inv = {v: k for k, v in seeker_map.items()}
    split_series = pl.Series("split", [split_map[inv[i]] for i in df["seeker_idx"].to_list()])
    df = df.with_columns(split_series)
    # job exposure count = # of TRAINING interactions (browsed OR delivered) on the job
    train_int = df.filter((pl.col("split") == "train") &
                          ((pl.col("browsed") == 1) | (pl.col("delivered") == 1)))
    pop = train_int.group_by("job_idx").len().rename({"len": "job_exposure"})
    df = df.join(pop, on="job_idx", how="left").with_columns(pl.col("job_exposure").fill_null(0))
    return df


def build_match_feats(users, jds, maps, mined, pairs_df):
    """Per-(seeker,job) MATCH pair features -> pairs_df + TIANCHI_MATCH_FEATURES columns (leak-free)."""
    seeker_map, job_map, skill_map = maps["seeker"], maps["job"], maps["skill"]
    nM, nJ = len(seeker_map), len(job_map)

    # --- seeker attribute tables (indexed by seeker idx) ---
    u = (users.filter(pl.col("user_id").is_in(list(seeker_map.keys())))
              .with_columns(pl.col("user_id").replace_strict(seeker_map).alias("_idx")).sort("_idx"))
    m_dcity = [frozenset()] * nM; m_live = [""] * nM
    m_dband = [None] * nM; m_cband = [None] * nM
    m_deg = np.full(nM, 2.0); m_tenure = np.full(nM, 3.0)
    m_titles = [frozenset()] * nM; m_skills = [frozenset()] * nM
    for r in u.iter_rows(named=True):
        i = r["_idx"]
        dc = r["desire_jd_city_id"]
        m_dcity[i] = frozenset(c.strip() for c in (dc.split(",") if dc else [])
                               if c.strip() not in ("", "-", "null"))
        m_live[i] = (r["live_city_id"] or "").strip()
        m_dband[i] = decode_salary_band(r["desire_jd_salary_id"])
        m_cband[i] = decode_salary_band(r["cur_salary_id"])
        m_deg[i] = DEGREE_ORD.get((r["cur_degree_id"] or "").strip(), 2)
        try:
            m_tenure[i] = min(max(2019 - int(r["start_work_date"]), 0), 45)
        except (TypeError, ValueError):
            pass
        dts = r["desire_jd_type_id"]
        titles = set(_norm(t) for t in (dts.split(",") if dts else []))
        titles.add(_norm(r["cur_jd_type"]))
        m_titles[i] = frozenset(t for t in titles if t)
        exp = r["experience"]
        m_skills[i] = frozenset(t for t in (_norm(x) for x in (exp.split("|") if exp else []))
                                if t in skill_map)

    # --- job attribute tables (indexed by job idx; mined already in this order) ---
    j = (jds.filter(pl.col("jd_no").is_in(list(job_map.keys())))
            .with_columns(pl.col("jd_no").replace_strict(job_map).alias("_idx")).sort("_idx"))
    j_city = [""] * nJ; j_min = np.zeros(nJ); j_max = np.zeros(nJ)
    j_edu = np.full(nJ, 2.0); j_yrs = np.zeros(nJ); j_sub = [""] * nJ
    def _num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return 0.0
    for r in j.iter_rows(named=True):
        i = r["_idx"]
        j_city[i] = (r["city"] or "").strip()
        j_min[i] = _num(r["min_salary"]); j_max[i] = _num(r["max_salary"])
        j_edu[i] = EDU_ORD.get((r["min_edu_level"] or "").strip(), 2)
        j_yrs[i] = MINYEARS_LOWER.get((r["min_years"] or "").strip(), 0)
        j_sub[i] = _norm(r["jd_sub_type"])
    j_skills = [frozenset(mined[i]) for i in range(nJ)]

    # --- per-pair features (order = pairs_df rows) ---
    mi = pairs_df["seeker_idx"].to_numpy(); ji = pairs_df["job_idx"].to_numpy()
    n = len(mi); F = np.zeros((n, TIANCHI_MATCH_DIM), dtype=np.float32)
    for row in range(n):
        m = int(mi[row]); jj = int(ji[row])
        jc = j_city[jj]; jmin = j_min[jj]; jmax = j_max[jj]
        F[row, 0] = 1.0 if jc and jc in m_dcity[m] else 0.0        # desire_city_match
        F[row, 1] = 1.0 if jc and jc == m_live[m] else 0.0         # live_city_match
        job_unspec = (jmin == 0 and jmax == 0)
        jmid = (jmin + jmax) / 2.0 if jmax > 0 else jmin
        db = m_dband[m]
        if db is None or job_unspec:
            F[row, 3] = 1.0                                        # salary_missing
        else:
            dlo, dhi = db; dhi_v = dhi if dhi is not None else 1e9
            jhi = jmax if jmax > 0 else 1e9
            F[row, 2] = 1.0 if (dlo <= jhi and jmin <= dhi_v) else 0.0   # salary_in_range
            dmid = _band_mid(db)
            if dmid and jmid:
                F[row, 4] = np.log1p(dmid) - np.log1p(jmid)        # salary_gap
        cb = m_cband[m]
        if cb is None or job_unspec:
            F[row, 6] = 1.0                                        # cur_salary_missing
        else:
            cmid = _band_mid(cb)
            if cmid and jmid:
                F[row, 5] = np.log1p(jmid) - np.log1p(cmid)        # cur_salary_gap (raise?)
        F[row, 7] = 1.0 if j_sub[jj] and j_sub[jj] in m_titles[m] else 0.0   # title_match
        ms = m_skills[m]; js = j_skills[jj]
        if ms and js:
            inter = len(ms & js)
            F[row, 8] = inter / max(len(ms | js), 1)              # skill_jaccard
            F[row, 9] = np.log1p(inter)                           # skill_overlap_count
        F[row, 10] = 1.0 if m_deg[m] >= j_edu[jj] else 0.0        # edu_meets
        F[row, 11] = (m_deg[m] - j_edu[jj]) / 5.0                 # edu_gap
        F[row, 12] = 1.0 if m_tenure[m] >= j_yrs[jj] else 0.0     # exp_meets
        F[row, 13] = (m_tenure[m] - j_yrs[jj]) / 10.0             # exp_gap

    for k, name in enumerate(TIANCHI_MATCH_FEATURES):
        pairs_df = pairs_df.with_columns(pl.Series(name, F[:, k]))
    nz = (F != 0).any(axis=0).sum()
    print(f"  match feats: {n} pairs x {TIANCHI_MATCH_DIM} dims ({int(nz)} non-degenerate cols)")
    return pairs_df


def make_split(seeker_keys, seed=SPLIT_SEED):
    rng = np.random.default_rng(seed)
    keys = list(seeker_keys)
    perm = rng.permutation(len(keys))
    n = len(keys)
    n_test = int(n * TEST_FRAC)
    n_val = int(n * VAL_FRAC)
    split_map = {}
    for rank, idx in enumerate(perm):
        if rank < n_test:
            split_map[keys[idx]] = "test"
        elif rank < n_test + n_val:
            split_map[keys[idx]] = "val"
        else:
            split_map[keys[idx]] = "train"
    return split_map


def main(boundary_validate: bool = False, out_dir=None):
    out = Path(out_dir) if out_dir is not None else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"miner={'boundary-validated (jieba)' if boundary_validate else 'substring'}  out={out}")
    print("loading tables ...")
    users, jds, act = load_tables()
    # split FIRST so the vocab can be fit on train seekers only
    print("splitting seekers (before vocab fit) ...")
    split_map = make_split(sorted(set(act["user_id"].unique().to_list())))
    print("building node maps (vocab fit on TRAIN split only) ...")
    maps, train_job_mask = build_node_maps(users, jds, act, split_map)
    for t, m in maps.items():
        print(f"  {t}: {len(m)}")

    print("building seeker features ...")
    xm = build_seeker_features(users, maps["seeker"])
    print("building job features + mining skills ...")
    xj, subtypes, mined = build_job_features(jds, maps["job"], maps["skill"],
                                             train_job_mask, boundary_validate=boundary_validate)
    n_skill, n_title = len(maps["skill"]), len(maps["title"])
    xs = np.eye(n_skill, dtype=np.float32)   # identity node features for attribute nodes
    xt = np.eye(n_title, dtype=np.float32)
    print(f"  seeker x {xm.shape} | job x {xj.shape} | skill x {xs.shape} | title x {xt.shape}")

    print("building edges ...")
    edges, a = build_edges(users, jds, act, maps)
    edges.update(build_job_attr_edges(maps["job"], maps["title"], maps["skill"], subtypes, mined))
    for rel, ei in edges.items():
        print(f"  {rel}: {ei.shape[1]} edges")

    print("building pairs ...")
    pairs = build_pairs(a, maps["seeker"], split_map)
    print("building match features ...")
    pairs = build_match_feats(users, jds, maps, mined, pairs)

    # ---- save ----
    with open(out / "node_maps.json", "w") as f:
        json.dump(maps, f)
    graph = {
        "num_nodes": {"seeker": len(maps["seeker"]), "job": len(maps["job"]),
                      "skill": n_skill, "title": n_title},
        "x": {"seeker": torch.from_numpy(xm), "job": torch.from_numpy(xj),
              "skill": torch.from_numpy(xs), "title": torch.from_numpy(xt)},
        "edges": {"__".join(rel): torch.from_numpy(ei.astype(np.int64)) for rel, ei in edges.items()},
    }
    torch.save(graph, out / "graph.pt")
    pairs.write_parquet(out / "pairs.parquet")

    # split / label summary
    print("\n=== SPLIT SUMMARY ===")
    print(pairs.group_by("split").agg([
        pl.len().alias("pairs"),
        pl.col("seeker_idx").n_unique().alias("seekers"),
        pl.col("job_idx").n_unique().alias("jobs"),
        pl.col("satisfied").sum().alias("pos"),
        pl.col("delivered").sum().alias("delivered"),
        (pl.col("satisfied").sum() / pl.col("delivered").sum()).alias("pass|deliv"),
    ]).sort("split"))
    print(f"\ndone in {time.time()-t0:.1f}s -> {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build the Tianchi PJF artifact (graph.pt / pairs.parquet / node_maps.json)")
    ap.add_argument("--boundary-validate", action="store_true",
                    help="validate job-skill substring matches against jieba word boundaries "
                         "(kills boundary-crossing junk; requires jieba + pyahocorasick)")
    ap.add_argument("--out", default=None,
                    help="output dir (default: data/tianchi_prepared; boundary-validated builds "
                         "default to data/tianchi_prepared_bv so the canonical artifact is never overwritten)")
    args = ap.parse_args()
    _out = args.out
    if args.boundary_validate and _out is None:
        _out = DATA_DIR / "tianchi_prepared_bv"
    main(boundary_validate=args.boundary_validate, out_dir=_out)
