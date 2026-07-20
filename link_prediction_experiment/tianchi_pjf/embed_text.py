"""Generate Tianchi seeker & job text embeddings with a frozen LLM encoder (default Qwen3-Embedding-0.6B).

Both sides embed to one shared space (seeker=query, job=document); L2-normalised, saved node-aligned
to data/tianchi_prepared/. Usage: python -m link_prediction_experiment.tianchi_pjf.embed_text
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch


def _ordered_ids(node_map):
    """Node keys in NODE-INDEX order + whether row-i == index-i (contiguous 0..n-1)."""
    items = sorted(node_map.items(), key=lambda kv: kv[1])
    ids = [k for k, _ in items]
    aligned = [v for _, v in items] == list(range(len(items)))
    return ids, aligned


PKG_DIR = Path(__file__).resolve().parent
TIANCHI = PKG_DIR.parent / "data" / "tianchi"
OUT_DIR = PKG_DIR.parent / "data" / "tianchi_prepared"

MODEL = "Qwen/Qwen3-Embedding-0.6B"
# PJF retrieval instruction for the seeker (query) side; jobs get no instruction.
SEEKER_TASK = ("Given a job seeker's work experience and skills, retrieve job "
               "descriptions of positions the seeker is a good fit for")


def _model_tag(model_name: str) -> str:
    """Short output-filename prefix per model family (default qwen3; e5 -> me5)."""
    n = model_name.lower()
    if "qwen" in n:    return "qwen3"
    if "e5" in n:      return "me5"
    base = model_name.rstrip("/").split("/")[-1].lower()
    return re.sub(r"[^a-z0-9]+", "_", base).strip("_") or "emb"


def _prompts(model_name: str, symmetric: bool):
    """(seeker_prompt, job_prompt, seeker_instruct, job_instruct) per model family; --symmetric drops all."""
    if symmetric:
        return None, None, None, None
    if "e5" in model_name.lower():
        return "query: ", "passage: ", "query:", "passage:"
    return f"Instruct: {SEEKER_TASK}\nQuery:", None, SEEKER_TASK, None


def _pick_device(spec: str) -> str:
    if spec != "auto":
        return spec
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _clean(v) -> str:
    return (v.strip() if isinstance(v, str) and v.strip()
            and v not in ("-", "null", "\\N") else "")


def _salary(code) -> str:
    """Decode a Tianchi salary code to a Chinese ¥/month range ('面议' negotiable, '' dirty)."""
    s = _clean(code)
    if not s or not s.isdigit():
        return ""
    if len(s) in (9, 11):
        s = "0" + s
    if len(s) not in (10, 12):
        return ""
    h = len(s) // 2
    lo, hi = int(s[:h]), int(s[h:])
    if lo == 0 and hi == 0:
        return "面议"
    if hi < lo or hi == 0:
        return ""                      # dirty (e.g. hi below lo)
    return f"{lo}-{hi}元/月"


SEEKER_COLS = ["desire_jd_type_id", "desire_jd_industry_id", "desire_jd_salary_id",
               "cur_jd_type", "cur_industry_id", "cur_salary_id", "cur_degree_id",
               "start_work_date", "experience"]


def _seeker_texts(users: pl.DataFrame, ids: list[str]) -> list[str]:
    """Profile passage: desired/current job-type & industry, salary, education, start year, experience."""
    d = {c: dict(zip(users["user_id"].to_list(), users[c].to_list())) for c in SEEKER_COLS}
    out = []
    for k in ids:
        parts = []
        seg = []
        if (v := _clean(d["desire_jd_type_id"].get(k))):  seg.append(f"求职意向：{v}")
        if (v := _clean(d["desire_jd_industry_id"].get(k))): seg.append(f"意向行业：{v}")
        if (v := _salary(d["desire_jd_salary_id"].get(k))): seg.append(f"期望薪资：{v}")
        if seg: parts.append("；".join(seg))
        seg = []
        if (v := _clean(d["cur_jd_type"].get(k))):     seg.append(f"当前职位：{v}")
        if (v := _clean(d["cur_industry_id"].get(k))): seg.append(f"当前行业：{v}")
        if (v := _salary(d["cur_salary_id"].get(k))):  seg.append(f"当前薪资：{v}")
        if seg: parts.append("；".join(seg))
        seg = []
        if (v := _clean(d["cur_degree_id"].get(k))):    seg.append(f"学历：{v}")
        if (v := _clean(d["start_work_date"].get(k))):  seg.append(f"参加工作年份：{v}")
        if seg: parts.append("；".join(seg))
        if (v := _clean(d["experience"].get(k))):
            terms = "、".join(t.strip() for t in v.split("|") if t.strip())
            if terms: parts.append(f"技能与经历：{terms}")
        out.append("\n".join(parts))
    return out


def _job_texts(jds: pl.DataFrame, ids: list[str]) -> list[str]:
    """jd_title + job_description (title first, then free text); empty only if both missing."""
    title = dict(zip(jds["jd_no"].to_list(), jds["jd_title"].to_list()))
    desc = dict(zip(jds["jd_no"].to_list(), jds["job_description"].to_list()))
    out = []
    for k in ids:
        t, d = _clean(title.get(k)), _clean(desc.get(k))
        out.append(f"{t}\n{d}".strip() if (t and d) else (t or d))
    return out


def _encode(model, texts, prompt, bs, chunk=3000):
    """Chunked encode with accelerator-cache release between chunks -> bounded peak memory."""
    kw = dict(batch_size=bs, normalize_embeddings=True, show_progress_bar=False,
              convert_to_numpy=True)
    if prompt is not None:
        kw["prompt"] = prompt
    dim = model.get_sentence_embedding_dimension()
    if not texts:
        return np.zeros((0, dim), np.float32)
    parts = []
    for i in range(0, len(texts), chunk):
        parts.append(model.encode(texts[i:i + chunk], **kw))
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"      {min(i + chunk, len(texts))}/{len(texts)}", flush=True)
    return np.concatenate(parts, axis=0)


def _encode_chunkpool(model, texts, prompt, bs, chunk=4000):
    """Full-text encode, NO truncation: window each doc, embed, token-weighted mean-pool + renormalise."""
    tok = model.tokenizer
    dim = model.get_sentence_embedding_dimension()
    if not texts:
        return np.zeros((0, dim), np.float32)
    win = max(8, int(model.max_seq_length) - 16)          # leave room for specials + prompt
    win_texts, bounds, wlen = [], [], []                  # windows, per-doc [start,end), token counts
    cur = 0
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"] if t else []
        if not ids:
            ids = [tok.unk_token_id or 0]                 # empty -> 1 dummy window (zeroed later via `empty`)
        a = cur
        for s in range(0, len(ids), win):
            piece = ids[s:s + win]
            win_texts.append(tok.decode(piece, skip_special_tokens=True))
            wlen.append(len(piece)); cur += 1
        bounds.append((a, cur))
    kw = dict(batch_size=bs, normalize_embeddings=True, show_progress_bar=False,
              convert_to_numpy=True)
    if prompt is not None:
        kw["prompt"] = prompt
    parts = []
    for i in range(0, len(win_texts), chunk):
        parts.append(model.encode(win_texts[i:i + chunk], **kw))
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"      {min(i + chunk, len(win_texts))}/{len(win_texts)} windows", flush=True)
    W = np.concatenate(parts, axis=0)
    wlen = np.asarray(wlen, np.float32)
    out = np.zeros((len(texts), dim), np.float32)
    for di, (a, b) in enumerate(bounds):                  # token-weighted mean-pool per doc
        v = (W[a:b] * wlen[a:b, None]).sum(0) / max(float(wlen[a:b].sum()), 1e-9)
        nrm = np.linalg.norm(v)
        out[di] = v / nrm if nrm > 0 else v
    return out


def _write_jsonl(path, ids, texts):
    with open(path, "w") as f:
        for k, t in zip(ids, texts):
            f.write(json.dumps({"id": k, "text": t}, ensure_ascii=False) + "\n")


def _save_side(name, field, instruct, ids, emb, empty, node_aligned, model,
               max_seq_len, pooling):
    """Single non-redundant artifact: emb in node-index order (empties zeroed), key2row for id lookup."""
    emb = np.ascontiguousarray(emb)
    emb[empty] = 0.0                                    # empty text -> zero vector
    obj = {"model": model, "dim": int(emb.shape[1]), "field": field,
           "instruct": instruct, "ids": list(ids),
           "key2row": {k: i for i, k in enumerate(ids)},
           "emb": torch.from_numpy(emb).float(),
           "empty": torch.from_numpy(empty), "node_aligned": bool(node_aligned),
           "max_seq_len": int(max_seq_len), "pooling": pooling}
    torch.save(obj, OUT_DIR / name)
    tag = "node-aligned" if node_aligned else "id-keyed only"
    print(f"  saved {name}: emb {tuple(obj['emb'].shape)} ({tag}), "
          f"{int((~empty).sum())}/{len(ids)} non-empty")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL,
                    help="HF encoder id (default Qwen3-Embedding-0.6B; the paper also compares "
                         "Qwen3-Embedding-4B/-8B and intfloat/multilingual-e5-large)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="small default: seq^2 attention memory on MPS is the OOM risk")
    ap.add_argument("--max-seq-len", type=int, default=1024,
                    help="truncate texts to this many tokens (speed/memory vs coverage)")
    ap.add_argument("--fp16", action="store_true", help="half precision (cuda/mps)")
    ap.add_argument("--bf16", action="store_true",
                    help="bfloat16 compute (cuda) -- the NATIVE precision for the larger "
                         "Qwen3-Embedding-4B/8B (fp16 can overflow; fp32 8B needs 32GB). "
                         "Compute-only: the pooled output is still stored float32 (ST upcasts + "
                         "_save_side .float()), so downstream graph loading is unchanged.")
    ap.add_argument("--symmetric", action="store_true",
                    help="no instruction on either side (both encoded as plain text)")
    ap.add_argument("--tag", default=None,
                    help="output filename prefix (default: auto from model, e.g. qwen3/me5)")
    ap.add_argument("--chunk-pool", action="store_true",
                    help="full text, NO truncation: split into <=max_seq_len windows, embed "
                         "each, token-weighted mean-pool (use for e5 whose 512 limit is hard)")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug/smoke: cap #ids per side (breaks node alignment)")
    ap.add_argument("--dry-run", action="store_true", help="build texts + stats, skip model")
    args = ap.parse_args()

    TAG = args.tag or _model_tag(args.model)
    seeker_prompt, job_prompt, s_instruct, j_instruct = _prompts(args.model, args.symmetric)
    print("loading tables ...")
    users = pl.read_csv(TIANCHI / "table1_user.txt", separator="\t", quote_char=None,
                        infer_schema_length=0)
    jds = pl.read_csv(TIANCHI / "table2_jd.txt", separator="\t", quote_char=None,
                      infer_schema_length=0, truncate_ragged_lines=True)

    nmp = OUT_DIR / "node_maps.json"
    node_maps = json.load(open(nmp)) if nmp.exists() else None
    if node_maps is None:
        raise SystemExit(f"missing {nmp}: run `python -m link_prediction_experiment."
                         f"tianchi_pjf.build` first to fit the node universe.")

    # node-index order -> emb doubles as the aligned node-feature matrix.
    seeker_ids, na_s = _ordered_ids(node_maps["seeker"])
    job_ids, na_j = _ordered_ids(node_maps["job"])
    if args.limit:
        seeker_ids, job_ids = seeker_ids[:args.limit], job_ids[:args.limit]
        na_s = na_j = False

    seeker_txt = _seeker_texts(users, seeker_ids)
    job_txt = _job_texts(jds, job_ids)
    del users, jds; gc.collect()      # free source frames before the model loads
    s_empty = np.array([t == "" for t in seeker_txt])
    j_empty = np.array([t == "" for t in job_txt])
    print(f"seekers: {len(seeker_ids)} ids, {int((~s_empty).sum())} non-empty "
          f"experience (avg {np.mean([len(t) for t in seeker_txt]):.0f} chars)")
    print(f"jobs   : {len(job_ids)} ids, {int((~j_empty).sum())} non-empty "
          f"job_description (avg {np.mean([len(t) for t in job_txt]):.0f} chars)")

    OUT_DIR.mkdir(exist_ok=True)
    if args.dry_run:
        _write_jsonl(OUT_DIR / f"{TAG}_seeker_emb.texts.jsonl", seeker_ids, seeker_txt)
        _write_jsonl(OUT_DIR / f"{TAG}_job_emb.texts.jsonl", job_ids, job_txt)
        print(f"[dry-run] wrote raw text -> {TAG}_{{seeker,job}}_emb.texts.jsonl")
        print("\n[dry-run] sample seeker text:\n" + (seeker_txt[0][:400] if seeker_txt else ""))
        print("\n[dry-run] sample job text:\n" + (job_txt[0][:200] if job_txt else ""))
        print("[dry-run] done (model not loaded)")
        return

    from sentence_transformers import SentenceTransformer
    device = _pick_device(args.device)
    if args.fp16 and args.bf16:
        raise SystemExit("--fp16 and --bf16 are mutually exclusive")
    if args.bf16 and device == "cuda":
        dtype = torch.bfloat16
    elif args.fp16 and device in ("cuda", "mps"):
        dtype = torch.float16
    else:
        dtype = torch.float32
    print(f"\nloading {args.model} on {device} ({dtype}) as tag '{TAG}' ...")
    t0 = time.time()
    model = SentenceTransformer(args.model, device=device,
                                trust_remote_code=args.trust_remote_code,
                                model_kwargs={"torch_dtype": dtype})
    # e5 is XLM-R based (512 positional limit) -> longer would index-error on long texts.
    model.max_seq_length = min(args.max_seq_len, 512) if "e5" in args.model.lower() else args.max_seq_len
    print(f"  model ready in {time.time()-t0:.1f}s (max_seq_len={model.max_seq_length}); encoding ...")

    # encode + save + free one side at a time to bound peak RAM
    def _do(name, field, instr, ids, txt, empty, prompt, na):
        _write_jsonl(OUT_DIR / (name[:-3] + ".texts.jsonl"), ids, txt)
        t = time.time()
        enc = _encode_chunkpool if args.chunk_pool else _encode
        emb = enc(model, txt, prompt, args.batch_size)
        txt.clear(); gc.collect()                       # drop text list before save
        _save_side(name, field, instr, ids, emb, empty, na, args.model,
                   model.max_seq_length, "chunkpool_meanpool" if args.chunk_pool else "single")
        del emb; gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()                     # release accelerator cache between sides
        print(f"    ({time.time()-t:.1f}s)")

    print("[seeker] encoding ...")
    _do(f"{TAG}_seeker_emb.pt", "profile(type+industry+salary+degree+start+experience)",
        s_instruct, seeker_ids, seeker_txt, s_empty, seeker_prompt, na_s)
    print("[job] encoding ...")
    _do(f"{TAG}_job_emb.pt", "jd_title+job_description", j_instruct,
        job_ids, job_txt, j_empty, job_prompt, na_j)
    print(f"\ndone -> {OUT_DIR}")


if __name__ == "__main__":
    main()
