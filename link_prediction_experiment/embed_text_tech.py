"""Generate Tech seeker & job text embeddings with a frozen LLM encoder (default Qwen3-Embedding-0.6B).

Tech counterpart of tianchi_pjf/embed_text.py. The Tech dataset is proprietary and NOT shipped
(bring your own masters); the public, fully-reproducible counterpart is tianchi_pjf/embed_text.py.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import time

import numpy as np
import polars as pl
import torch

from . import paths
from .data import load_node_maps, USER_COL, JOB_COL

OUT_DIR = paths.TECH_PREPARED_DIR
MODEL = "Qwen/Qwen3-Embedding-0.6B"
SEEKER_TASK = ("Given a job seeker's profile, retrieve job postings that fit the seeker")

SEEKER_TEXT_COL = "profile_text"
JOB_TEXT_COLS = ["job_title", "job_description", "job_feature"]


def _ordered_ids(node_map):
    """Node keys in NODE-INDEX order + whether row-i==index-i (contiguous 0..n-1)."""
    items = sorted(node_map.items(), key=lambda kv: kv[1])
    ids = [k for k, _ in items]
    aligned = [v for _, v in items] == list(range(len(items)))
    return ids, aligned


def _model_tag(model_name: str) -> str:
    """Short output-filename prefix per model family (default qwen3; e5 -> me5)."""
    n = model_name.lower()
    if "qwen" in n:  return "qwen3"
    if "e5" in n:    return "me5"
    base = model_name.rstrip("/").split("/")[-1].lower()
    return re.sub(r"[^a-z0-9]+", "_", base).strip("_") or "emb"


def _prompts(model_name: str, symmetric: bool):
    """(seeker_prompt, job_prompt, seeker_instruct, job_instruct); --symmetric drops all prompts."""
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
    return v.strip() if isinstance(v, str) and v.strip() else ""


def _latest(parquet, id_col, keep_ids, text_cols):
    """Latest row per entity, restricted to keep_ids (string ids) -> {id:{col:text}}."""
    df = (pl.scan_parquet(parquet)
          .select([id_col, "snapshot_date"] + text_cols)
          .with_columns(pl.col(id_col).cast(pl.Utf8))
          .filter(pl.col(id_col).is_in(keep_ids))
          .collect()
          .sort("snapshot_date", descending=True)
          .unique(subset=[id_col], keep="first"))
    out = {}
    cols = {c: df[c].to_list() for c in [id_col] + text_cols}
    for i, rid in enumerate(cols[id_col]):
        out[str(rid)] = {c: cols[c][i] for c in text_cols}
    return out


def _seeker_texts(ids):                                # ids = "m_<seeker_id>" node-idx order
    raw = [k[2:] for k in ids]
    rows = _latest(paths.SEEKER_PARQUET, USER_COL, raw, [SEEKER_TEXT_COL])
    texts = [_clean(rows.get(i, {}).get(SEEKER_TEXT_COL)) for i in raw]
    del rows; gc.collect()
    return texts


def _job_texts(ids):                                   # ids = "j_<job_id>"
    raw = [k[2:] for k in ids]
    rows = _latest(paths.JOB_PARQUET, JOB_COL, raw, JOB_TEXT_COLS)
    texts = []
    for i in raw:
        r = rows.get(i, {})
        segs = [_clean(r.get(c)) for c in JOB_TEXT_COLS]     # title, description, feature
        texts.append("\n".join(s for s in segs if s))
    del rows; gc.collect()
    return texts


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
    win = max(8, int(model.max_seq_length) - 16)
    win_texts, bounds, wlen = [], [], []
    cur = 0
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"] if t else []
        if not ids:
            ids = [tok.unk_token_id or 0]
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
    for di, (a, b) in enumerate(bounds):
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
    emb[empty] = 0.0
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
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true",
                    help="bfloat16 compute (cuda) -- native precision for Qwen3-Embedding-4B/8B; "
                         "output still stored float32 so downstream graph loading is unchanged.")
    ap.add_argument("--symmetric", action="store_true")
    ap.add_argument("--tag", default=None, help="output filename prefix (default from model)")
    ap.add_argument("--chunk-pool", action="store_true",
                    help="full text, NO truncation (use for e5 whose 512 limit is hard)")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build texts + stats, skip model")
    args = ap.parse_args()

    TAG = args.tag or _model_tag(args.model)
    seeker_prompt, job_prompt, s_instruct, j_instruct = _prompts(args.model, args.symmetric)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("loading node_maps + master text ...")
    nm = load_node_maps()
    s_ids, na_s = _ordered_ids(nm.seeker)
    j_ids, na_j = _ordered_ids(nm.job)
    s_txt = _seeker_texts(s_ids)
    j_txt = _job_texts(j_ids)
    s_empty = np.array([t == "" for t in s_txt])
    j_empty = np.array([t == "" for t in j_txt])
    print(f"seekers: {len(s_ids)} ids, {int((~s_empty).sum())} non-empty {SEEKER_TEXT_COL} "
          f"(avg {np.mean([len(t) for t in s_txt]):.0f} chars)")
    print(f"jobs   : {len(j_ids)} ids, {int((~j_empty).sum())} non-empty "
          f"(avg {np.mean([len(t) for t in j_txt]):.0f} chars)")

    if args.dry_run:
        _write_jsonl(OUT_DIR / f"{TAG}_seeker_emb.texts.jsonl", s_ids, s_txt)
        _write_jsonl(OUT_DIR / f"{TAG}_job_emb.texts.jsonl", j_ids, j_txt)
        print(f"[dry-run] wrote raw text -> {TAG}_{{seeker,job}}_emb.texts.jsonl")
        print("\n[dry-run] sample seeker text:\n" + (s_txt[0][:400] if s_txt else ""))
        print("\n[dry-run] sample job text:\n" + (j_txt[0][:300] if j_txt else ""))
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
    model.max_seq_length = min(args.max_seq_len, 512) if "e5" in args.model.lower() else args.max_seq_len
    print(f"  model ready in {time.time()-t0:.1f}s (max_seq_len={model.max_seq_length}); encoding ...")

    def _do(name, field, instr, ids, txt, empty, prompt, na):
        _write_jsonl(OUT_DIR / (name[:-3] + ".texts.jsonl"), ids, txt)
        t = time.time()
        enc = _encode_chunkpool if args.chunk_pool else _encode
        emb = enc(model, txt, prompt, args.batch_size)
        txt.clear(); gc.collect()
        _save_side(name, field, instr, ids, emb, empty, na, args.model,
                   model.max_seq_length, "chunkpool_meanpool" if args.chunk_pool else "single")
        del emb; gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"    ({time.time()-t:.1f}s)")

    print(f"[seeker] encoding {SEEKER_TEXT_COL} ...")
    _do(f"{TAG}_seeker_emb.pt", SEEKER_TEXT_COL, s_instruct, s_ids, s_txt, s_empty, seeker_prompt, na_s)
    print("[job] encoding job_title+job_description+job_feature ...")
    _do(f"{TAG}_job_emb.pt", "job_title+job_description+job_feature", j_instruct,
        j_ids, j_txt, j_empty, job_prompt, na_j)
    print(f"\ndone -> {OUT_DIR}")


if __name__ == "__main__":
    main()
