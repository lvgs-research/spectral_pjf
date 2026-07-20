"""Filesystem locations for the Tech link-prediction experiment.

Inputs live in the package's data/ directory; outputs are created at import time.
"""
from __future__ import annotations

from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent

# --- input artifacts (all inside the package's data/ directory) ---
DATA_DIR = PKG_DIR / "data"
TECH_DATA_DIR = DATA_DIR / "tech"
TARGET_PARQUET = TECH_DATA_DIR / "interactions.parquet"
SEEKER_PARQUET = TECH_DATA_DIR / "seekers.parquet"
JOB_PARQUET = TECH_DATA_DIR / "jobs.parquet"
SKILLS_VOCAB_PARQUET = TECH_DATA_DIR / "skills_vocab.parquet"

# --- prebuilt heterogeneous marketplace graph (timestamped, PIT-ready) ---
GRAPH_DIR = DATA_DIR / "tech_graph"
HETERO_GRAPH_PT = GRAPH_DIR / "hetero_graph.pt"
NODE_MAPS_JSON = GRAPH_DIR / "node_maps.json"

# Node-aligned frozen text-content embeddings (seeker+job), leak-safe (no corpus fit)
TECH_PREPARED_DIR = DATA_DIR / "tech_prepared"


def tech_emb_file(tag: str, side: str):
    """Path to a node-aligned frozen-encoder content file."""
    return TECH_PREPARED_DIR / f"{tag}_{side}_emb.pt"


# --- experiment outputs / cache ---
CACHE_DIR = PKG_DIR / "_cache"
RESULTS_DIR = PKG_DIR / "_results"
CKPT_DIR = PKG_DIR / "_checkpoints"
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)
