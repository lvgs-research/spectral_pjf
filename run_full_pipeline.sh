#!/usr/bin/env bash
# FULL reproduction pipeline (real settings: 5 seeds, full epochs, exact PIT).
#
# This runs the exact commands that produce the paper's Table 2 (perf_full_tabular),
# the effective-rank figure, and the spectral / PC-projection tables — but it
# requires the REAL input artifacts to be present under
# link_prediction_experiment/data/ (see README.md, "Bring your own data"):
#
#   data/tech/            the proprietary "Tech" dataset (NOT distributed)
#   data/tech_graph/      hetero_graph.pt + node_maps.json
#   data/tianchi/         table1_user.txt / table2_jd.txt / table3_action.txt (public — download)
#
# The Tianchi graph.pt / pairs.parquet / node_maps.json are BUILT below from the raw
# tables by tianchi_pjf/build.py, and the frozen node-content embeddings
# (data/{tech,tianchi}_prepared/<tag>_{seeker,job}_emb.pt) are BUILT below by the
# embed_text pipeline (no pre-built graph or embeddings needed). The Tech dataset is
# proprietary and NOT part of this archive; only the public Tianchi half is
# reproducible by third parties (see README).
# A GPU is strongly recommended. To validate the plumbing WITHOUT real data,
# run ./run_smoke_test.sh instead (it fabricates inputs).
set -euo pipefail
cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONUNBUFFERED=1

PY=${PY:-python3}
DEVICE=${DEVICE:-cuda}          # cuda | mps | cpu
SEEDS=0,1,2,3,4
ENCS="me5 qwen3_4b qwen3_8b"
TCART=link_prediction_experiment/data/tianchi_prepared_bv

# ----------------------------------------------------------------------------
# Text embeddings (frozen node content) — Tech side, all four encoders.
# Qwen3-0.6B is the default (tag qwen3); the larger Qwen3 sizes/mE5 need explicit
# --tag/--model. Reads data/tech/ masters + tech_graph/node_maps.json.
# ----------------------------------------------------------------------------
$PY -m link_prediction_experiment.embed_text_tech
$PY -m link_prediction_experiment.embed_text_tech --model intfloat/multilingual-e5-large --chunk-pool
$PY -m link_prediction_experiment.embed_text_tech --model Qwen/Qwen3-Embedding-4B --tag qwen3_4b --bf16
$PY -m link_prediction_experiment.embed_text_tech --model Qwen/Qwen3-Embedding-8B --tag qwen3_8b --bf16

# ----------------------------------------------------------------------------
# Tech (proprietary) — HeteroSAGE controlled family + re-score
# ----------------------------------------------------------------------------
$PY -m link_prediction_experiment.run --mode paired \
    --only p1_mlp,p2_sage_con,p3_sage_coff --seeds $SEEDS --tag paired_tech_v2 --device $DEVICE
$PY -m link_prediction_experiment.run --mode ctrlp8 --seeds $SEEDS --tag paired_tech_ctrl --device $DEVICE
RESCORE_TAGS=paired_tech_v2,paired_tech_ctrl RESCORE_DEVICE=$DEVICE \
    $PY link_prediction_experiment/analysis/rescore_paired_tech.py

# Tech — HeteroGATv2 (model grid) + encoder axis
$PY -m link_prediction_experiment.run --mode modelgrid \
    --only p2_hetgat,p3_hetgat,p8_hetgat --seeds $SEEDS --tag paired_tech_modelgrid --device $DEVICE
for enc in $ENCS; do
  $PY -m link_prediction_experiment.run --mode paired \
      --only p1_mlp,p2_sage_con,p3_sage_coff --seeds $SEEDS \
      --text-emb "$enc" --tag "paired_tech_ctrl123_${enc}" --device $DEVICE
  $PY -m link_prediction_experiment.run --mode ctrlp8 --seeds $SEEDS \
      --text-emb "$enc" --tag "paired_tech_ctrl_${enc}" --device $DEVICE
done

# ----------------------------------------------------------------------------
# Tianchi (public) — build the graph from the raw tables, then train
# ----------------------------------------------------------------------------
# Construct graph.pt / pairs.parquet / node_maps.json from data/tianchi/*.txt.
# The --boundary-validate variant (the graph the paper trains on) needs `pip install jieba`.
$PY -m link_prediction_experiment.tianchi_pjf.build
$PY -m link_prediction_experiment.tianchi_pjf.build --boundary-validate --out "$TCART"
# frozen node-content embeddings (seeker+job aligned to tianchi_prepared/node_maps.json;
# seeker/job are shared with the boundary-validated build, so these serve both).
$PY -m link_prediction_experiment.tianchi_pjf.embed_text
$PY -m link_prediction_experiment.tianchi_pjf.embed_text --model intfloat/multilingual-e5-large --chunk-pool
$PY -m link_prediction_experiment.tianchi_pjf.embed_text --model Qwen/Qwen3-Embedding-4B --tag qwen3_4b --bf16
$PY -m link_prediction_experiment.tianchi_pjf.embed_text --model Qwen/Qwen3-Embedding-8B --tag qwen3_8b --bf16
$PY -m link_prediction_experiment.tianchi_pjf.run --mode hybrid \
    --only p1_ctrl,p2_ctrl,p3_ctrl,p8_ctrl --seeds 5 \
    --artifact-dir $TCART --save-ckpt --tag paired_tc_ctrl --device $DEVICE
$PY -m link_prediction_experiment.tianchi_pjf.run --mode hybrid \
    --only p2_hetgat,p3_hetgat,p8_hetgat --seeds 5 \
    --artifact-dir $TCART --save-ckpt --tag paired_tc_modelgrid --device $DEVICE
for enc in $ENCS; do
  $PY -m link_prediction_experiment.tianchi_pjf.run --mode hybrid \
      --only p1_ctrl,p2_ctrl,p3_ctrl --seeds 5 --emb-model "$enc" \
      --artifact-dir $TCART --save-ckpt --tag "paired_tc_ctrl123_${enc}" --device $DEVICE
  $PY -m link_prediction_experiment.tianchi_pjf.run --mode hybrid \
      --only p8_ctrl --seeds 5 --emb-model "$enc" \
      --artifact-dir $TCART --save-ckpt --tag "paired_tc_ctrl_${enc}" --device $DEVICE
done

# ----------------------------------------------------------------------------
# Analysis — effective rank, spectral diagnostics, PC-projection ablation
# ----------------------------------------------------------------------------
$PY -m link_prediction_experiment.analysis.effrank_trajectory --dataset tech --dump-embeddings
$PY -m link_prediction_experiment.analysis.effrank_trajectory --dataset tianchi --dump-embeddings
$PY -m link_prediction_experiment.analysis.effrank_trajectory --dataset tech --encoders
$PY -m link_prediction_experiment.analysis.effrank_trajectory --dataset tianchi --encoders
$PY -m link_prediction_experiment.analysis.effrank_trajectory --dump-finalize
$PY link_prediction_experiment/analysis/effrank_spectral_roy.py
$PY link_prediction_experiment/analysis/effrank_spectral_roy.py --encoders
$PY -m link_prediction_experiment.analysis.effrank_raw_unique_roy
$PY -m link_prediction_experiment.analysis.pc1_semantics
$PY -m link_prediction_experiment.analysis.pc_projection_sweep
$PY link_prediction_experiment/analysis/make_effrank_fig.py

# ----------------------------------------------------------------------------
# Tables — consolidate + build the LaTeX table bodies (tables/out/*.tex)
# ----------------------------------------------------------------------------
$PY tables/build_dataset.py
$PY tables/build_tex_tables.py

echo "Done. LaTeX table bodies in tables/out/, the effective-rank figure in figures/."
