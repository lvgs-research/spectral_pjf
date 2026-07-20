#!/usr/bin/env bash
# End-to-end SMOKE TEST on fully synthetic data.
#
# Generates fake data and runs the ENTIRE pipeline (both datasets, both graph
# convolutions, the encoder axis, the effective-rank / spectral / PC-projection
# analyses, and the LaTeX table builders) at a tiny scale (1 epoch, few seeds,
# capped) so it finishes in a couple of minutes on a laptop CPU. 
set -euo pipefail
cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONUNBUFFERED=1

PY=${PY:-python3}
SEEDS=0,1,2,3,4
CAP="--device cpu --epochs 1 --max-train-pos 600 --max-eval-pos 300 --max-eval-exposed 600 --chunks-train 3 --chunks-eval 2"
ENCS="me5 qwen3_4b qwen3_8b"

echo "############################################################"
echo "# 0. generate synthetic data (all values are fake)"
echo "############################################################"
$PY make_fake_data.py --encoders

echo "############################################################"
echo "# 0b. Text-embedding pipeline — dry-run both datasets"
echo "#     (builds the seeker/job texts + exercises the plumbing;"
echo "#      --dry-run skips the LLM so no model download is needed)"
echo "############################################################"
$PY -m link_prediction_experiment.tianchi_pjf.embed_text --dry-run
$PY -m link_prediction_experiment.embed_text_tech --dry-run

echo "############################################################"
echo "# 1. Tech — HeteroSAGE controlled family (p1/p2/p3 + p8)"
echo "############################################################"
$PY -m link_prediction_experiment.run --mode paired \
    --only p1_mlp,p2_sage_con,p3_sage_coff --seeds $SEEDS --tag paired_tech_v2 $CAP
$PY -m link_prediction_experiment.run --mode ctrlp8 --seeds $SEEDS --tag paired_tech_ctrl $CAP

echo "############################################################"
echo "# 2. Tech — re-score checkpoints -> shards (feeds the tables)"
echo "############################################################"
RESCORE_TAGS=paired_tech_v2,paired_tech_ctrl RESCORE_DEVICE=cpu \
    $PY link_prediction_experiment/analysis/rescore_paired_tech.py

echo "############################################################"
echo "# 3. Tech — HeteroGATv2 (model-grid hetgat arms)"
echo "############################################################"
$PY -m link_prediction_experiment.run --mode modelgrid \
    --only p2_hetgat,p3_hetgat,p8_hetgat --seeds $SEEDS --tag paired_tech_modelgrid $CAP

echo "############################################################"
echo "# 4. Tech — encoder axis (mE5 / Qwen3-4B / Qwen3-8B)"
echo "############################################################"
for enc in $ENCS; do
  $PY -m link_prediction_experiment.run --mode paired \
      --only p1_mlp,p2_sage_con,p3_sage_coff --seeds $SEEDS \
      --text-emb "$enc" --tag "paired_tech_ctrl123_${enc}" $CAP
  $PY -m link_prediction_experiment.run --mode ctrlp8 --seeds $SEEDS \
      --text-emb "$enc" --tag "paired_tech_ctrl_${enc}" $CAP
done

echo "############################################################"
echo "# 5. Tianchi — HeteroSAGE + GATv2 + encoder axis"
echo "############################################################"
TCART=link_prediction_experiment/data/tianchi_prepared_bv
$PY -m link_prediction_experiment.tianchi_pjf.run --mode hybrid \
    --only p1_ctrl,p2_ctrl,p3_ctrl,p8_ctrl --seeds 5 \
    --artifact-dir $TCART --save-ckpt --tag paired_tc_ctrl --device cpu --epochs 1
$PY -m link_prediction_experiment.tianchi_pjf.run --mode hybrid \
    --only p2_hetgat,p3_hetgat,p8_hetgat --seeds 5 \
    --artifact-dir $TCART --save-ckpt --tag paired_tc_modelgrid --device cpu --epochs 1
for enc in $ENCS; do
  $PY -m link_prediction_experiment.tianchi_pjf.run --mode hybrid \
      --only p1_ctrl,p2_ctrl,p3_ctrl --seeds 5 --emb-model "$enc" \
      --artifact-dir $TCART --save-ckpt --tag "paired_tc_ctrl123_${enc}" --device cpu --epochs 1
  $PY -m link_prediction_experiment.tianchi_pjf.run --mode hybrid \
      --only p8_ctrl --seeds 5 --emb-model "$enc" \
      --artifact-dir $TCART --save-ckpt --tag "paired_tc_ctrl_${enc}" --device cpu --epochs 1
done

echo "############################################################"
echo "# 6. Analysis — effective rank, spectral, PC projection"
echo "############################################################"
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

echo "############################################################"
echo "# 7. Tables — consolidate results + build LaTeX table bodies"
echo "############################################################"
$PY tables/build_dataset.py
$PY tables/build_tex_tables.py

echo
echo "SMOKE TEST COMPLETE — generated:"
ls -1 tables/out/*.tex figures/*.pdf 2>/dev/null || true
