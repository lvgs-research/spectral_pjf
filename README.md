# The code for the paper *"A Spectral Analysis of Heterogeneous Graph Recommenders for Person-Job Fit"*

This code is for reproducing the experiments and anlysis in the paper.

> **Datasets.** The Tech dataset is not included due to privacy concerns. Tianchi dataset is public and thus the results can be reproduced. 
> Please see section 4 for the download.  

---

## 1. Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

./run_smoke_test.sh
```

This creates a tiny synthetic dataset and runs the whole pipeline at a tiny scale (1 epoch, 5 seeds, capped) in a few minutes on a laptop
CPU. On success it writes:

- `tables/out/perf_tabular.tex`, `perf_full_tabular.tex`, `perf_ranking_tabular.tex`
- `figures/effrank_stages.pdf`


---

## 2. Layout

```
reproducibility/
├── README.md  requirements.txt
├── make_fake_data.py           # synthetic-data generator (fake values only)
├── run_smoke_test.sh           # full pipeline on synthetic data (fast)
├── run_full_pipeline.sh        # full pipeline on REAL data (5 seeds, full epochs)
├── link_prediction_experiment/ # the experiment package
│   ├── run.py                  # Tech entry point (--mode paired|ctrlp8|modelgrid)
│   ├── config.py               # dataclasses + the controlled p1/p2/p3/p8 presets
│   ├── data.py graph_build.py features.py match.py …   # data / graph / feature layer
│   ├── models.py train.py evaluate.py runner.py        # model / training / eval
│   ├── audit.py                # leak-safety audits (A1–A6, A8) run on every config
│   ├── embed_text_tech.py      # Tech seeker/job text → frozen Qwen3/mE5 node embeddings
│   ├── tianchi_pjf/            # the Tianchi (public dataset) half
│   │   ├── build.py            # raw Tianchi tables → graph.pt / pairs.parquet / node_maps.json
│   │   ├── embed_text.py       # Tianchi seeker/job text → frozen Qwen3/mE5 node embeddings
│   │   └── run.py …            # --mode hybrid entry point
│   ├── analysis/               # effective-rank, spectral, PC-projection analyses
│   └── data/                   # ← ALL input artifacts live here (see §5)
└── tables/
    ├── build_dataset.py build_tex_tables.py            # results → LaTeX table bodies
    ├── data/                   # consolidated per-experiment metric JSONs
    └── out/                    # generated .tex table bodies
```

Runtime outputs are written next to the code: `link_prediction_experiment/_results/`,
`_checkpoints/`, `_cache/`, `tianchi_pjf/_results/`, `tianchi_pjf/_checkpoints/`.

---

## 3. The experiments

Following experiments can be performed:

| arm | code (Tech / Tianchi)           | paper name                          | content placement           |
|-----|---------------------------------|-------------------------------------|-----------------------------|
| p1  | `p1_mlp` / `p1_ctrl`            | Content MLP (no graph)              | content only                |
| p2  | `p2_sage_con` / `p2_ctrl`      | Content+graph (naive) — coupled     | content **inside** MP       |
| p3  | `p3_sage_coff` / `p3_ctrl`     | Graph only (content-off)            | **no** content              |
| p8  | `p8_parallel` / `p8_ctrl`      | Content+graph (decoupled) = p1 ∥ p3 | content **out** of MP, late |


---

## 4. Reproducing with real data

`run_full_pipeline.sh` runs the exact commands (5 seeds, full epochs, exact-PIT
evaluation). It expects the real input artifacts under `link_prediction_experiment/data/`
(see §5 for the contract). A GPU is strongly recommended.

```bash
DEVICE=cuda ./run_full_pipeline.sh
```

Individual steps (see the script for the full list):

```bash
# Tech — HeteroSAGE p1/p2/p3, then the strict decoupled p8
python -m link_prediction_experiment.run --mode paired \
    --only p1_mlp,p2_sage_con,p3_sage_coff --seeds 0,1,2,3,4 --tag paired_tech_v2
python -m link_prediction_experiment.run --mode ctrlp8 --seeds 0,1,2,3,4 --tag paired_tech_ctrl

# Tianchi — the hybrid-visibility controlled family
python -m link_prediction_experiment.tianchi_pjf.run --mode hybrid \
    --only p1_ctrl,p2_ctrl,p3_ctrl,p8_ctrl --seeds 5 \
    --artifact-dir link_prediction_experiment/data/tianchi_prepared_bv \
    --save-ckpt --tag paired_tc_ctrl
```

### Tianchi (public dataset)

Obtain the Tianchidataset from the url below:

> <https://tianchi.aliyun.com/competition/entrance/231728/information>

Place them under `link_prediction_experiment/data/tianchi/`.

**The Tianchi graph construction** (`tianchi_pjf/build.py`) — It prepares the necessary input.

```bash
# build the prepared Tianchi artifacts from the downloaded raw tables
python -m link_prediction_experiment.tianchi_pjf.build            # → data/tianchi_prepared/
python -m link_prediction_experiment.tianchi_pjf.build --boundary-validate \
    --out link_prediction_experiment/data/tianchi_prepared_bv     # jieba-refined job→skill edges
```

### Text embeddings (node content)

Every variant except *graph only* consumes frozen LLM text embeddings of the
seeker/job text — one vector per node, aligned to the graph's `node_maps.json`.
Build them once with the `embed_text` pipeline. The default encoder is
**Qwen3-Embedding-0.6B**; swap `--model` for the paper's other encoders
(multilingual-e5-large, Qwen3-Embedding-4B / -8B). Requires
`pip install sentence-transformers`.

```bash
# Tianchi (public) — reads data/tianchi/*.txt + tianchi_prepared/node_maps.json
#   seeker text = profile fields (desired/current role, industry, decoded salary,
#   education, experience); job text = jd_title + job_description.
python -m link_prediction_experiment.tianchi_pjf.embed_text        # → tianchi_prepared/qwen3_{seeker,job}_emb.pt
python -m link_prediction_experiment.tianchi_pjf.embed_text \
    --model intfloat/multilingual-e5-large --chunk-pool            # → tianchi_prepared/me5_{seeker,job}_emb.pt
```

---
