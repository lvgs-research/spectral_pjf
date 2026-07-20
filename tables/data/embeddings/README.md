# Saved intermediate features (effective-rank trajectory)

One `.npz` per `<conv>__<experiment>__seed<s>.npz` under `tech/` and `tianchi/`, row-aligned to
the pos+neg TEST edges (Tech 1-day exact PIT; Tianchi static hybrid). Produced by
`analysis/effrank_trajectory.py --dump-embeddings`. RAW inputs are NOT saved (too large).

Keys per file:
- `seeker_idx`, `job_idx` (int32), `y` (int8: Tech passed / TC joint deliver&satisfy)
- `predec_seeker`, `predec_job` (float32) — pre-decoder node embeddings `model.encode(...)`
- `prelogit_accept` (float32) — accept-head pre-logit `head_accept[:-1](concat[z_m,z_j,match])`
- `prelogit_pass`   (float32) — PASS/FAIL (cond) head pre-logit `head_cond[:-1](...)`

Load: `d = np.load('tech/sage__p8__seed0.npz'); d['predec_seeker'], d['prelogit_pass'], ...`
`gatv2__p1__*` are copies of `sage__p1__*` (a content MLP has no graph convolution).
