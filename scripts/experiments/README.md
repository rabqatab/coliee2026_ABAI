# COLIEE 2026 Journal Experiments

Reproducible, deduplicated rewrites of the trusted ad-hoc journal scripts.
Shared building blocks (the 34-feature list, feature-matrix loader, ranking
metrics, per-query cutoff rules, LightGBM params, result writer) live in
`common.py`. Each experiment writes `output/experiments/<name>.json` (stamped
with the git short SHA + script name) and also refreshes its legacy path.

Run everything with `uv run python`.

## Experiments

| Experiment | What it tests | Reproduce | Headline result |
|---|---|---|---|
| `stage_metrics.py` | Per-stage Recall@k / NDCG@k / MAP over the BM25-RRF pool (how ranking quality evolves stage to stage; Reviewer 1 request). | `uv run python scripts/experiments/stage_metrics.py` (~1-2 min) | BM25-RRF Recall@200 (micro) = **0.5777**; cross-encoder is the strongest reranker (NDCG@10 micro = **0.2715**, the max across stages). |
| `calibration.py` | Authoritative temporal/calibration experiment: clean 3-way chronological split (train_core 70% / calib 10% / test 20%), one model, all decision-rule params fit on calib only, evaluated on test. | `uv run python scripts/experiments/calibration.py` (~2-3 min) | `global_train` (transferred global threshold) F1 = **0.3453**; oracle_global F1 = 0.3523; **calibration_penalty = 0.0070**. Per-query rules do NOT beat the global threshold here. |
| `recall_ceiling_bm25.py` | BM25 k1/b re-tuning to lift the first-stage recall ceiling; pool Recall@{50,100,200} of the multi-view RRF pool, micro-averaged over all 2001 train queries. | `uv run python scripts/experiments/recall_ceiling_bm25.py` (**~25 min — slow**; trust `output/w1/c1_full.json`) | Baseline (k1=1.5, b=0.75) Recall@200 ≈ **0.578**. See `output/w1/c1_full.json` for the best grid point and lift. |

## Pending (C2, GPU / Docker)

`dense_retrieval` (dense first-stage recall) is **pending** its GPU run and is
not yet integrated here. It lives at:

- `scripts/w2_c2_dense_recall.py`
- `scripts/run_c2_docker.sh`
- `scripts/_c2_gpu_smoke.py` (smoke test)

These three files are intentionally left in `scripts/` (not moved into this
package) because a GPU job actively uses them. Integrate as
`scripts/experiments/dense_retrieval.py` after the run completes.

## Interpretation

See `paperwork/JOURNAL_PLAN.md` §9 ("W1 results — diagnosis re-prioritized")
for how these numbers feed the paper narrative.

## Archived / superseded

Superseded scripts are preserved (not deleted) in `scripts/_archive/`:

- `w1_c1_bm25_sweep.py` — subsample version of the recall-ceiling sweep;
  superseded by `recall_ceiling_bm25.py` (full 2001 queries).
- `w1_d_refine.py` — reported a **0.188** "calibration penalty". That figure was
  a **score-scale artifact** (threshold transferred across models / score
  scales). The clean 3-way split in `calibration.py` corrects it to **0.007**:
  the penalty is essentially zero, so the meta-learner's global threshold
  transfers well across time and per-query rules are not needed.
- `w1_d_temporal_split.py` — 45-feature, in-sample-threshold temporal split;
  superseded by `calibration.py`.

## Lesson recorded

The w1_d_refine "0.188 calibration penalty" was not a real generalization gap —
it was an artifact of comparing thresholds across differently-scaled model
outputs. With a single model and a held-out calibration split (`calibration.py`),
the true penalty is 0.007.
