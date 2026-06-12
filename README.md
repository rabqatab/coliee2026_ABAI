# COLIEE 2026 Task 1 — Legal Case Retrieval

**Team:** ABAI (AlphaBridge)

A four-stage hybrid retrieval pipeline for [COLIEE 2026](https://coliee.org/) Task 1: given a query case with citations removed, retrieve the cases it cites from a corpus of ~9,500 Federal Court of Canada decisions. Citation passages are suppressed with `<FRAGMENT_SUPPRESSED>` markers, creating a lexical gap between query and cited cases.

## Pipeline

Four independently trained stages. Stages 1–3 also emit features consumed by the Stage 4 meta-learner (34 features total).

| Stage | Component | Description | Features |
|-------|-----------|-------------|----------|
| 1 | Retrieval & Fusion | Multi-view BM25 (full-doc + citation-context windows) with RRF fusion → top-200 candidate pool | 9 |
| 2 | Neural Reranking | Fine-tuned BGE-large bi-encoder (LoRA) + BGE-M3 (dense/sparse/ColBERT) + DeBERTa-v3-large cross-encoder (selective truncation) | 8 |
| 3 | Graph-Based Reranking | GraphRAG Lite (regex entities → bipartite graph → Leiden communities) + 2-layer GAT on corpus k-NN graph | 11 |
| 4 | Meta-Learner | LightGBM fusing all 34 features (incl. 6 score-distribution features), GroupKFold 5-fold CV | 6 |

All models are open-source. No closed-source LLMs or external APIs.

## Setup

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

```bash
# Train the full pipeline
uv run python -m coliee_task1 train

# Predict on test queries
uv run python -m coliee_task1 predict --output output/predictions.json

# Evaluate predictions against gold labels
uv run python -m coliee_task1 evaluate \
  -p output/predictions.json \
  -g data/task1/task1_train_labels_2026.json
```

## Repository Structure

```
src/
  coliee_task1/            # Main pipeline package
    config.py              # Central configuration
    cli.py                 # CLI entry point (train/predict/evaluate)
    pipeline.py            # Pipeline orchestration
    stages/                # Pipeline stage modules
    utils/                 # Shared utilities (metrics, normalization, regex)
  baselines/               # Baseline reimplementations for comparison
  analysis/                # EDA and label-signal validation scripts
scripts/                   # Experiment scripts
tests/                     # Unit tests
data/                      # Competition corpus (not in repo)
output/                    # Model weights, caches, submissions (not in repo)
```

## Results

| Evaluation | F1 | Precision | Recall |
|------------|-----|-----------|--------|
| Cross-validation (5-fold, honest) | 0.311 | 0.368 | 0.270 |
| **Official test set** (run 2, best) | **0.177** | 0.160 | 0.199 |

Our best run ranked **35th of 54 official runs** (15th of 22 teams by each team's best run). The ~43% relative drop from cross-validation to test is the central finding, attributed to three factors: a **57.8% recall ceiling** from BM25 top-200 retrieval, temporal distribution shift, and threshold miscalibration from gold-injected training pools.

**Ablation (ΔF1 when component removed):**
- Cross-encoder −0.075 (24% relative — dominant component)
- Lexical features −0.023
- GNN + BGE-M3 −0.012
- GraphRAG −0.006
- Bi-encoder −0.001

The camera-ready paper appears in the COLIEE 2026 proceedings.

## License

Competition use only.
