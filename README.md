# COLIEE 2026 Task 1 — Legal Case Retrieval

**Team:** AlphaBridge-TeamAI

An 8-stage hybrid retrieval pipeline for [COLIEE 2026](https://sites.ualberta.ca/~rabelo/COLIEE2026/) Task 1: given a query case with citations removed, retrieve the cases it cites from a corpus of ~9,500 Federal Court of Canada decisions.

## Pipeline

| Stage | Component | Description |
|-------|-----------|-------------|
| 1 | Preprocessing | Citation-context extraction around `<FRAGMENT_SUPPRESSED>` markers |
| 2 | BM25 | Multi-view BM25 (full-doc + per-context) with RRF fusion |
| 3 | Bi-encoder | Fine-tuned BGE-large-en-v1.5 with LoRA |
| 3b | Multi-signal | BGE-M3 dense + sparse + ColBERT retrieval |
| 4 | Cross-encoder | Fine-tuned DeBERTa-v3-large with citation-context-aware truncation |
| 5 | GraphRAG | Regex entity extraction, bipartite graph, Leiden communities |
| 5.5 | GNN | 2-layer GAT on corpus k-NN graph for score refinement |
| 6 | Meta-learner | LightGBM fusing 38 features, GroupKFold 5-fold CV |

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
scripts/                   # Experiment scripts
tests/                     # Unit tests
data/                      # Competition corpus (not in repo)
output/                    # Model weights, caches, submissions (not in repo)
```

## Results

**Cross-validation (honest, no leakage):** F1 = 0.3108 (P = 0.3720, R = 0.2669)

Ablation highlights:
- Cross-encoder contributes the most (+0.075 F1)
- Lexical features are second (+0.023 F1)
- GNN + BGE-M3 combined contribute +0.012 F1

## License

Competition use only.
