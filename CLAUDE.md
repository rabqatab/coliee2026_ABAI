# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a competition project for **COLIEE 2026** (Competition on Legal Information Extraction/Entailment), focusing on two tasks over Federal Court of Canada case law:

- **Task 1 (Legal Case Retrieval):** Given a query case (citations removed), find which cases it cites ("noticed cases") from a corpus of ~7,350 documents. Evaluated by micro-averaged F1.
- **Task 2 (Legal Case Entailment):** Given a base case and an entailed decision fragment, identify which paragraph(s) from a precedent case logically support that fragment. Evaluated by micro-averaged F1.

## Competition Constraints

- **Open-source models only** — no GPT-4o, Gemini, or other closed-source LLMs
- **Fully automated** — no human intervention at any stage, no manual tuning on test queries
- Submissions: max 3 runs per team per task

## Development Environment

- Python 3.12, managed via `uv`
- **IMPORTANT:** Always use `uv run` to execute Python — never use bare `python` or `python3` commands. The `uv` environment must be used for all execution to ensure correct dependencies.
- Interactive notebooks use **marimo** (not Jupyter)
- No test framework or linter is currently configured

### Common Commands

```bash
uv sync                              # Install dependencies from uv.lock
uv add <package>                     # Add a dependency
uv run marimo edit notebooks/<file>.py   # Open a marimo notebook for editing
uv run python <script>.py            # Run a script
uv run python -m coliee_task1 train         # Run full training pipeline
uv run python -m coliee_task1 predict       # Run prediction on test queries
uv run python -m coliee_task1 evaluate -p output/predictions.json -g data/task1/task1_train_labels_2026.json
```

## Data Layout

All data lives under `data/` and must not be committed to git.

**Task 1** — flat text files + JSON labels:
- `data/task1/task1_train_files_2025/` — 7,350 `.txt` case files (6-digit zero-padded IDs)
- `data/task1/task1_test_files_2025/` — 2,159 `.txt` case files
- `task1_train_labels_2025.json` — `{"query.txt": ["noticed1.txt", ...]}`
- `task1_test_no_labels_2025.json` — same schema, empty lists (submission template)

**Task 2** — hierarchical case folders + JSON labels:
- `data/task2/task2_train_files_2025/{001..825}/` each containing:
  - `base_case.txt` — query case text
  - `entailed_fragment.txt` — the decision fragment to entail
  - `paragraphs/` — numbered `.txt` files from the precedent case (variable count)
- `data/task2/task2_test_files_2025/{826..925}/` — same structure, 100 cases
- `task2_train_labels_2025.json` — `{"001": ["027.txt"], "003": ["003.txt", "004.txt"]}`
- **Caveat:** `task2_test_labels_2025.json` uses comma-separated strings instead of arrays (e.g., `"827": "025.txt, 027.txt"`) — handle both formats when parsing

## Repository Structure

```
src/
  coliee_task1/            # Main pipeline package
    config.py              # Central configuration
    cli.py                 # CLI entry point (train/predict/evaluate)
    pipeline.py            # Pipeline orchestration
    stages/                # Pipeline stage modules (8 stages)
    utils/                 # Shared utilities (metrics, normalization, regex)
  baselines/               # Baseline implementations for comparison
notebooks/                 # Marimo notebooks for EDA
scripts/                   # One-off experiment scripts
docs/                      # Competition rules, reports, plans
  analysis/                # Analysis reports with plots
data/                      # Competition corpus (not in git)
output/                    # Model weights, caches, submissions (not in git)
tests/                     # Test suite
```

All generated plots go in `docs/analysis/plots/`. Each analysis has its own subdirectory under `docs/analysis/` with a `README.md` report.

## Key Reference Documents

- `docs/COLIEE2026_Competition_Rules.md` — full competition rules for Tasks 1 & 2
- `docs/COLIEE2026_Data_Structure.md` — detailed corpus format with examples
- `docs/COLIEE_APPROACHES_REPORT.md` — literature review of winning approaches from prior years
- `docs/LITERATURE_REVIEW.md` — additional research references
- `docs/analysis/eda/` — EDA report and corpus noise analysis
- `docs/analysis/signal_validation/` — Task 1 label signal validation report

## Domain Context

- Prior winning approaches use **hybrid pipelines**: BM25 first-stage retrieval + neural reranking (BERT, T5, etc.)
- Ensemble methods consistently outperform single models
- Graph-based methods (GNNs over citation networks) are an emerging trend
- Case texts are long legal documents; paragraph-level encoding improves results
