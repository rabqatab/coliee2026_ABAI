# Codebase Refactor Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up dead code, reorganize package structure, and fix stale artifacts so the codebase reflects what's actually used in the Option C pipeline.

**Architecture:** The active pipeline lives in `src/graphrag/` with 12 core modules. There are ~8 legacy modules (1,200+ lines) that served earlier experimentation but are now dead code. The `src/baselines/` package (reference implementations of prior winners) is independent and well-structured. The refactor groups files by role, removes dead code, and fixes stale cross-references.

**Tech Stack:** Python 3.12, uv, pytest, hatch build system

---

## Current State Analysis

### What's Active (KEEP — the Option C pipeline)

| Module | Lines | Role |
|--------|-------|------|
| `config.py` | 107 | Central config |
| `preprocess.py` | 79 | Text cleaning |
| `citation_context.py` | 236 | Citation window extraction |
| `bm25.py` | 183 | BM25 retrieval + RRF |
| `finetune_biencoder.py` | 229 | BGE-large + LoRA |
| `finetune_crossencoder.py` | 286 | DeBERTa-v3 reranker |
| `graphrag_lite.py` | 404 | Entity graph + Leiden |
| `extract_regex.py` | 71 | Regex entity extraction |
| `normalize.py` | 111 | Entity normalization |
| `meta_learner.py` | 344 | LightGBM fusion |
| `metrics.py` | 96 | F1, threshold optimization |
| `run_pipeline_v2.py` | 805 | Pipeline orchestrator |
| `baselines.py` | 688 | Quick baselines (BM25/TF-IDF/LightGBM) |
| `__init__.py` | 1 | Package marker |
| **Total** | **3,640** | |

### What's Dead (REMOVE)

| Module | Lines | Why dead |
|--------|-------|----------|
| `graph.py` | 117 | Superseded by `graphrag_lite.py` |
| `community.py` | 218 | Superseded by `graphrag_lite.py` |
| `retrieve.py` | 159 | Old retrieval, not imported by pipeline |
| `reasoning.py` | 180 | LLM reasoning, not used |
| `run_index.py` | 123 | Old indexing script |
| `run_pipeline.py` | 203 | V1 pipeline, superseded by v2 |
| `embed.py` | 53 | Legacy Ollama embeddings |
| `run_benchmark_embed.py` | 219 | Benchmark script, one-off |
| `run_benchmark_llm.py` | 169 | Benchmark script, one-off |
| **Total** | **1,441** | |

### Borderline (MOVE to `legacy/`)

| Module | Lines | Status |
|--------|-------|--------|
| `ollama_client.py` | 125 | Used by `run_extract.py` (LLM extraction still running) |
| `extract_llm.py` | 124 | Used by `run_extract.py` |
| `run_extract.py` | 210 | Active background process but not part of Option C |

### Stale Tests (tests for dead modules)

| Test | Tests for |
|------|-----------|
| `test_graph.py` | Dead `graph.py` |
| `test_retrieve.py` | Dead `retrieve.py` |
| `test_ollama_client.py` | Borderline `ollama_client.py` |

### Stale Scripts

| Script | Issue |
|--------|-------|
| `scripts/train_pipeline.py` | Duplicate of `run_pipeline_v2.py`, hardcoded Docker paths, uses old `build_feature_matrix` signature (missing lexical_features) |
| `scripts/run_ablation.py` | Uses old LGBM params + old feature groups (missing lexical group), expects `feature_matrix.parquet` that pipeline doesn't produce |

---

## File Structure After Refactor

```
src/graphrag/
├── __init__.py
├── config.py              (central config — unchanged)
├── preprocess.py          (text cleaning — unchanged)
├── citation_context.py    (citation windows — unchanged)
├── bm25.py                (retrieval — unchanged)
├── extract_regex.py       (regex entities — unchanged)
├── normalize.py           (entity normalization — unchanged)
├── graphrag_lite.py       (entity graph + Leiden — unchanged)
├── finetune_biencoder.py  (bi-encoder — unchanged)
├── finetune_crossencoder.py (cross-encoder — unchanged)
├── meta_learner.py        (LightGBM fusion — unchanged)
├── metrics.py             (evaluation — unchanged)
├── run_pipeline_v2.py     (pipeline orchestrator — renamed to run_pipeline.py)
├── baselines.py           (quick baselines — unchanged)
├── ollama_client.py       (kept for LLM extraction)
├── extract_llm.py         (kept for LLM extraction)
├── run_extract.py         (kept for LLM extraction)
scripts/
├── run_ablation.py        (updated feature groups + params)
tests/
├── conftest.py            (updated — remove dead fixtures)
├── test_extract_regex.py  (keep)
├── test_normalize.py      (keep)
├── baselines/             (keep as-is)
```

**Removed:** `graph.py`, `community.py`, `retrieve.py`, `reasoning.py`, `run_index.py`, `embed.py`, `run_benchmark_embed.py`, `run_benchmark_llm.py`, old `run_pipeline.py`
**Removed tests:** `test_graph.py`, `test_retrieve.py`, `test_ollama_client.py`
**Removed scripts:** `scripts/train_pipeline.py` (redundant with `run_pipeline_v2.py`)

---

## Chunk 1: Remove Dead Code

### Task 1: Delete dead modules and their tests

**Files:**
- Delete: `src/graphrag/graph.py`
- Delete: `src/graphrag/community.py`
- Delete: `src/graphrag/retrieve.py`
- Delete: `src/graphrag/reasoning.py`
- Delete: `src/graphrag/run_index.py`
- Delete: `src/graphrag/embed.py`
- Delete: `src/graphrag/run_benchmark_embed.py`
- Delete: `src/graphrag/run_benchmark_llm.py`
- Delete: `src/graphrag/run_pipeline.py` (old v1)
- Delete: `tests/test_graph.py`
- Delete: `tests/test_retrieve.py`
- Delete: `tests/test_ollama_client.py`

- [ ] **Step 1: Verify no active code imports these modules**

```bash
# Should return ONLY cross-references between dead modules (not from active ones)
uv run python -c "
from graphrag.config import *
from graphrag.preprocess import *
from graphrag.citation_context import *
from graphrag.bm25 import *
from graphrag.extract_regex import *
from graphrag.normalize import *
from graphrag.graphrag_lite import *
from graphrag.meta_learner import *
from graphrag.metrics import *
print('All active imports OK')
"
```

Expected: `All active imports OK` — confirms none of the active modules import dead ones.

- [ ] **Step 2: Delete dead modules**

```bash
cd /home/alphabridge/Research/coliee2026
git rm src/graphrag/graph.py
git rm src/graphrag/community.py
git rm src/graphrag/retrieve.py
git rm src/graphrag/reasoning.py
git rm src/graphrag/run_index.py
git rm src/graphrag/embed.py
git rm src/graphrag/run_benchmark_embed.py
git rm src/graphrag/run_benchmark_llm.py
git rm src/graphrag/run_pipeline.py
```

- [ ] **Step 3: Delete stale tests**

```bash
git rm tests/test_graph.py
git rm tests/test_retrieve.py
git rm tests/test_ollama_client.py
```

- [ ] **Step 4: Run existing tests to verify nothing broke**

```bash
uv run pytest tests/test_extract_regex.py tests/test_normalize.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: remove 9 dead modules and 3 stale tests (1,441 lines)

Removed legacy modules superseded by graphrag_lite.py and run_pipeline_v2.py:
graph.py, community.py, retrieve.py, reasoning.py, run_index.py, embed.py,
run_benchmark_embed.py, run_benchmark_llm.py, run_pipeline.py (v1).

Kept ollama_client.py, extract_llm.py, run_extract.py (LLM extraction still active)."
```

---

### Task 2: Remove redundant train_pipeline.py script

**Files:**
- Delete: `scripts/train_pipeline.py`

This is a stale copy of the pipeline logic. `run_pipeline_v2.py` is the canonical entry point and has all the latest fixes (lexical features, negative sampling, threshold calibration). The script also has hardcoded Docker paths and uses the old `build_feature_matrix` signature.

- [ ] **Step 1: Delete the script**

```bash
git rm scripts/train_pipeline.py
```

- [ ] **Step 2: Commit**

```bash
git commit -m "refactor: remove stale train_pipeline.py (use run_pipeline_v2.py instead)"
```

---

## Chunk 2: Rename and Fix Stale References

### Task 3: Rename run_pipeline_v2.py → run_pipeline.py

Now that v1 is deleted, the "v2" suffix is unnecessary.

**Files:**
- Rename: `src/graphrag/run_pipeline_v2.py` → `src/graphrag/run_pipeline.py`

- [ ] **Step 1: Rename**

```bash
git mv src/graphrag/run_pipeline_v2.py src/graphrag/run_pipeline.py
```

- [ ] **Step 2: Verify CLI still works**

```bash
uv run python -m graphrag.run_pipeline --help
```

Expected: Shows argparse help with `train`/`predict` modes.

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor: rename run_pipeline_v2.py → run_pipeline.py (v1 removed)"
```

---

### Task 4: Update ablation script

The ablation script has stale LGBM params (old `is_unbalance`, old `num_leaves=63`, `lr=0.05`) and is missing the `lexical` feature group. Update it to match the current pipeline config.

**Files:**
- Modify: `scripts/run_ablation.py`

- [ ] **Step 1: Update feature groups to include lexical**

Add the lexical feature group and include it in ablation configs:

```python
FEATURE_GROUPS = {
    "bm25": ["bm25_score", "bm25_rrf_score"],
    "lexical": ["tfidf_cosine", "jaccard", "shared_bigrams", "length_ratio", "shared_legal_terms"],
    "context": ["n_context_matches", "max_context_bm25"],
    "biencoder": ["biencoder_score", "biencoder_rank"],
    "crossencoder": ["crossencoder_score", "crossencoder_rank"],
    "graphrag": [
        "same_community_0.5", "same_community_1.0", "same_community_2.0",
        "community_jaccard", "shared_statutes", "shared_judges",
        "same_domain", "same_outcome", "entity_overlap_score",
    ],
}

ABLATION_CONFIGS = {
    "A: BM25 only":             ["bm25"],
    "B: + Lexical":             ["bm25", "lexical"],
    "C: + Citation Context":    ["bm25", "lexical", "context"],
    "D: + Bi-encoder":          ["bm25", "lexical", "context", "biencoder"],
    "E: + Cross-encoder":       ["bm25", "lexical", "context", "biencoder", "crossencoder"],
    "F: + GraphRAG Lite":       ["bm25", "lexical", "context", "biencoder", "crossencoder", "graphrag"],
    # Drop-one from full
    "G: Full - GraphRAG":       ["bm25", "lexical", "context", "biencoder", "crossencoder"],
    "H: Full - Cross-encoder":  ["bm25", "lexical", "context", "biencoder", "graphrag"],
    "I: Full - Bi-encoder":     ["bm25", "lexical", "context", "crossencoder", "graphrag"],
    "J: Full - Context":        ["bm25", "lexical", "biencoder", "crossencoder", "graphrag"],
    "K: Full - Lexical":        ["bm25", "context", "biencoder", "crossencoder", "graphrag"],
}
```

- [ ] **Step 2: Update LGBM params to match config.py**

```python
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.02,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "min_child_samples": 20,
}
N_ESTIMATORS = 800
EARLY_STOPPING = 80
```

- [ ] **Step 3: Update drop-one analysis section to use correct full config key**

Change `"E: + GraphRAG Lite"` → `"F: + GraphRAG Lite"` in the drop-one analysis, and adjust the drop config keys (G through K).

- [ ] **Step 4: Commit**

```bash
git add scripts/run_ablation.py
git commit -m "refactor: update ablation script with lexical features and tuned LGBM params"
```

---

## Chunk 3: Housekeeping

### Task 5: Clean up pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update project description**

Change: `description = "Add your description here"`
To: `description = "COLIEE 2026 Legal Case Retrieval — Hybrid GraphRAG Pipeline"`

- [ ] **Step 2: Remove unused dependency**

`rank-bm25` is not imported anywhere (the pipeline uses a custom BM25 implementation in `bm25.py`). Also `networkx` — only `igraph` is used.

Verify:
```bash
grep -r "rank_bm25\|from rank_bm25\|import rank_bm25" src/ tests/ scripts/
grep -r "import networkx\|from networkx" src/ tests/ scripts/
```

If both return empty, remove them from dependencies.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
uv sync
git add uv.lock
git commit -m "chore: update project description, remove unused deps (rank-bm25, networkx)"
```

---

### Task 6: Clean up tests/conftest.py

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Read conftest and remove fixtures for deleted modules**

If it has fixtures for `graph.py`, `retrieve.py`, or `ollama_client.py`, remove them.

- [ ] **Step 2: Run full test suite**

```bash
uv run pytest tests/ -v --ignore=tests/baselines/
```

Expected: All remaining tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "chore: clean up test fixtures for removed modules"
```

---

### Task 7: Update CLAUDE.md repository structure

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the Repository Structure section**

```markdown
## Repository Structure

```
src/graphrag/          # Option C pipeline (12 active modules + 3 LLM extraction)
src/baselines/         # Reference implementations of prior winning approaches
src/analysis/          # EDA and signal validation scripts (standalone)
scripts/               # Ablation study runner
notebooks/             # Marimo notebooks for EDA
docs/                  # Competition rules, literature review, approaches report
docs/analysis/         # Analysis reports with plots
data/                  # Competition corpus (not in git)
output/                # Pipeline cache, models, predictions (not in git)
tests/                 # pytest tests for core modules and baselines
```
```

- [ ] **Step 2: Update Common Commands section**

Add:
```bash
uv run python -m graphrag.run_pipeline train --no-finetune  # CPU training with CV
uv run python -m graphrag.run_pipeline predict               # Generate test predictions
uv run python scripts/run_ablation.py                        # Feature group ablation
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md to reflect refactored structure"
```

---

## Summary of Changes

| Metric | Before | After |
|--------|--------|-------|
| Files in `src/graphrag/` | 25 | 16 |
| Lines in `src/graphrag/` | 5,540 | ~4,100 |
| Dead code lines | 1,441 | 0 |
| Test files | 9 | 6 |
| Scripts | 2 | 1 |
| Unused dependencies | 2 | 0 |

**Not touched:** `src/baselines/` (well-structured, independent), `src/analysis/` (standalone scripts), `notebooks/`, `docs/`.

**Intentionally kept:** `ollama_client.py`, `extract_llm.py`, `run_extract.py` — LLM extraction is still running and may be useful later. If the LLM extraction process is confirmed abandoned, these 3 files (459 lines) can be removed in a follow-up.
