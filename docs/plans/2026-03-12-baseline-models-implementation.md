# SoTA Baseline Models Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 5 SoTA baseline models from COLIEE 2024-2025 winners, benchmarked against our Option C GraphRAG pipeline on the 2026 dataset.

**Architecture:** Each baseline implements a `BaselineModel` ABC with `train()` and `predict()` methods. A shared harness loads the corpus once, runs each baseline, and produces a comparison table with micro-averaged F1. All baselines share a single BM25 index and data loader. GPU-requiring baselines run inside the existing Docker container `coliee_optionc`.

**Tech Stack:** Python 3.12, LightGBM, sentence-transformers, PyTorch, torch-geometric (CaseLink only), scikit-learn.

**Design Spec:** `docs/superpowers/specs/2026-03-11-baseline-models-design.md`

---

## File Structure

```
src/baselines/
├── __init__.py
├── common/
│   ├── __init__.py
│   ├── base_model.py          # BaselineModel ABC
│   ├── data_loader.py         # Corpus loading + 80/20 chronological split
│   ├── bm25_index.py          # Shared BM25 wrapper (reuses graphrag.bm25)
│   └── run_harness.py         # Run baselines, comparison table, output
├── bm25/
│   ├── __init__.py
│   └── model.py               # Vanilla BM25 baseline (~50 lines)
├── jnlp/
│   ├── __init__.py
│   ├── sailer_encoder.py      # SAILER checkpoint + embedding cache
│   ├── features.py            # QLD, BM25 paragraph features
│   └── model.py               # LightGBM reranker
├── tqm/
│   ├── __init__.py
│   ├── bi_encoder.py          # Fine-tune sentence-transformer + cache
│   ├── features.py            # TF-IDF, dense cosine, heuristics
│   └── model.py               # LambdaMART via LightGBM lambdarank
├── umnlp/
│   ├── __init__.py
│   ├── propositions.py        # Citation context extraction + similarity
│   ├── judge_match.py         # Judge name extraction and matching
│   ├── features.py            # Paragraph/sentence similarity, quotation
│   └── model.py               # FF NN classifier (PyTorch)
├── caselink/
│   ├── __init__.py
│   ├── graph.py               # Global case graph construction
│   ├── node_features.py       # Text -> embedding for each node
│   ├── gnn.py                 # GNN architecture (PyG)
│   └── model.py               # Training loop + inductive inference
├── graphrag_adapter/
│   ├── __init__.py
│   └── model.py               # Thin wrapper around Option C pipeline
├── run_baseline.py            # CLI: run single baseline by name
└── run_all.py                 # CLI: run all baselines, produce comparison

tests/baselines/
├── __init__.py
├── test_data_loader.py
├── test_bm25_baseline.py
├── test_jnlp.py
├── test_tqm.py
├── test_umnlp.py
├── test_caselink.py
└── conftest.py                # Shared fixtures (mini corpus, mini labels)
```

**Modifications to existing files:**
- `pyproject.toml` — add `src/baselines` to wheel packages, add `torch-geometric` dependency

---

## Chunk 1: Shared Infrastructure + BM25 Baseline (Tasks 1-7)

### Task 1: Project scaffolding

**Files:**
- Create: `src/baselines/__init__.py` and all subdirectory `__init__.py` files
- Modify: `pyproject.toml`

- [ ] **Step 1: Create directory structure and empty `__init__.py` files**

```bash
mkdir -p src/baselines/common src/baselines/bm25 src/baselines/jnlp \
         src/baselines/tqm src/baselines/umnlp src/baselines/caselink \
         src/baselines/graphrag_adapter tests/baselines
touch src/baselines/__init__.py src/baselines/common/__init__.py \
      src/baselines/bm25/__init__.py src/baselines/jnlp/__init__.py \
      src/baselines/tqm/__init__.py src/baselines/umnlp/__init__.py \
      src/baselines/caselink/__init__.py src/baselines/graphrag_adapter/__init__.py \
      tests/baselines/__init__.py
```

- [ ] **Step 2: Update `pyproject.toml` wheel packages**

Change:
```toml
packages = ["src/graphrag"]
```
To:
```toml
packages = ["src/graphrag", "src/baselines"]
```

- [ ] **Step 3: Commit**

---

### Task 2: BaselineModel ABC

**Files:** Create `src/baselines/common/base_model.py`

Abstract base class with `name()`, `train()`, `predict()`, and a default `predict_batch()`. See design spec Section 2 for interface. The `train()` accepts `corpus`, `train_queries`, `labels`, and optional `bm25_candidates`. The `predict()` returns `list[tuple[str, float]]` sorted descending, excluding the query document itself.

- [ ] **Step 1: Write base_model.py**
- [ ] **Step 2: Commit**

---

### Task 3: Data loader with chronological split

**Files:** Create `src/baselines/common/data_loader.py`, `tests/baselines/test_data_loader.py`

Loads ALL docs from both `task1_train_files_2026/` and `task1_test_files_2026/` (~9,556 total). Applies `graphrag.preprocess.preprocess()`. Splits the 2,001 labeled queries 80/20 by numeric ID (higher IDs = newer = validation). Returns a `Dataset` dataclass.

- [ ] **Step 1: Write test verifying structure (9000+ docs, 80/20 split, val IDs > train IDs)**
- [ ] **Step 2: Run test — expect FAIL**
- [ ] **Step 3: Write implementation importing from `graphrag.config` for paths**
- [ ] **Step 4: Run test — expect PASS**
- [ ] **Step 5: Commit**

---

### Task 4: Shared BM25 index wrapper

**Files:** Create `src/baselines/common/bm25_index.py`

Wraps `graphrag.bm25.BM25Index`. Provides `build_shared_bm25(corpus, top_k=200)` that returns `(bm25_index, candidates_cache)` where `candidates_cache` is `{doc_id: [(cand_id, score), ...]}` for ALL documents (not just queries). This avoids re-computing BM25 for each baseline.

- [ ] **Step 1: Write implementation**
- [ ] **Step 2: Commit**

---

### Task 5: Run harness

**Files:** Create `src/baselines/common/run_harness.py`

For each baseline: (1) call `train()`, (2) optimize threshold on training queries, (3) predict on val queries, (4) compute micro-F1. Outputs comparison table to console and saves to `output/baselines/comparison.csv` and `comparison.json`.

Reuses `graphrag.metrics.micro_f1`, `optimize_threshold`, `scores_to_predictions`.

- [ ] **Step 1: Write `assess_baseline()` for single model and `run_comparison()` for multiple**
- [ ] **Step 2: Commit**

---

### Task 6: Vanilla BM25 baseline (Baseline E)

**Files:** Create `src/baselines/bm25/model.py`, `tests/baselines/test_bm25_baseline.py`

~50 lines. Wraps `BM25Index.query()` + threshold optimization. No training step.

- [ ] **Step 1: Write test (interface check + mini corpus rank check)**
- [ ] **Step 2: Run test — expect FAIL**
- [ ] **Step 3: Write implementation**
- [ ] **Step 4: Run test — expect PASS**
- [ ] **Step 5: Commit**

---

### Task 7: CLI runners + end-to-end smoke test

**Files:** Create `src/baselines/run_baseline.py`, `src/baselines/run_all.py`

`run_baseline.py` takes a name argument (bm25, jnlp, tqm, umnlp, caselink). Uses a registry dict.
`run_all.py` imports all available baselines (try/except ImportError for missing ones), runs comparison.

- [ ] **Step 1: Write run_baseline.py with registry**
- [ ] **Step 2: Write run_all.py with graceful import fallbacks**
- [ ] **Step 3: Smoke test: `PYTHONPATH=src uv run python -m baselines.run_baseline bm25`**
- [ ] **Step 4: Commit**

**Checkpoint:** BM25 baseline runs end-to-end on 2026 dataset. Expected val F1 ~0.15-0.20.

---

## Chunk 2: JNLP 2025 Baseline (Tasks 8-10)

Reproduces 2025 Task 1 winner (Paper 29). Stages: BM25 retrieval -> feature extraction (BM25 full/paragraph, QLD, SAILER) -> LightGBM binary classifier.

### Task 8: SAILER encoder with embedding cache

**Files:** Create `src/baselines/jnlp/sailer_encoder.py`

SAILER (`CSHaitao/SAILER_en_finetune`) uses asymmetric encoder-decoder — NOT standard AutoModel. Try checkpoints in order: `SAILER_en_finetune` -> `SAILER_en` -> `nlpaueb/legal-bert-base-uncased`. CLS token embedding, L2-normalized. Disk cache per doc in `output/baselines/sailer_embeddings/`.

- [ ] **Step 1: Write SAILEREncoder class with `encode_corpus()` and `similarity()`**
- [ ] **Step 2: Commit**

### Task 9: JNLP feature extraction

**Files:** Create `src/baselines/jnlp/features.py`

8 features per (query, candidate) pair: `bm25_full`, `bm25_para_max` (max BM25 across query paragraphs), `qld_score` (Query Likelihood, Dirichlet mu=2000), `sailer_sim`, `bm25_rank`, `bm25_ratio`, `query_len`, `doc_len`.

QLD formula: `score = sum(log((tf + mu * P(t|C)) / (|d| + mu)))` for each query term.
`bm25_para_max`: split query into paragraphs via `extract_paragraphs()`, score each paragraph against candidate using `BM25Index._score_tokens()`, take max.

- [ ] **Step 1: Write `qld_score()`, `extract_jnlp_features()`, `build_corpus_freq()`**
- [ ] **Step 2: Commit**

### Task 10: JNLP model (LightGBM reranker)

**Files:** Create `src/baselines/jnlp/model.py`, `tests/baselines/test_jnlp.py`

LightGBM binary classifier. Params: `num_leaves=31, lr=0.05, feature_fraction=0.8, bagging_fraction=0.8, n_estimators=500`. Train on features from top-200 BM25 candidates of training queries. Register in `run_baseline.py`.

- [ ] **Step 1: Write test (interface check)**
- [ ] **Step 2: Write JNLPBaseline implementing BaselineModel**
- [ ] **Step 3: Register in run_baseline.py**
- [ ] **Step 4: Run test**
- [ ] **Step 5: Commit**

**Checkpoint:** `PYTHONPATH=src uv run python -m baselines.run_baseline jnlp`. Expected val F1 ~0.28-0.35. SAILER encoding takes ~30-60 min first run.

---

## Chunk 3: TQM 2024 Baseline (Tasks 11-12)

Reproduces 2024 Task 1 winner (Paper 26, arXiv 2404.00947). No public code — reverse-engineered from paper. Stages: lexical matching + semantic bi-encoder -> LambdaMART LTR fusion.

### Task 11: TQM bi-encoder (fine-tune + embed cache)

**Files:** Create `src/baselines/tqm/bi_encoder.py`

**Requires GPU.** Base model: `all-MiniLM-L6-v2`. Fine-tune with `MultipleNegativesRankingLoss` on (query, noticed, bm25_hard_negative) triplets. 3 epochs, batch_size=32, lr=2e-5. Save model to `output/baselines/models/tqm_biencoder/`. Cache all doc embeddings to `output/baselines/tqm_embeddings/`.

- [ ] **Step 1: Write TQMBiEncoder with `finetune()`, `encode_corpus()`, `similarity()`**
- [ ] **Step 2: Commit**

### Task 12: TQM features + LambdaMART model

**Files:** Create `src/baselines/tqm/features.py`, `src/baselines/tqm/model.py`, `tests/baselines/test_tqm.py`

10 features: `bm25_score`, `tfidf_cosine` (sklearn TfidfVectorizer, max_features=50000), `dense_cosine` (bi-encoder), `query_len`, `doc_len`, `len_ratio`, `paragraph_count`, `bm25_rank`, `year_proximity` (regex from header), `shared_top_k` (BM25 top-50 neighbor overlap).

LambdaMART via LightGBM: `objective="lambdarank"`, `metric="ndcg"`, `num_leaves=63`, `n_estimators=500`.

- [ ] **Step 1: Write TQMFeatureExtractor**
- [ ] **Step 2: Write TQMBaseline model**
- [ ] **Step 3: Test, register, commit**

**Checkpoint:** Run inside Docker. Expected val F1 ~0.30-0.40.

---

## Chunk 4: UMNLP 2024 Baseline (Tasks 13-14)

Reproduces 2024 Task 1 runner-up (Paper 27, GitHub `dc435/COLIEE_2024_Task1`). Stages: proposition extraction + judge matching + quotation detection -> FF neural network.

### Task 13: UMNLP proposition + judge features

**Files:** Create `src/baselines/umnlp/propositions.py`, `src/baselines/umnlp/judge_match.py`, `src/baselines/umnlp/features.py`

**Propositions:** Extract text windows (100 words) around `<FRAGMENT_SUPPRESSED>` markers. Encode with `all-MiniLM-L6-v2`. Max cosine similarity between query and candidate proposition embeddings.

**Judge matching:** Reuse `graphrag.extract_regex.extract_judges()` + `graphrag.normalize.normalize_judge()`. Binary match + IDF-weighted rarity score.

**Quotation:** 5-gram overlap fraction between query and candidate.

**Features:** `proposition_sim`, `judge_match`, `judge_rarity_score`, `quotation_overlap`, `para_max_sim`, `para_mean_sim`, `bm25_score`, `doc_len_ratio`.

- [ ] **Step 1: Write PropositionExtractor**
- [ ] **Step 2: Write JudgeMatcher**
- [ ] **Step 3: Write features.py with quotation_overlap and assemble function**
- [ ] **Step 4: Commit**

### Task 14: UMNLP FF NN model

**Files:** Create `src/baselines/umnlp/model.py`, `tests/baselines/test_umnlp.py`

PyTorch FF NN: Linear(8, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 64) -> ReLU -> Dropout(0.3) -> Linear(64, 1) -> Sigmoid. BCE loss, Adam lr=1e-3, batch_size=256, 20 epochs.

- [ ] **Step 1: Write PropositionNN module and UMNLPBaseline model**
- [ ] **Step 2: Test, register, commit**

**Checkpoint:** Run inside Docker. Expected val F1 ~0.30-0.40.

---

## Chunk 5: CaseLink 2025 Baseline (Tasks 15-17)

Reproduces 2025 Task 1 runner-up (Paper 30, arXiv 2505.20743, GitHub `yanran-tang/CaseLink`). GNN over heterogeneous case graph.

### Task 15: Add torch-geometric dependency

- [ ] **Step 1: `uv add torch-geometric`**
- [ ] **Step 2: Commit pyproject.toml + uv.lock**

### Task 16: CaseLink graph + node features

**Files:** Create `src/baselines/caselink/graph.py`, `src/baselines/caselink/node_features.py`

**Graph:** Heterogeneous graph with Case-Case edges (from labels), Case-Statute edges (reuse `extract_regex.extract_statutes()`), Statute-Statute edges (co-occurrence threshold >= 3). "Charge" from paper = "Statute" for our civil law corpus.

**Node features:** Cases encoded with `all-MiniLM-L6-v2`. Statutes = mean-pooled associated case embeddings. Cache to `output/baselines/caselink_embeddings/`.

- [ ] **Step 1: Write graph builder**
- [ ] **Step 2: Write node feature generator**
- [ ] **Step 3: Commit**

### Task 17: CaseLink GNN + model

**Files:** Create `src/baselines/caselink/gnn.py`, `src/baselines/caselink/model.py`, `tests/baselines/test_caselink.py`

**GNN:** 2-layer GraphSAGE (PyG `SAGEConv`), hidden_dim=256, dropout=0.3, L2-normalized output.

**Training:** InfoNCE contrastive loss on (query, noticed) positive pairs with in-batch negatives. Adam lr=1e-3, 100 epochs.

**Inference:** Cosine similarity of GNN-produced case embeddings. Top-200 ranked results.

- [ ] **Step 1: Write CaseLinkGNN module**
- [ ] **Step 2: Write CaseLinkBaseline model with train/predict**
- [ ] **Step 3: Test, register, commit**

**Checkpoint:** Run inside Docker. Expected val F1 ~0.25-0.30.

---

## Chunk 6: GraphRAG Adapter + Full Comparison (Tasks 18-19)

### Task 18: GraphRAG adapter

**Files:** Create `src/baselines/graphrag_adapter/model.py`

Thin wrapper calling `graphrag.bm25.BM25Index` and `graphrag.graphrag_lite.GraphRAGLite`. Loads pre-trained meta-learner fold models from `output/models_v2/meta_learner/` if available. Falls back to BM25+GraphRAG entity overlap fusion if meta-learner not trained yet.

- [ ] **Step 1: Write GraphRAGAdapter implementing BaselineModel**
- [ ] **Step 2: Register, commit**

### Task 19: Test fixtures + full comparison

**Files:** Create `tests/baselines/conftest.py`

- [ ] **Step 1: Write shared fixtures (mini_corpus, mini_labels)**
- [ ] **Step 2: Run all unit tests: `PYTHONPATH=src uv run pytest tests/baselines/ -v`**
- [ ] **Step 3: Run full comparison on host (BM25 + JNLP — no GPU needed)**
- [ ] **Step 4: Run full comparison in Docker (all baselines including GPU ones)**
- [ ] **Step 5: Final commit**

**Final checkpoint:** All baselines produce F1 scores on 2026 dataset:

```
Baseline                              F1      Prec    Recall
--------------------------------------------------------------
BM25 (vanilla)                       ~0.18   ~0.14   ~0.25
JNLP 2025 (BM25+SAILER+LightGBM)   ~0.30   ~0.28   ~0.33
TQM 2024 (LTR Fusion)               ~0.35   ~0.38   ~0.32
UMNLP 2024 (Propositions+NN)        ~0.33   ~0.35   ~0.31
CaseLink 2025 (GNN)                  ~0.27   ~0.26   ~0.28
GraphRAG Option C (ours)             ~0.??   ~0.??   ~0.??
```

---

## Execution Notes

### GPU Requirements
- **No GPU needed:** BM25, JNLP (SAILER works on CPU, just slower)
- **GPU recommended:** TQM (bi-encoder fine-tuning), UMNLP (NN training), CaseLink (GNN)
- Docker command: `docker exec -e PYTHONPATH=/workspace/coliee2026/src coliee_optionc python -m baselines.run_baseline <name>`

### Option C Pipeline Dependency
- The GraphRAG adapter (Task 18) needs Option C training to have completed
- All other baselines are fully independent

### Embedding Caches
- SAILER: `output/baselines/sailer_embeddings/` (~9,556 `.npy` files)
- TQM: `output/baselines/tqm_embeddings/`
- CaseLink: `output/baselines/caselink_embeddings/`
- First run encodes all docs; subsequent runs load from cache

### Parallelism
- Chunks 2-5 (JNLP, TQM, UMNLP, CaseLink) can be implemented in parallel by separate agents — they share only the Chunk 1 infrastructure
- Within each chunk, tasks are sequential

### Time Budget
- Chunk 1: ~1 hour (mostly BM25 candidate precomputation)
- Chunk 2 (JNLP): ~2 hours (SAILER encoding dominates)
- Chunk 3 (TQM): ~3 hours (bi-encoder fine-tuning on GPU)
- Chunk 4 (UMNLP): ~2 hours (proposition encoding)
- Chunk 5 (CaseLink): ~2 hours (GNN training on GPU)
- Chunk 6: ~30 min (adapter + comparison)
