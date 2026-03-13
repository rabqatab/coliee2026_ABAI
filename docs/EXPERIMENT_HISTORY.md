# COLIEE 2026 Task 1 — Experiment History & Results

> **Last updated:** 2026-03-13
> **Purpose:** Complete record of all experimental runs, baselines, and their results.

---

## Competition Benchmarks (External Reference)

### COLIEE 2024 Task 1 Winners
| Rank | Team | Method | F1 |
|------|------|--------|----|
| 1 | TQM | LTR fusion (BM25 + neural) | **0.4432** |
| 2 | UMNLP | Propositions + neural network | 0.4134 |
| 3 | YR | — | 0.3605 |
| 4 | JNLP | BM25 + SAILER + LightGBM | 0.3246 |

### COLIEE 2025 Task 1 Winners
| Rank | Team | Method | F1 |
|------|------|--------|----|
| 1 | JNLP | BM25 + SAILER + LightGBM | **0.3353** |
| 2 | UQLegalAI (CaseLink) | GNN-based | 0.2962 |
| 3 | AIIR Lab | — | 0.2171 |

> 2025 scores dropped significantly due to harder/larger test set.

Source: `docs/COLIEE_2024_RESULTS.md`, `docs/COLIEE_2025_RESULTS.md`

---

## A. Simple Baselines (`output/baseline_results.json`)

**Date:** ~2026-03-11
**Corpus:** Train-only (7,708 docs), 5-fold CV

### Task 1
| Method | CV F1 | Precision | Recall | Threshold | Time |
|--------|-------|-----------|--------|-----------|------|
| BM25-only | 0.0230 | 0.0117 | 0.5685 | 0.01 | 477s |
| TF-IDF Cosine | 0.0842 | 0.0728 | 0.0999 | 0.35 | 29s |
| **BM25 + Lexical → LightGBM** | **0.3489** | 0.3007 | 0.4312 | 0.168 | 622s |

The BM25+Lexical→LightGBM is the **reference baseline** (6 lexical features, no neural components).

### Task 2
| Method | F1 | Precision | Recall |
|--------|-----|-----------|--------|
| BM25-only | 0.0732 | 0.0380 | 0.9966 |
| TF-IDF Cosine | 0.4351 | 0.5219 | 0.3731 |

---

## B. Prior Winner Reproduction (`output/baselines/comparison.json`)

**Date:** 2026-03-12
**Setup:** 2026 dataset, 80/20 train/val split (not CV), GPU Docker

| System | Val F1 | Train F1 | Notes |
|--------|--------|----------|-------|
| BM25 (vanilla) | 0.0216 | 0.0213 | — |
| JNLP 2025 (BM25+SAILER+LightGBM) | 0.0358 | 0.3395 | Massive overfit |
| TQM 2024 (LTR Fusion) | 0.0271 | 0.2776 | Massive overfit |
| UMNLP 2024 (Propositions+NN) | 0.0244 | 0.0259 | Underfit |
| CaseLink 2025 (GNN) | 0.0202 | 0.0331 | Underfit |
| GraphRAG Option C (early) | 0.0216 | 0.0213 | Just BM25, pre-meta-learner |

> **⚠️ CAVEAT:** These results used an 80/20 split (not 5-fold CV) and all systems showed massive train-to-val degradation. These numbers are not directly comparable to the CV results below. The 2026 dataset appears significantly harder.

---

## C. Option C Pipeline Runs (Chronological)

All runs use the Option C architecture:
**Citation Context → BM25 RRF → Bi-encoder → Cross-encoder → GraphRAG Lite → LightGBM meta-learner**

### Run 1: First full run — CRASHED
- **Date:** 2026-03-11 22:32 – 2026-03-12 00:54
- **Log:** `output/pipeline_v2_noft.log`
- **Config:** No-finetune CPU, no caching, 7,708 docs
- **Stages 1-5 completed:** BM25 1286s, bi-encoder 7114s (CPU), GraphRAG 87s
- **Stage 6: CRASHED** — `PermissionError` on `output/models/meta_learner` directory
- **Features:** 17 (no lexical features)
- **Result:** No F1 produced

### Run 2: First successful — severely underfit
- **Date:** 2026-03-12 06:36 – 08:58
- **Log:** `output/pipeline_v2_noft_run2.log`
- **Config:** No-finetune CPU, caching enabled, 7,708 docs
- **Features:** 17 (no lexical, no gold injection, no neg sampling)
- **Issue:** LightGBM early-stopped at **1 tree per fold** (1:78 neg:pos ratio too high)
- **Result:** CV F1=0.0912, P=0.1079, R=0.0790
- **Duration:** 141.6 min

### Run 3: Added lexical features — still underfit
- **Date:** 2026-03-12 10:01 – 10:09
- **Log:** `output/pipeline_v2_lexical_fix.log`
- **Config:** +5 lexical features (tfidf_cosine, jaccard, shared_bigrams, length_ratio, shared_legal_terms) = 22 total
- **Issue:** Still no gold injection or neg sampling → LightGBM 1 tree
- **Result:** CV F1=0.2484, P=0.2593, R=0.2383
- **Duration:** 8 min (stages cached)

### Run 4: Gold injection + neg sampling — big improvement
- **Date:** 2026-03-12 10:23 – 10:33
- **Log:** `output/pipeline_v2_lexical_v2.log`
- **Config:** 22 features + 3,224 gold positives added + negative sampling (1:8 ratio, cap 50)
- **LightGBM:** Now training properly (~600-700 trees/fold)
- **Sampled OOF:** F1=0.6635, P=0.7207, R=0.6147 (t=0.290)
- **Full pool re-opt:** F1=0.2319 (threshold too low for real distribution)
- **Final reported:** CV F1=0.4475, P=0.3308, R=0.6912
- **Duration:** 10.5 min

### Run 5: Threshold re-optimization attempt — same issue
- **Date:** 2026-03-12 10:40 – 10:51
- **Log:** `output/pipeline_v2_lexical_v3.log`
- **Config:** Same as Run 4, threshold re-opt on full pool
- **Result:** CV F1=0.2319 (re-opt logic not yet correct)
- **Duration:** 11 min

### Run 6: Correct threshold re-optimization — BEST CPU RESULT ★
- **Date:** 2026-03-12 10:59 – 11:11
- **Log:** `output/pipeline_v2_lexical_v4.log`
- **Config:** 22 features + gold positives + neg sampling (1:8) + **corrected** full-pool threshold re-optimization
- **Sampled OOF:** F1=0.6642, P=0.7437, R=0.6000 (t=0.310)
- **Full pool re-optimized:** **F1=0.5924, P=0.8530, R=0.4538 (t=0.690)**
- **Top features (gain):** bm25_rrf_score (289K), biencoder_score (41K), tfidf_cosine (32K), biencoder_rank (28K), shared_bigrams (26K)
- **Duration:** 11.4 min
- **Saved to:** `output/models_v2/meta_learner/config.json`

---

## D. GPU Fine-tuning Attempts

### GPU Run 1: Original standalone script — bi-encoder OK, cross-encoder crashed
- **Date:** 2026-03-11 13:00 – 2026-03-12 08:14+
- **Log:** `output/training.log`
- **Script:** `scripts/train_pipeline.py` (old standalone version, not run_pipeline_v2)
- **Corpus:** 9,556 docs (train + test merged)
- **Bi-encoder:** BGE-large-en-v1.5 + LoRA, 57,757 triplets, 3 epochs, 10,830 steps. **Completed in ~19 hours.** Saved to `output/models/biencoder/final/`
- **Cross-encoder:** Started DeBERTa-v3-large training, then **process died silently** (~08:14 UTC). No traceback, no OOM. Empty `output/models/crossencoder/` dir.
- **BM25-only baseline (9,556 docs):** F1=0.0792, P=0.0572, R=0.1282 (t=0.040)

### GPU Run 2: Rewritten script — killed (was retraining bi-encoder)
- **Date:** 2026-03-13 00:20 – 00:30
- **Log:** `output/training_gpu.log` (overwritten)
- **Script:** `scripts/train_pipeline.py` → `run_pipeline_v2.py` (new wrapper)
- **Issue:** `stage3_biencoder(train=True)` started retraining from scratch despite saved model. Manually killed and fixed the code to check model existence first.

### GPU Run 3: tiktoken/sentencepiece crash
- **Date:** 2026-03-13 01:16 – 01:22
- **Stage 3:** Loaded saved bi-encoder, encoded 9,556 docs in 306s (5 min GPU vs ~2h CPU)
- **Stage 4:** CRASHED — `ValueError: tiktoken is required` then `sentencepiece` also missing
- **Fix:** `pip install tiktoken sentencepiece` in Docker

### GPU Run 4: Currently running ★
- **Date:** 2026-03-13 01:28+
- **Stages 1-3:** Loaded from cache (instant)
- **Stage 4:** Cross-encoder training started. 8,251 pos + 33,004 neg pairs, 3 epochs, 2,579 steps/epoch
- **Status:** IN PROGRESS — cross-encoder training (~2-4 hours expected)
- **Remaining:** Stage 5 (GraphRAG ~5 min) → Stage 6 (meta-learner ~15 min)

---

## E. Summary: All Results Ranked by F1

| Rank | Run | Date | Features | Neural Fine-tuning | Key Fixes | CV F1 | P | R |
|------|-----|------|----------|-------------------|-----------|-------|---|---|
| **1** | **Option C v4 (CPU)** | 03-12 | 22 | None (base models) | Gold inject + neg sample + threshold re-opt | **0.5924** | 0.8530 | 0.4538 |
| 2 | Option C v2 (CPU) | 03-12 | 22 | None | Gold inject + neg sample | 0.4475 | 0.3308 | 0.6912 |
| 3 | BM25+Lexical baseline | 03-11 | 6 | None | — | 0.3489 | 0.3007 | 0.4312 |
| 4 | Option C v3 (CPU) | 03-12 | 22 | None | Threshold re-opt (buggy) | 0.2319 | 0.2245 | 0.2398 |
| 5 | Option C lexical fix | 03-12 | 22 | None | +lexical features | 0.2484 | 0.2593 | 0.2383 |
| 6 | Option C run 2 (CPU) | 03-12 | 17 | None | — | 0.0912 | 0.1079 | 0.0790 |
| 7 | TF-IDF cosine | 03-11 | 1 | None | — | 0.0842 | 0.0728 | 0.0999 |
| 8 | BM25-only (9556 docs) | 03-11 | 0 | None | — | 0.0792 | 0.0572 | 0.1282 |
| 9 | BM25-only (7708 docs) | 03-11 | 0 | None | — | 0.0230 | 0.0117 | 0.5685 |
| **?** | **Option C GPU (running)** | 03-13 | 22+ | Bi-enc LoRA + Cross-enc | All fixes + fine-tuned models | **TBD** | — | — |

---

## F. Key Lessons Learned (Chronological)

1. **17 features without neg sampling = useless** (Run 2, F1=0.09). LightGBM early-stops at 1 tree with 1:78 pos:neg ratio.

2. **Lexical features matter** (Run 3, F1=0.25). Adding tfidf_cosine, jaccard, shared_bigrams, length_ratio, shared_legal_terms gave +0.16 F1 even without other fixes.

3. **Gold positive injection is critical** (Run 4, F1=0.45). 3,224 of 8,251 gold positives (39%) fall outside BM25 top-200. Without adding them, the meta-learner never sees them as positives.

4. **Negative subsampling is critical** (Run 4, F1=0.45). Capping at 10x ratio / 50 per query lets LightGBM train ~600 trees instead of 1.

5. **Threshold re-optimization on full pool is critical** (Run 6, F1=0.59). Sampled OOF has different pos/neg ratio than the real candidate pool. The threshold from OOF (0.31) is too low; re-optimizing on the full pool gives 0.69.

6. **Recall ceiling at 61%** — only 5,027/8,251 gold positives in BM25 RRF top-200. The remaining 39% are injected for training but can't be predicted at inference without expanding retrieval.

7. **Cross-encoder has been the bottleneck** — crashed 3 times across different runs (silent death, tiktoken missing, sentencepiece missing). Currently on 4th attempt.

---

## G. Enhancement Flags (Not Yet Tested)

These are implemented in code but all set to `False`/`1` (disabled) in `src/graphrag/config.py`:

| Flag | Description | Expected Impact |
|------|-------------|-----------------|
| `USE_LAMBDARANK` | Ranking objective instead of binary | May help ranking quality |
| `USE_PPR_FEATURES` | Personalized PageRank on entity graph | +2 features |
| `USE_CONVEX_FUSION` | Learned weighted fusion replacing RRF | Better score combination |
| `USE_STRATIFIED_NEGATIVES` | 50% hard / 30% medium / 20% easy | Better negative diversity |
| `MULTI_SEED_RUNS` | Average predictions across multiple seeds | Reduce variance |

Each can be A/B tested in ~11 min by toggling flags and re-running stage 6 only (stages 1-5 cached).

See `docs/plans/2026-03-12-enhancement-options.md` for full sprint plan with 20 options.
