# Option C Pipeline Enhancement Options

**Context:** Option C (Hybrid Multi-View) pipeline achieves CV F1=0.5924 (no-finetune, CPU-only). This beats the COLIEE 2024 winner (TQM F1=0.4432) by 34%. The main bottleneck is recall ceiling (~61% — only 5,027/8,251 gold positives appear in BM25 RRF top-200).

**Design principle:** Every option below targets a specific bottleneck in the current pipeline. None duplicate prior COLIEE approaches (BM25+BERT reranking, TF-IDF ensembles). Each is selected for compatibility with the existing 6-stage architecture.

---

## Tier 1: High Impact, Low Effort (days, not weeks)

### Option 1: Personalized PageRank on the Bipartite Entity Graph

**Stage affected:** 5 (GraphRAG Lite)
**Bottleneck addressed:** Graph features are currently binary community membership flags — they don't capture *distance* or *relevance propagation* through the graph.

**What:** Run Personalized PageRank (PPR) from each query node on the existing bipartite entity graph (cases ↔ statutes/judges/domains). The PPR score for each candidate becomes a new feature: how much "relevance mass" flows from the query to the candidate through shared legal structure.

**Why this is novel for COLIEE:** No prior COLIEE team has used PPR on entity graphs. HippoRAG (NeurIPS 2024) showed PPR on entity graphs improves multi-hop QA by ~20%. Legal case retrieval is inherently multi-hop (query → statute → related cases).

**Implementation:**
- `igraph` already has `personalized_pagerank()` — no new dependencies
- ~50-80 lines added to `graphrag_lite.py`
- Produces 1-3 new features per pair: `ppr_score`, optionally `ppr_rank`, `ppr_score_reverse` (candidate→query)
- Teleport vector: uniform over query's entity neighbors

**Expected impact:** +2-5% F1 based on HippoRAG results on comparable tasks. Graph features currently rank low in importance (community flags are coarse); PPR adds a continuous, distance-aware signal.

**References:**
- HippoRAG (Hu et al., NeurIPS 2024) — PPR on knowledge graphs for multi-hop retrieval
- KET-RAG (2025) — PageRank-based core chunk selection in RAG pipelines
- PathRAG (2025) — Flow-based path scoring through entity graphs

---

### Option 2: LambdaRank Objective for LightGBM

**Stage affected:** 6 (Meta-Learner)
**Bottleneck addressed:** Binary classification (`objective: binary`) optimizes per-pair accuracy, not ranking quality. The pipeline needs to rank candidates *within each query* — a ranking objective directly optimizes this.

**What:** Replace `objective: "binary"` with `objective: "lambdarank"` in LightGBM. LambdaRank directly optimizes NDCG (a ranking metric), which aligns better with the retrieval task than binary cross-entropy.

**Why this is novel for COLIEE:** Prior COLIEE teams using LightGBM/XGBoost all use binary classification or regression objectives. No prior work uses learning-to-rank objectives for the fusion stage.

**Implementation:**
- Change 3 lines in `config.py`:
  ```python
  LGBM_PARAMS = {
      "objective": "lambdarank",
      "metric": "ndcg",
      "eval_at": [5, 10, 50],
      # rest stays the same
  }
  ```
- Requires `group` parameter in LightGBM training (number of candidates per query) — small change to `train_meta_learner()`
- Threshold optimization still applies (convert ranking scores to binary decisions)

**Expected impact:** +1-4% F1. Learning-to-rank objectives consistently outperform binary classification for retrieval fusion in IR benchmarks. The improvement is larger when feature quality varies across queries (which it does — some queries have strong BM25 signal, others don't).

**Risk:** LambdaRank optimizes NDCG, not F1. May need to tune `label_gain` parameter or try `rank_xendcg` objective. Easy to A/B test.

**References:**
- Burges (2010) — LambdaRank/LambdaMART original paper
- LightGBM docs — native lambdarank support with query groups

---

### Option 3: Convex Combination Replacing RRF

**Stage affected:** 2 (BM25 retrieval)
**Bottleneck addressed:** RRF is a rank-based fusion that discards score magnitudes. When one BM25 view (full-doc vs. context-window) has a much stronger signal for a particular query, RRF averages away that advantage.

**What:** Replace RRF fusion with a learned convex combination: `score = α · score_fulldoc + (1-α) · score_context`. Learn α by optimizing recall@200 on training data (grid search over [0, 1]).

**Why this is novel:** RRF is the standard in every multi-view retrieval system. Recent work (Goswami et al., SIGIR 2025) shows convex combinations consistently outperform RRF when score distributions are heterogeneous — which they are here (full-doc BM25 scores have different scale than per-context scores).

**Implementation:**
- ~30 lines in `bm25.py`: min-max normalize each score distribution, grid search α on training recall@K
- No new dependencies
- Can also extend to per-query α (train a small model to predict optimal α per query)

**Expected impact:** +1-3% recall@200 (which directly lifts the recall ceiling). Even small recall improvements are amplified through the pipeline.

**References:**
- Goswami et al. (SIGIR 2025) — Convex combination vs. RRF for heterogeneous score fusion

---

### Option 4: Passage-Level Aggregation Features

**Stage affected:** 6 (Meta-Learner) — new features
**Bottleneck addressed:** Current features treat each document as a monolithic unit. Legal cases are long (avg ~3,000 words) with relevant information concentrated in specific passages. Passage-level signals are lost in document-level averaging.

**What:** For each (query, candidate) pair, compute bi-encoder similarity at the paragraph level, then aggregate:
- `max_passage_sim` (MaxP) — strongest paragraph match
- `avg_top3_passage_sim` (AvgTop3) — average of top-3 paragraph matches
- `n_passages_above_threshold` — count of paragraphs above a similarity threshold

**Why this is novel:** Prior COLIEE teams encode full documents or use paragraph-level only for Task 2. Using passage-level *aggregation features* in the meta-learner is novel — it captures both the peak relevance and the breadth of relevance across a candidate document.

**Implementation:**
- Split candidate docs into paragraphs (already have `\n\n` paragraph boundaries)
- Encode paragraphs with the bi-encoder (batch inference)
- Compute similarities and aggregate — ~60 lines in a new section of `run_pipeline_v2.py`
- Adds 3 features to FEATURE_COLS

**Expected impact:** +2-4% F1. MaxP is particularly powerful for legal retrieval — a case may cite another for a single paragraph's holding, making the full-document similarity misleadingly low.

**Risk:** Increases bi-encoder inference time (more passages to encode). Can be mitigated by only encoding paragraphs for candidates in the RRF top-200.

**References:**
- Dai & Callan (2019) — MaxP/FirstP passage aggregation for document retrieval
- PARADE (Li et al., 2020) — learned passage aggregation

---

## Tier 2: High Impact, Moderate Effort (1-2 weeks)

### Option 5: SPLADE-v3 Learned Sparse Retrieval

**Stage affected:** 2 (replaces/supplements BM25)
**Bottleneck addressed:** BM25 relies on exact lexical matching. Legal language uses synonyms, paraphrases, and domain-specific terminology that BM25 misses. The recall ceiling (61%) is largely a BM25 limitation.

**What:** Replace or supplement BM25 with SPLADE-v3, a learned sparse retrieval model that expands queries with semantically related terms while maintaining an inverted index for fast retrieval.

**Why this is novel:** No COLIEE team has used learned sparse retrieval. SPLADE-v3 bridges the gap between BM25 (fast, interpretable) and dense retrieval (semantic matching) — it does term expansion (e.g., "deportation" → also matches "removal order", "inadmissibility") while keeping the efficiency of inverted index lookup.

**Implementation:**
- Use `naver/splade-v3` from HuggingFace (open-source, MIT license)
- Encode corpus once, build sparse index
- Fuse SPLADE scores into existing RRF or convex combination
- ~150 lines for encoding + retrieval

**Expected impact:** +5-10% recall@200. SPLADE-v3 typically improves recall by 15-30% over BM25 on BEIR benchmarks. Even conservative estimates would push the recall ceiling from 61% to ~70%.

**Risk:** Requires GPU for corpus encoding (one-time). Model is 110M params (manageable on DGX Spark).

**References:**
- Lassance et al. (2024) — SPLADE-v3: learned sparse retrieval
- Formal et al. (2022) — SPLADE++ distillation

---

### Option 6: BGE-M3 Multi-Vector Retrieval

**Stage affected:** 3 (replaces bi-encoder)
**Bottleneck addressed:** Current bi-encoder (BGE-large) produces a single vector per document. This loses fine-grained token-level matching information that's critical for legal text (specific statutory references, case names, legal tests).

**What:** Replace BGE-large with BGE-M3, which produces three representations simultaneously:
1. Dense embedding (like current bi-encoder)
2. Sparse embedding (like SPLADE — term expansion)
3. ColBERT-style multi-vector embeddings (token-level late interaction)

The model outputs all three in a single forward pass. Each becomes a feature (or set of features) for the meta-learner.

**Why this is novel:** BGE-M3 is the state-of-the-art multi-vector model (released 2024). No COLIEE team has used multi-vector retrieval. The ColBERT component captures exact token-level matches (statute numbers, judge names) that single-vector models compress away.

**Implementation:**
- Use `BAAI/bge-m3` from HuggingFace (open-source)
- Single forward pass → 3 score types per pair
- Can still apply LoRA fine-tuning
- ~200 lines to replace current bi-encoder

**Expected impact:** +3-6% F1. BGE-M3 consistently outperforms single-vector models on legal/long-document benchmarks. The sparse component provides BM25-like recall with semantic understanding.

**Risk:** Larger model (568M params). Inference is ~3x slower than BGE-large. ColBERT multi-vector storage requires more disk.

**References:**
- Chen et al. (2024) — BGE-M3: multi-lingual, multi-functionality, multi-granularity
- Khattab & Zaharia (2020) — ColBERT: efficient late interaction

---

### Option 7: Cross-Encoder Distillation to Bi-Encoder

**Stage affected:** 3 (bi-encoder training)
**Bottleneck addressed:** Current bi-encoder training uses InfoNCE loss with BM25-mined hard negatives. The cross-encoder (stage 4) has much better discrimination ability — distilling its knowledge into the bi-encoder improves retrieval without slowing inference.

**What:** After training the cross-encoder (stage 4), use its scores as soft labels to fine-tune the bi-encoder:
```
loss = KL(bi-encoder_distribution || cross-encoder_distribution)
```

This is a two-pass training: (1) train bi-encoder with InfoNCE, (2) refine with cross-encoder distillation.

**Why this is novel:** Standard in general IR (used by Sentence-Transformers), but no COLIEE team has applied cross-encoder → bi-encoder distillation. It's particularly powerful here because the cross-encoder sees (query, candidate) jointly and can capture legal reasoning patterns that the bi-encoder's independent encoding misses.

**Implementation:**
- ~80 lines added to `finetune_biencoder.py`
- Use cross-encoder scores for top-200 candidates as soft labels
- MarginMSE loss or KL divergence
- Second training pass after cross-encoder is trained

**Expected impact:** +2-4% recall@200. The bi-encoder becomes a better first-stage retriever, pushing more gold positives into the top-200.

**References:**
- Hofstätter et al. (2021) — MarginMSE distillation for dense retrieval
- Sentence-Transformers — cross-encoder distillation training guide

---

### Option 8: Continued Pre-Training (DAPT) on Legal Corpus

**Stage affected:** 3+4 (bi-encoder and cross-encoder base models)
**Bottleneck addressed:** BGE-large and DeBERTa-v3 are pre-trained on general text. Legal language has domain-specific terminology, citation patterns, and reasoning structures that these models haven't seen enough of.

**What:** Continue pre-training the base models on the COLIEE corpus (7,708 documents) with masked language modeling (MLM) before fine-tuning for retrieval. This adapts the model's language understanding to Federal Court of Canada case law.

**Why this is novel:** Only one prior COLIEE team (THUIR, 2023) attempted DAPT, and they used it on a much smaller corpus. The 2026 dataset is larger and DAPT benefits scale with corpus size.

**Implementation:**
- ~100 lines: load corpus, create MLM dataloader, train for 2-3 epochs
- Use the same LoRA adapters (efficient — no full model training)
- Train on both train and test documents (unsupervised — no labels needed)

**Expected impact:** +1-3% F1 across both bi-encoder and cross-encoder. DAPT typically shows 2-5% improvement on domain-specific tasks (Gururangan et al., ACL 2020).

**Risk:** Moderate compute cost (~4-8 GPU hours). Risk of catastrophic forgetting mitigated by LoRA (only adapters are updated, base model frozen).

**References:**
- Gururangan et al. (ACL 2020) — Don't Stop Pretraining: domain-adaptive pre-training
- THUIR (COLIEE 2023) — legal domain pre-training attempt

---

## Tier 3: Experimental / Longer-Term

### Option 9: Graph Contrastive Learning (LEXA-Style)

**Stage affected:** 5 (GraphRAG Lite)
**Bottleneck addressed:** Current graph features are hand-engineered (community flags, Jaccard similarity). A learned graph representation could capture complex structural patterns that manual features miss.

**What:** Train a graph contrastive learning model on the bipartite entity graph. Use graph augmentations (edge dropout, node masking) to learn case embeddings that capture structural similarity. The learned embeddings become features for the meta-learner.

**Reference result:** LEXA achieved F1=47.5 on COLIEE 2022 Task 1 using graph contrastive learning alone (no BM25, no neural reranking). Combined with the existing pipeline, this could be powerful.

**Implementation:** ~300 lines, requires PyTorch Geometric or DGL. Training is fast (~minutes on GPU for small graphs).

**Expected impact:** +3-6% F1 if the graph structure carries enough signal. High variance — depends on entity extraction quality.

---

### Option 10: Synthetic Query Generation (InPars-Style)

**Stage affected:** 3+4 (training data augmentation)
**Bottleneck addressed:** Limited training data (2,001 queries, 8,251 positive pairs). More diverse training examples improve model generalization.

**What:** Use an open-source LLM (e.g., the existing DeepSeek-R1:8b on Ollama) to generate synthetic queries for corpus documents. Given a case, generate a "query case" that would cite it. Use these synthetic pairs for pre-training the bi-encoder before fine-tuning on real labels.

**Implementation:** ~150 lines. Requires LLM inference for ~7,000 documents (~2-4 hours on Ollama).

**Expected impact:** +1-3% F1. InPars showed consistent improvements on BEIR. Effect is larger when real training data is limited (which it is — 2,001 queries is modest).

---

### Option 11: Curriculum Learning for Hard Negatives

**Stage affected:** 3+4 (training)
**Bottleneck addressed:** Current hard negative mining uses a static set of BM25-top-K negatives. Some are too easy (trivially different), some are too hard (nearly positive). Curriculum learning schedules difficulty to improve training stability.

**What:** Train bi-encoder in three phases:
1. Random negatives (easy — learn basic relevance)
2. BM25 hard negatives (medium — learn beyond lexical matching)
3. Cross-encoder-scored hard negatives (hard — learn fine-grained discrimination)

**Implementation:** ~60 lines of training loop modification.

**Expected impact:** +1-2% on bi-encoder recall. Small but consistent improvement from better training dynamics.

---

## Sprint-Feasible Options (hours, not days)

These are smaller, high-certainty enhancements that can each be implemented and tested in under half a day. They stack on top of the pipeline with minimal risk.

### Option 12: GPU Fine-Tuning Activation

**Stage affected:** 3+4 (bi-encoder and cross-encoder)
**Bottleneck addressed:** The current F1=0.5924 is with **no fine-tuning at all**. The bi-encoder uses base BGE-large weights, and the cross-encoder is entirely skipped. Fine-tuning code already exists in `finetune_biencoder.py` and `finetune_crossencoder.py` — it just needs GPU time.

**What:** Run `uv run python -m graphrag.run_pipeline train` (without `--no-finetune`). The LoRA adapters train the bi-encoder with InfoNCE + hard negatives, and the cross-encoder with binary classification on RRF candidates.

**Implementation:**
- Zero new code — just remove the `--no-finetune` flag
- Requires GPU (GB10 should handle BGE-large + LoRA and DeBERTa-v3-large)
- Stages 1, 2, 5 use existing cache — only stages 3, 4, 6 rerun
- Estimated runtime: ~2-4h (bi-encoder encoding is the bottleneck)

**Expected impact:** +5-10% F1. This is the single largest expected improvement. Cross-encoder alone adds 2 powerful features that are currently all zeros. Bi-encoder adapts from general to legal domain similarity.

**Risk:** May need to clear stage3/stage5 cache. If GB10 memory is tight, reduce `BIENCODER_BATCH_SIZE` to 8.

---

### Option 13: BM25 Top-K Expansion

**Stage affected:** 2 (BM25)
**Bottleneck addressed:** Current recall ceiling is 61% (5,027/8,251 gold positives in RRF top-200). Simply widening the candidate pool catches more positives.

**What:** Increase `BM25_TOP_K` from 200 to 300 or 500. More candidates enter the pool → more gold positives → higher recall ceiling.

**Implementation:**
- Change 1 line in `config.py`: `BM25_TOP_K = 300`
- Must clear stage2 cache and re-run stages 2-6
- Also increase `BIENCODER_TOP_K` to match

**Expected impact:** +3-8% recall ceiling. BM25 recall@300 is typically 5-10% above recall@200 (diminishing returns, but still meaningful). Each additional gold positive that enters the pool translates to better meta-learner training.

**Trade-off:** More candidates = more meta-learner training pairs = slightly longer stage 6. Also more bi-encoder scoring (but these are dot products, cheap).

**Diagnostic step:** Before committing, measure recall@K for K in [200, 300, 500, 1000] on the cached BM25 results to find the sweet spot.

---

### Option 14: Score Distribution Features

**Stage affected:** 6 (Meta-Learner) — new features
**Bottleneck addressed:** Current features are absolute scores per pair. The meta-learner doesn't know where a candidate's score falls *relative to other candidates for the same query*. A score of 0.7 means different things if the query's best candidate scores 0.8 vs. 0.95.

**What:** Add query-relative score features for each signal:
- `bm25_rrf_rank_norm` — normalized rank within query (0=best, 1=worst)
- `bm25_rrf_score_gap` — score minus query mean
- `biencoder_score_gap` — same for bi-encoder
- `score_above_query_median` — binary: is this candidate above the median score?
- `top_score_ratio` — candidate score / max score for this query

**Implementation:**
- ~40 lines in `meta_learner.py`: compute per-query statistics in `assemble_features()` or as a post-processing step on the feature matrix
- Adds 4-6 features to FEATURE_COLS
- No cache invalidation needed (only stage 6 changes)

**Expected impact:** +1-3% F1. Relative rank/score features are standard in learning-to-rank but missing from the current feature set. They help the meta-learner adapt to queries with different score distributions.

---

### Option 15: Per-Query Adaptive Thresholding

**Stage affected:** 6 (Meta-Learner, post-prediction)
**Bottleneck addressed:** A single global threshold (0.690) is applied to all queries. But queries have varying numbers of gold positives (1-20+), and the meta-learner's score distribution varies per query. A query with concentrated scores needs a different threshold than one with spread-out scores.

**What:** Replace global thresholding with per-query adaptive thresholds:
1. **Score-gap method:** For each query, sort candidates by score. Predict positive for candidates above the largest score gap (natural cluster boundary).
2. **Top-K adaptive:** Predict positive for top-K candidates where K = round(mean_score / global_threshold * avg_positives_per_query).
3. **Hybrid:** Use global threshold as floor, but also include any candidate within X% of the top score.

**Implementation:**
- ~30 lines in `metrics.py` or `meta_learner.py`
- Try all 3 methods, pick best on CV
- No retraining needed — only changes prediction decoding

**Expected impact:** +1-2% F1. Most effective for queries at the extremes (very few or very many positives).

---

### Option 16: Multi-Seed LightGBM Ensemble

**Stage affected:** 6 (Meta-Learner)
**Bottleneck addressed:** Single LightGBM training has variance from random initialization. Different seeds produce slightly different decision boundaries.

**What:** Train 3-5 LightGBM models with different `RANDOM_SEED` values, average their predictions before thresholding. This is orthogonal to the existing 5-fold CV — each seed produces 5 folds, total 15-25 models in the final ensemble.

**Implementation:**
- ~20 lines wrapping `train_meta_learner()` in a seed loop
- Average predictions across all seed×fold models
- Re-optimize threshold on the averaged scores

**Expected impact:** +0.5-1.5% F1. Small but nearly free. Ensemble averaging smooths out noise in individual model predictions.

---

### Option 17: BM25 Parameter Tuning

**Stage affected:** 2 (BM25)
**Bottleneck addressed:** BM25 uses default parameters k1=1.5, b=0.75. These are tuned for general web documents, not legal case law. Legal documents have highly variable length and different term distribution characteristics.

**What:** Grid search k1 ∈ [0.5, 1.0, 1.5, 2.0, 2.5] and b ∈ [0.25, 0.5, 0.75, 1.0] on training recall@200.

**Implementation:**
- ~30 lines: loop over parameter grid, fit BM25Index, measure recall
- Legal docs are long → lower b (less length normalization) may help
- Legal queries have specific terminology → higher k1 (more term saturation) may help

**Expected impact:** +0.5-2% recall@200. Parameter tuning on in-domain data consistently helps.

**Risk:** Requires re-running stage 2 for each setting. But BM25 indexing is fast (~30s), so a 20-point grid takes ~10 min.

---

### Option 18: Top-1 Guarantee Heuristic

**Stage affected:** Post-prediction
**Bottleneck addressed:** Some queries may get zero predictions if no candidate exceeds the threshold. In COLIEE, every query has at least one noticed case. Returning zero predictions wastes recall.

**What:** For any query with zero predictions after thresholding, always include the top-1 candidate (highest meta-learner score). Optionally include top-K where K scales with the number of queries that have zero predictions.

**Implementation:**
- ~10 lines in `scores_to_predictions()` or as a post-processing step
- Guaranteed to improve recall (adds true positives for queries where the model was uncertain but correct in ranking)
- Cannot hurt: if the top-1 candidate is wrong, we only add 1 FP

**Expected impact:** +0.5-1% F1 depending on how many queries get zero predictions. Purely additive.

---

### Option 19: Negative Sampling Refinement

**Stage affected:** 6 (Meta-Learner, training)
**Bottleneck addressed:** Current random negative sampling (10:1 ratio) treats all negatives equally. But negatives near the decision boundary (hard negatives with high BM25/bi-encoder scores) are more informative than easy negatives.

**What:** Stratified negative sampling:
- 50% hard negatives (top RRF rank, not gold positive)
- 30% medium negatives (mid-range RRF rank)
- 20% random negatives (outside RRF pool or low rank)

This gives the meta-learner more practice on the cases it'll actually see during prediction (high-RRF-score candidates).

**Implementation:**
- ~30 lines modifying `build_feature_matrix()` in `meta_learner.py`
- Stratify by RRF rank before sampling

**Expected impact:** +0.5-2% F1. Hard negative mining is a well-established technique in metric learning.

---

### Option 20: Submission Ensemble (Multiple Runs)

**Stage affected:** Post-pipeline
**Bottleneck addressed:** Competition allows 3 runs. Using all 3 slots maximizes chances.

**What:** Submit 3 different pipeline configurations as 3 runs:
1. **Conservative (high precision):** Higher threshold, LambdaRank, all enhancements
2. **Balanced:** Optimized threshold, standard config
3. **Aggressive (high recall):** Lower threshold, expanded top-K, top-1 guarantee

Each run uses different hyperparameters but the same trained models. Diversifying submission strategies hedges against threshold miscalibration on the unseen test set.

**Implementation:**
- Run `predict` mode 3 times with different config overrides
- Produce 3 JSON submission files

**Expected impact:** The best of 3 diverse runs typically beats any single run by 1-3%.

---

## 3-Day Sprint Plan

### Day 1: Meta-Learner Quick Wins (CPU-only, no cache invalidation)

| Time | Task | Option | Touches |
|------|------|--------|---------|
| AM | LambdaRank objective | #2 | `config.py`, `meta_learner.py` |
| AM | Score distribution features | #14 | `meta_learner.py` |
| PM | Per-query adaptive threshold | #15 | `metrics.py` or `meta_learner.py` |
| PM | Multi-seed ensemble | #16 | `run_pipeline_v2.py` |
| PM | Top-1 guarantee heuristic | #18 | `metrics.py` |
| PM | Negative sampling refinement | #19 | `meta_learner.py` |
| EOD | Re-run stage 6, compare CV F1 for each change | — | — |

**Why this order:** All changes are in stage 6 only. Stages 1-5 stay cached. Each can be A/B tested independently in ~11 min (stage 6 runtime). By EOD we know which changes help and keep only those.

### Day 2: Retrieval Improvements (requires cache invalidation)

| Time | Task | Option | Touches |
|------|------|--------|---------|
| AM | BM25 recall@K diagnostic (measure ceiling at 200/300/500) | #13 | script |
| AM | BM25 parameter tuning (k1, b grid search) | #17 | `bm25.py` |
| AM | Convex combination vs RRF | #3 | `bm25.py` |
| PM | PPR features on bipartite graph | #1 | `graphrag_lite.py` |
| PM | Re-run stages 2+5+6 with best config | — | — |
| PM | GPU fine-tuning activation (kick off, runs overnight) | #12 | clear stage3/4 cache |

**Why this order:** Morning is diagnostic/tuning (fast). PPR is the meatiest coding task. GPU fine-tuning starts in the evening and runs overnight.

### Day 3: Final Assembly + Submission

| Time | Task | Option | Touches |
|------|------|--------|---------|
| AM | Integrate fine-tuned models (stages 3+4+6 rerun) | #12 | — |
| AM | Stack all confirmed improvements | — | — |
| PM | Generate 3 test predictions (conservative/balanced/aggressive) | #20 | `run_pipeline_v2.py` |
| PM | Format and submit | — | — |

---

## All Options Priority Table

| Priority | Option | Effort | Expected F1 Gain | Sprint Day |
|----------|--------|--------|-------------------|------------|
| **0** | **GPU fine-tuning (#12)** | 0 (code exists) | +5-10% | Day 2 (overnight) |
| 1 | PPR on entity graph (#1) | 2-3h | +2-5% | Day 2 |
| 2 | LambdaRank objective (#2) | 1h | +1-4% | Day 1 |
| 3 | Score distribution features (#14) | 1-2h | +1-3% | Day 1 |
| 4 | BM25 top-K expansion (#13) | 30min + rerun | +3-8% recall | Day 2 |
| 5 | Convex combination (#3) | 2h | +1-3% recall | Day 2 |
| 6 | Per-query adaptive threshold (#15) | 1-2h | +1-2% | Day 1 |
| 7 | BM25 parameter tuning (#17) | 1h | +0.5-2% recall | Day 2 |
| 8 | Top-1 guarantee heuristic (#18) | 15min | +0.5-1% | Day 1 |
| 9 | Multi-seed ensemble (#16) | 30min | +0.5-1.5% | Day 1 |
| 10 | Negative sampling refinement (#19) | 1h | +0.5-2% | Day 1 |
| 11 | Submission ensemble (#20) | 1h | +1-3% (hedge) | Day 3 |
| — | Passage-level features (#4) | 3-4h | +2-4% | If time allows |
| — | SPLADE-v3 (#5) | 3-4 days | +5-10% recall | Post-sprint |
| — | BGE-M3 (#6) | 3-5 days | +3-6% | Post-sprint |
| — | Cross-encoder distillation (#7) | 2-3 days | +2-4% recall | Post-sprint |
| — | DAPT (#8) | 2-3 days | +1-3% | Post-sprint |
| — | Graph contrastive (#9) | 3+ days | +3-6% | Post-sprint |
| — | Synthetic queries (#10) | 2-3 days | +1-3% | Post-sprint |
| — | Curriculum learning (#11) | 1-2 days | +1-2% | Post-sprint |

**Estimated combined gain (sprint-feasible, stacking):** +10-20% F1 over current 0.5924, targeting **0.65-0.75 F1** on CV. The biggest single contributor will be GPU fine-tuning.

---

## What We're NOT Doing (and Why)

| Approach | Why skip |
|----------|----------|
| BM25 + BERT reranking | Already done by every COLIEE top team. Not novel. |
| TF-IDF ensembles | Our baselines.py covers this. Ceiling ~35% F1. |
| Closed-source LLM reranking | Competition rules: open-source only. |
| Full GNN on citation graph | Citation links are suppressed in queries. Graph is bipartite entities only. |
| Query expansion with LLM | Tested by prior teams, marginal gains. SPLADE does this better. |
| Dense-only retrieval | Legal text requires exact matching (statute numbers). Hybrid is essential. |
