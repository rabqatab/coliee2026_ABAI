# Novel Pipeline Enhancements Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 4 novel components to the Option C pipeline, each providing new meta-learner features, targeting CV F1 > 0.65 (up from 0.5924).

**Architecture:** Each chunk adds one independent capability that produces new features for the existing LightGBM meta-learner. The pipeline stages are additive -- each enhancement can be A/B tested by toggling a config flag. New stages slot between existing ones: BGE-M3 replaces Stage 3, GNN adds Stage 5.5, Reasoning Reranker adds Stage 4.5, Synthetic Data improves Stage 3/4 training data.

**Tech Stack:** Python 3.12, `uv run`, PyTorch, sentence-transformers, FlagEmbedding (BGE-M3), torch-geometric (GNN), transformers (reasoning reranker), LightGBM.

**Reference:** `docs/NOVEL_APPROACHES_SURVEY.md` -- literature survey with 99 papers.

---

## File Structure

### New Files

| File | Responsibility | Lines (est.) |
|------|----------------|-------------|
| `src/graphrag/multi_retrieval.py` | BGE-M3 dense+sparse+ColBERT retrieval | ~200 |
| `src/graphrag/gnn_reranker.py` | GAT-based score refinement on corpus graph | ~250 |
| `src/graphrag/reasoning_reranker.py` | 7B reasoning reranker with chain-of-thought | ~300 |
| `src/graphrag/synthetic_data.py` | LLM-based synthetic training pair generation | ~200 |
| `tests/test_multi_retrieval.py` | Unit tests for BGE-M3 retrieval | ~60 |
| `tests/test_gnn_reranker.py` | Unit tests for GNN reranker | ~60 |
| `tests/test_reasoning_reranker.py` | Unit tests for reasoning reranker | ~40 |
| `tests/test_synthetic_data.py` | Unit tests for synthetic data generation | ~40 |

### Modified Files

| File | Changes |
|------|---------|
| `src/graphrag/config.py` | Add config for BGE-M3, GNN, reasoning reranker, synthetic data |
| `src/graphrag/meta_learner.py` | Add new feature columns + extraction in `assemble_features()` |
| `src/graphrag/run_pipeline_v2.py` | Add new stages, pass features to meta-learner |

---

## Chunk 1: BGE-M3 Multi-Signal Retrieval

**Rationale:** Replace single bi-encoder (BGE-large, dense-only) with BGE-M3 which produces 3 retrieval signals (dense, sparse, ColBERT) from a single model inference. This gives the meta-learner 3 diversified features instead of 1, with no additional training required.

**Paper:** Chen et al. "M3-Embedding" (ACL Findings 2024, arXiv:2402.03216)

### Task 1: Add BGE-M3 config parameters

**Files:**
- Modify: `src/graphrag/config.py`

- [ ] **Step 1: Add BGE-M3 config to config.py**

Add after the bi-encoder config block (after line 49):

```python
# === BGE-M3 Multi-Signal Retrieval (Stage 3 replacement) ===
USE_BGE_M3 = True  # Toggle: True = BGE-M3 triple retrieval, False = original bi-encoder
BGE_M3_MODEL = "BAAI/bge-m3"
BGE_M3_BATCH_SIZE = 8  # Smaller batch - model is larger than BGE-large
BGE_M3_MAX_LENGTH = 8192  # BGE-M3 supports up to 8192 tokens
BGE_M3_WEIGHTS = {"dense": 0.4, "sparse": 0.3, "colbert": 0.3}  # Fusion weights
```

- [ ] **Step 2: Commit**

```bash
git add src/graphrag/config.py
git commit -m "feat: add BGE-M3 multi-signal retrieval config"
```

### Task 2: Create multi_retrieval.py module

**Files:**
- Create: `src/graphrag/multi_retrieval.py`
- Test: `tests/test_multi_retrieval.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_multi_retrieval.py
"""Tests for BGE-M3 multi-signal retrieval."""
import pytest
import numpy as np


def test_fuse_scores_combines_signals():
    """Verify weighted score fusion across dense, sparse, and ColBERT signals."""
    from graphrag.multi_retrieval import fuse_multi_scores

    dense = {"doc1": 0.8, "doc2": 0.5, "doc3": 0.3}
    sparse = {"doc1": 0.6, "doc2": 0.9, "doc3": 0.1}
    colbert = {"doc1": 0.7, "doc2": 0.4, "doc3": 0.5}

    weights = {"dense": 0.4, "sparse": 0.3, "colbert": 0.3}
    fused = fuse_multi_scores(dense, sparse, colbert, weights)

    # doc1: 0.4*0.8 + 0.3*0.6 + 0.3*0.7 = 0.32 + 0.18 + 0.21 = 0.71
    assert abs(fused["doc1"] - 0.71) < 1e-6
    # doc2: 0.4*0.5 + 0.3*0.9 + 0.3*0.4 = 0.20 + 0.27 + 0.12 = 0.59
    assert abs(fused["doc2"] - 0.59) < 1e-6


def test_fuse_scores_handles_missing_keys():
    """If a doc is in one signal but not another, treat missing as 0."""
    from graphrag.multi_retrieval import fuse_multi_scores

    dense = {"doc1": 0.8}
    sparse = {"doc1": 0.6, "doc2": 0.5}
    colbert = {"doc2": 0.4}

    weights = {"dense": 0.4, "sparse": 0.3, "colbert": 0.3}
    fused = fuse_multi_scores(dense, sparse, colbert, weights)

    assert "doc1" in fused
    assert "doc2" in fused
    assert abs(fused["doc1"] - (0.4 * 0.8 + 0.3 * 0.6)) < 1e-6


def test_extract_multi_features():
    """Verify per-pair feature extraction returns all 3 signals + fused."""
    from graphrag.multi_retrieval import extract_multi_features

    scores = {
        "q1": {
            "dense": {"doc1": 0.8, "doc2": 0.5},
            "sparse": {"doc1": 0.6, "doc2": 0.9},
            "colbert": {"doc1": 0.7, "doc2": 0.4},
            "fused": {"doc1": 0.71, "doc2": 0.59},
        }
    }

    feats = extract_multi_features("q1", "doc1", scores)
    assert "m3_dense_score" in feats
    assert "m3_sparse_score" in feats
    assert "m3_colbert_score" in feats
    assert "m3_fused_score" in feats
    assert abs(feats["m3_dense_score"] - 0.8) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_multi_retrieval.py -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

- [ ] **Step 3: Write the implementation**

```python
# src/graphrag/multi_retrieval.py
"""BGE-M3 multi-signal retrieval: dense + sparse + ColBERT from a single model.

Produces 4 features per (query, candidate) pair:
  - m3_dense_score: dense embedding cosine similarity
  - m3_sparse_score: learned sparse (lexical match) score
  - m3_colbert_score: late-interaction (token-level) MaxSim score
  - m3_fused_score: weighted combination of the three signals

Reference: Chen et al., "M3-Embedding" (ACL Findings 2024, arXiv:2402.03216)
"""
import logging
import time
from pathlib import Path

import numpy as np

from graphrag.config import (
    BGE_M3_MODEL,
    BGE_M3_BATCH_SIZE,
    BGE_M3_MAX_LENGTH,
    BGE_M3_WEIGHTS,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)


def fuse_multi_scores(
    dense: dict[str, float],
    sparse: dict[str, float],
    colbert: dict[str, float],
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Fuse dense, sparse, and ColBERT scores with weighted combination.

    Missing scores for a document in any signal are treated as 0.
    """
    if weights is None:
        weights = BGE_M3_WEIGHTS

    all_keys = set(dense) | set(sparse) | set(colbert)
    fused = {}
    for key in all_keys:
        fused[key] = (
            weights["dense"] * dense.get(key, 0.0)
            + weights["sparse"] * sparse.get(key, 0.0)
            + weights["colbert"] * colbert.get(key, 0.0)
        )
    return fused


def extract_multi_features(
    query_id: str,
    candidate_id: str,
    multi_scores: dict[str, dict[str, dict[str, float]]],
) -> dict[str, float]:
    """Extract per-pair features from pre-computed multi-signal scores.

    Args:
        query_id: Query document ID.
        candidate_id: Candidate document ID.
        multi_scores: {query_id: {signal_name: {candidate_id: score}}}.

    Returns:
        Dict with m3_dense_score, m3_sparse_score, m3_colbert_score, m3_fused_score.
    """
    q_scores = multi_scores.get(query_id, {})
    return {
        "m3_dense_score": q_scores.get("dense", {}).get(candidate_id, 0.0),
        "m3_sparse_score": q_scores.get("sparse", {}).get(candidate_id, 0.0),
        "m3_colbert_score": q_scores.get("colbert", {}).get(candidate_id, 0.0),
        "m3_fused_score": q_scores.get("fused", {}).get(candidate_id, 0.0),
    }


def encode_corpus_m3(
    corpus_texts: dict[str, str],
    model_name: str = BGE_M3_MODEL,
    batch_size: int = BGE_M3_BATCH_SIZE,
    max_length: int = BGE_M3_MAX_LENGTH,
) -> dict[str, dict]:
    """Encode entire corpus with BGE-M3, returning dense, sparse, ColBERT representations.

    Returns:
        {doc_id: {"dense": np.ndarray, "sparse": dict, "colbert": np.ndarray}}
    """
    from FlagEmbedding import BGEM3FlagModel

    logger.info("Loading BGE-M3 model: %s", model_name)
    model = BGEM3FlagModel(model_name, use_fp16=True)

    doc_ids = sorted(corpus_texts.keys())
    texts = [corpus_texts[d][:max_length * 4] for d in doc_ids]  # rough char limit

    logger.info("Encoding %d documents with BGE-M3 (batch_size=%d) ...", len(texts), batch_size)
    t0 = time.time()

    output = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )

    logger.info("BGE-M3 encoding complete in %.1f seconds", time.time() - t0)

    corpus_repr = {}
    for i, doc_id in enumerate(doc_ids):
        corpus_repr[doc_id] = {
            "dense": output["dense_vecs"][i],
            "sparse": output["lexical_weights"][i],
            "colbert": output["colbert_vecs"][i],
        }

    return corpus_repr


def score_candidates_m3(
    query_ids: list[str],
    candidate_lists: dict[str, list[str]],
    corpus_repr: dict[str, dict],
    weights: dict[str, float] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Score all (query, candidate) pairs using pre-computed BGE-M3 representations.

    Returns:
        {query_id: {"dense": {cid: score}, "sparse": {cid: score},
                     "colbert": {cid: score}, "fused": {cid: score}}}
    """
    if weights is None:
        weights = BGE_M3_WEIGHTS

    logger.info("Scoring candidates with BGE-M3 multi-signal ...")
    t0 = time.time()
    results = {}

    for qid in query_ids:
        if qid not in corpus_repr:
            continue

        q_repr = corpus_repr[qid]
        candidates = candidate_lists.get(qid, [])
        dense_scores = {}
        sparse_scores = {}
        colbert_scores = {}

        for cid in candidates:
            if cid not in corpus_repr:
                continue
            c_repr = corpus_repr[cid]

            # Dense: cosine similarity (vectors already normalized by BGE-M3)
            dense_scores[cid] = float(np.dot(q_repr["dense"], c_repr["dense"]))

            # Sparse: dot product of lexical weight dicts
            q_sparse = q_repr["sparse"]
            c_sparse = c_repr["sparse"]
            shared_tokens = set(q_sparse.keys()) & set(c_sparse.keys())
            sparse_scores[cid] = sum(
                q_sparse[t] * c_sparse[t] for t in shared_tokens
            )

            # ColBERT: MaxSim (max over candidate tokens per query token, then sum)
            q_colbert = q_repr["colbert"]  # (n_q_tokens, dim)
            c_colbert = c_repr["colbert"]  # (n_c_tokens, dim)
            if len(q_colbert) > 0 and len(c_colbert) > 0:
                sim_matrix = np.dot(q_colbert, c_colbert.T)
                max_sim = sim_matrix.max(axis=1)
                colbert_scores[cid] = float(max_sim.sum())
            else:
                colbert_scores[cid] = 0.0

        # Normalize scores to [0, 1] within each signal per query
        for scores_dict in [dense_scores, sparse_scores, colbert_scores]:
            if scores_dict:
                max_s = max(scores_dict.values())
                min_s = min(scores_dict.values())
                rng = max_s - min_s
                if rng > 0:
                    for k in scores_dict:
                        scores_dict[k] = (scores_dict[k] - min_s) / rng

        fused = fuse_multi_scores(dense_scores, sparse_scores, colbert_scores, weights)

        results[qid] = {
            "dense": dense_scores,
            "sparse": sparse_scores,
            "colbert": colbert_scores,
            "fused": fused,
        }

    logger.info("BGE-M3 scoring complete in %.1f seconds", time.time() - t0)
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_multi_retrieval.py -v`
Expected: PASS (all 3 tests, no model download needed for unit tests)

- [ ] **Step 5: Commit**

```bash
git add src/graphrag/multi_retrieval.py tests/test_multi_retrieval.py
git commit -m "feat: add BGE-M3 multi-signal retrieval module"
```

### Task 3: Integrate BGE-M3 into meta-learner features

**Files:**
- Modify: `src/graphrag/meta_learner.py:27-76` (feature columns)
- Modify: `src/graphrag/meta_learner.py:78-99` (assemble_features signature + body)

- [ ] **Step 1: Add M3 feature columns to meta_learner.py**

In `meta_learner.py`, after `"max_context_bm25"` (line 55), add:

```python
    # BGE-M3 multi-signal features (4) -- when USE_BGE_M3=True
    "m3_dense_score",
    "m3_sparse_score",
    "m3_colbert_score",
    "m3_fused_score",
```

- [ ] **Step 2: Add multi_scores parameter to assemble_features()**

Add `multi_scores` parameter to `assemble_features()` signature:

```python
def assemble_features(
    query_id: str,
    candidate_id: str,
    bm25_scores: dict[str, dict[str, float]],
    bm25_rrf_scores: dict[str, dict[str, float]],
    biencoder_scores: dict[str, dict[str, float]],
    crossencoder_scores: dict[str, dict[str, float]],
    graphrag_features: dict[tuple[str, str], dict[str, float]],
    context_features: dict[str, dict[str, dict[str, float]]],
    lexical_features: dict[tuple[str, str], dict[str, float]] | None = None,
    multi_scores: dict[str, dict[str, dict[str, float]]] | None = None,
) -> dict[str, float]:
```

At the end of the function body (before `return feats`), add:

```python
    # BGE-M3 multi-signal features
    if multi_scores is not None:
        from graphrag.multi_retrieval import extract_multi_features
        m3_feats = extract_multi_features(query_id, candidate_id, multi_scores)
        feats.update(m3_feats)
```

- [ ] **Step 3: Pass multi_scores through build_feature_matrix()**

Add `multi_scores` parameter to `build_feature_matrix()` signature and pass it to `assemble_features()`.

- [ ] **Step 4: Commit**

```bash
git add src/graphrag/meta_learner.py
git commit -m "feat: integrate BGE-M3 features into meta-learner"
```

### Task 4: Add BGE-M3 stage to pipeline orchestrator

**Files:**
- Modify: `src/graphrag/run_pipeline_v2.py`

- [ ] **Step 1: Add stage3_multi_retrieval function**

Add after `stage3_biencoder` (after line 297):

```python
def stage3_multi_retrieval(
    clean_corpus: dict[str, str],
    query_ids: list[str],
    rrf_results: dict[str, list[tuple[str, float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Stage 3 (alt): BGE-M3 multi-signal retrieval.

    Returns {query_id: {signal: {candidate_id: score}}} for dense, sparse, colbert, fused.
    """
    from graphrag.multi_retrieval import encode_corpus_m3, score_candidates_m3

    logger.info("=== Stage 3: BGE-M3 Multi-Signal Retrieval ===")
    t0 = time.time()

    corpus_repr = encode_corpus_m3(clean_corpus)

    candidate_lists = {
        qid: [cid for cid, _ in rrf_results.get(qid, [])]
        for qid in query_ids
    }
    multi_scores = score_candidates_m3(query_ids, candidate_lists, corpus_repr)

    logger.info("Stage 3 (BGE-M3) complete in %.1f seconds", time.time() - t0)
    return multi_scores
```

- [ ] **Step 2: Wire into run_train_pipeline()**

In `run_train_pipeline()`, after the existing Stage 3 block, add:

```python
    # Stage 3 (alt): BGE-M3 multi-signal retrieval
    multi_scores = None
    if USE_BGE_M3:
        cached = _load_cache("stage3_m3") if use_cache else None
        if cached is not None:
            multi_scores = cached
        else:
            multi_scores = stage3_multi_retrieval(clean_corpus, query_ids, rrf_results)
            _save_cache("stage3_m3", multi_scores)
```

Pass `multi_scores` to `stage6_meta_learner()` and propagate to `build_feature_matrix()`.

- [ ] **Step 3: Add FlagEmbedding dependency**

Run: `uv add FlagEmbedding`

- [ ] **Step 4: Test end-to-end**

Run: `uv run python -c "from graphrag.multi_retrieval import encode_corpus_m3; print('OK')"`
Expected: "OK" (validates import and BGE-M3 availability)

- [ ] **Step 5: Commit**

```bash
git add src/graphrag/run_pipeline_v2.py pyproject.toml uv.lock
git commit -m "feat: integrate BGE-M3 multi-signal retrieval into pipeline"
```

---

## Chunk 2: GNN Score Refinement

**Rationale:** Build a corpus graph from bi-encoder embeddings and entity overlap, then train a 2-layer GAT to refine retrieval scores. The GNN captures cross-document interactions -- if candidate A is relevant and candidate B is structurally similar to A, B should rank higher.

**Papers:** Di Francesco et al., "GNRR" (SIGIR 2024, arXiv:2406.11720); Wendlinger et al., "The Missing Link" (DEXA 2025, arXiv:2506.22165)

### Task 5: Add GNN config parameters

**Files:**
- Modify: `src/graphrag/config.py`

- [ ] **Step 1: Add GNN config**

```python
# === GNN Score Refinement (Stage 5.5) ===
USE_GNN_RERANKER = True
GNN_HIDDEN_DIM = 64
GNN_NUM_LAYERS = 2
GNN_HEADS = 4  # GAT attention heads
GNN_DROPOUT = 0.1
GNN_LR = 1e-3
GNN_EPOCHS = 50
GNN_K_NEIGHBORS = 8  # Semantic graph neighborhood size
GNN_ENTITY_WEIGHT = 0.3  # Weight for entity-overlap edges vs semantic edges
```

- [ ] **Step 2: Commit**

```bash
git add src/graphrag/config.py
git commit -m "feat: add GNN reranker config parameters"
```

### Task 6: Create gnn_reranker.py module

**Files:**
- Create: `src/graphrag/gnn_reranker.py`
- Test: `tests/test_gnn_reranker.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_gnn_reranker.py
"""Tests for GNN score refinement."""
import pytest
import numpy as np


def test_build_subgraph_extracts_candidates():
    """Subgraph should contain query + all candidates as nodes."""
    from graphrag.gnn_reranker import build_query_subgraph

    # Mock corpus graph adjacency (5 nodes)
    adj = {0: [1, 2], 1: [0, 3], 2: [0, 4], 3: [1], 4: [2]}
    node_features = np.random.randn(5, 16).astype(np.float32)
    query_idx = 0
    candidate_idxs = [1, 2, 3]

    subgraph = build_query_subgraph(adj, node_features, query_idx, candidate_idxs)

    assert subgraph["num_nodes"] >= 4  # query + 3 candidates
    assert subgraph["node_features"].shape[0] == subgraph["num_nodes"]
    assert subgraph["edge_index"].shape[0] == 2  # (2, num_edges)


def test_gnn_forward_produces_scores():
    """GNN model forward pass should produce one score per candidate."""
    import torch
    from graphrag.gnn_reranker import GNNReranker

    model = GNNReranker(input_dim=16, hidden_dim=32, num_layers=2, heads=2)
    # Simulate 5-node subgraph
    x = torch.randn(5, 16)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    candidate_mask = torch.tensor([False, True, True, True, False])

    scores = model(x, edge_index, candidate_mask)
    assert scores.shape == (3,)  # one score per candidate
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gnn_reranker.py -v`
Expected: FAIL with "ImportError"

- [ ] **Step 3: Write the implementation**

Create `src/graphrag/gnn_reranker.py` with:

- `GNNReranker(nn.Module)`: 2-layer GAT with score head. Forward takes node features, edge_index, candidate_mask and returns per-candidate scores.
- `build_corpus_graph()`: Builds doc-doc graph from embeddings (k-NN cosine) + entity overlap edges.
- `build_query_subgraph()`: Extracts query + candidates + 1-hop neighbors as a local subgraph.
- `build_node_features()`: Combines truncated embeddings (32-dim) + per-query retrieval scores (BM25, bi-encoder, cross-encoder) as node features.
- `train_gnn_reranker()`: Trains GNN on training queries with BCE loss. Each query creates one subgraph with binary labels (gold positive = 1).
- `gnn_rerank()`: Inference -- runs trained GNN on test query subgraphs, returns `{query_id: {candidate_id: gnn_score}}`.

Key implementation details:
- Input dim = 32 (truncated embedding) + 3 (retrieval scores) = 35
- GAT with 4 heads, 64 hidden dim, 2 layers, ELU activation
- BCE loss per query subgraph
- Save model to `models_v2/gnn_reranker/gnn_reranker.pt`

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_gnn_reranker.py -v`
Expected: PASS

- [ ] **Step 5: Add torch-geometric dependency**

Run: `uv add torch-geometric`

- [ ] **Step 6: Commit**

```bash
git add src/graphrag/gnn_reranker.py tests/test_gnn_reranker.py pyproject.toml uv.lock
git commit -m "feat: add GNN score refinement reranker"
```

### Task 7: Integrate GNN into pipeline + meta-learner

**Files:**
- Modify: `src/graphrag/meta_learner.py`
- Modify: `src/graphrag/run_pipeline_v2.py`

- [ ] **Step 1: Add GNN feature columns to meta_learner.py**

Add to `BASE_FEATURE_COLS`:

```python
    # GNN reranker features (2)
    "gnn_score",
    "gnn_rank",
```

Add GNN extraction to `assemble_features()`:

```python
    # GNN reranker features
    if gnn_scores is not None:
        gnn_q = gnn_scores.get(query_id, {})
        feats["gnn_score"] = gnn_q.get(candidate_id, 0.0)
```

- [ ] **Step 2: Add stage5_5_gnn() to run_pipeline_v2.py**

Add after stage5_graphrag with caching (`stage5_5.pkl`).
Requires: doc_ids, doc_embeddings (from Stage 3), rrf_results, all retrieval scores, graphrag_features, labels.
Returns: `{query_id: {candidate_id: gnn_score}}`

Wire into `run_train_pipeline()` and pass `gnn_scores` to meta-learner.

- [ ] **Step 3: Commit**

```bash
git add src/graphrag/run_pipeline_v2.py src/graphrag/meta_learner.py
git commit -m "feat: integrate GNN reranker into pipeline"
```

---

## Chunk 3: Reasoning Reranker

**Rationale:** Legal citation is inherently reasoning-intensive -- judges cite cases for specific legal reasoning connections, not just textual similarity. A reasoning reranker generates explicit chains-of-thought explaining WHY case A cites case B, producing reasoning-informed scores.

**Papers:** Weller et al., "Rank1" (arXiv:2502.18418); Zhang et al., "Rank-R1" (arXiv:2503.06034)

### Task 8: Add reasoning reranker config

**Files:**
- Modify: `src/graphrag/config.py`

- [ ] **Step 1: Add reasoning reranker config**

```python
# === Reasoning Reranker (Stage 4.5) ===
USE_REASONING_RERANKER = True
REASONING_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # Open-source, fits GB10
REASONING_MAX_LENGTH = 4096
REASONING_BATCH_SIZE = 1  # Sequential for reasoning chains
REASONING_TOP_K = 30  # Rerank top-30 from RRF (reasoning is slow)
REASONING_TEMPERATURE = 0.1
```

- [ ] **Step 2: Commit**

```bash
git add src/graphrag/config.py
git commit -m "feat: add reasoning reranker config"
```

### Task 9: Create reasoning_reranker.py module

**Files:**
- Create: `src/graphrag/reasoning_reranker.py`
- Test: `tests/test_reasoning_reranker.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_reasoning_reranker.py
"""Tests for reasoning reranker."""


def test_build_reasoning_prompt():
    """Verify prompt construction includes query and candidate text."""
    from graphrag.reasoning_reranker import build_reasoning_prompt

    prompt = build_reasoning_prompt(
        query_text="The applicant seeks judicial review...",
        candidate_text="In Smith v. Canada, the court held...",
    )
    assert "applicant seeks judicial review" in prompt
    assert "Smith v. Canada" in prompt
    assert "relevant" in prompt.lower() or "cite" in prompt.lower()


def test_parse_relevance_score():
    """Verify score extraction from model output."""
    from graphrag.reasoning_reranker import parse_relevance_score

    output1 = "<think>Both cases deal with immigration...</think>\nRelevant: Yes\nScore: 0.85"
    score1 = parse_relevance_score(output1)
    assert 0.8 <= score1 <= 0.9

    output2 = "<think>Different legal areas...</think>\nRelevant: No\nScore: 0.15"
    score2 = parse_relevance_score(output2)
    assert 0.1 <= score2 <= 0.2

    # Fallback for unparseable output
    score3 = parse_relevance_score("gibberish")
    assert score3 == 0.5  # neutral fallback
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_reasoning_reranker.py -v`
Expected: FAIL

- [ ] **Step 3: Write the implementation**

Create `src/graphrag/reasoning_reranker.py` with:

- `REASONING_PROMPT_TEMPLATE`: Structured prompt asking the LLM to reason step-by-step about citation relevance (shared area of law, similar legal issues, precedent relationship, shared statutes).
- `build_reasoning_prompt(query_text, candidate_text)`: Constructs prompt with truncated texts (500 words each).
- `parse_relevance_score(output)`: Extracts score from model output. Looks for `Score: X.XX` pattern, falls back to Yes=0.75, No=0.25, unparseable=0.5.
- `load_reasoning_model(model_name)`: Loads HuggingFace causal LM with fp16 and device_map="auto".
- `reasoning_rerank(model, tokenizer, query_text, candidates)`: Scores each candidate sequentially with reasoning. Returns `[(doc_id, score, reasoning_text)]`.
- `batch_reasoning_rerank(query_ids, corpus_texts, rrf_results)`: Full batch inference across all queries. Logs progress every 50 queries.

Key implementation details:
- Uses `tokenizer.apply_chat_template()` for proper instruction formatting
- Generates max 512 new tokens per pair (reasoning + verdict)
- Temperature 0.1 for near-deterministic output
- Sequential processing (batch_size=1) since each pair needs full reasoning

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_reasoning_reranker.py -v`
Expected: PASS (tests only test prompt building and score parsing, no model loading)

- [ ] **Step 5: Commit**

```bash
git add src/graphrag/reasoning_reranker.py tests/test_reasoning_reranker.py
git commit -m "feat: add reasoning reranker with chain-of-thought"
```

### Task 10: Integrate reasoning reranker into pipeline + meta-learner

**Files:**
- Modify: `src/graphrag/meta_learner.py`
- Modify: `src/graphrag/run_pipeline_v2.py`

- [ ] **Step 1: Add reasoning feature columns**

In `BASE_FEATURE_COLS`:

```python
    # Reasoning reranker features (2)
    "reasoning_score",
    "reasoning_rank",
```

- [ ] **Step 2: Add stage4_5_reasoning() to pipeline**

```python
def stage4_5_reasoning(
    raw_corpus: dict[str, str],
    query_ids: list[str],
    rrf_results: dict[str, list[tuple[str, float]]],
) -> dict[str, dict[str, float]]:
    """Stage 4.5: Reasoning reranker."""
    from graphrag.reasoning_reranker import batch_reasoning_rerank

    logger.info("=== Stage 4.5: Reasoning Reranker ===")
    return batch_reasoning_rerank(query_ids, raw_corpus, rrf_results)
```

Wire into `run_train_pipeline()` with caching (`stage4_5.pkl`) and pass `reasoning_scores` to meta-learner.

- [ ] **Step 3: Commit**

```bash
git add src/graphrag/run_pipeline_v2.py src/graphrag/meta_learner.py
git commit -m "feat: integrate reasoning reranker into pipeline"
```

---

## Chunk 4: Synthetic Data Augmentation

**Rationale:** Our training set has only 8,251 positive pairs. Generating additional synthetic pairs using an open-source LLM can dramatically improve bi-encoder and cross-encoder training. Uses the LEAD methodology adapted for Canadian case law.

**Papers:** Gao et al., "LEAD" (EMNLP 2024, arXiv:2410.06581); InPars+ (arXiv:2508.13930)

### Task 11: Create synthetic_data.py module

**Files:**
- Create: `src/graphrag/synthetic_data.py`
- Modify: `src/graphrag/config.py`
- Test: `tests/test_synthetic_data.py`

- [ ] **Step 1: Add synthetic data config**

```python
# === Synthetic Data Augmentation ===
USE_SYNTHETIC_DATA = False  # Disabled by default -- enable when ready
SYNTHETIC_LLM_MODEL = "deepseek-r1:8b"  # Via Ollama (already running)
SYNTHETIC_N_PAIRS = 20000  # Target number of synthetic pairs
SYNTHETIC_MAX_WORDS = 300  # Max words per extracted summary
```

- [ ] **Step 2: Write the test**

```python
# tests/test_synthetic_data.py
"""Tests for synthetic data generation."""


def test_extract_key_facts_prompt():
    """Verify prompt construction for key fact extraction."""
    from graphrag.synthetic_data import build_extraction_prompt

    prompt = build_extraction_prompt(
        "The applicant is a citizen of Iran who sought refugee status..."
    )
    assert "key" in prompt.lower()
    assert "Iran" in prompt


def test_build_synthetic_pair():
    """Verify synthetic pair construction."""
    from graphrag.synthetic_data import build_synthetic_pair

    pair = build_synthetic_pair(
        query_summary="Immigration case about refugee status from Iran",
        candidate_text="In this case, the applicant from Iran...",
        candidate_id="012345.txt",
    )
    assert pair["query"] is not None
    assert pair["candidate_id"] == "012345.txt"
    assert pair["label"] == 1
```

- [ ] **Step 3: Write the implementation**

Create `src/graphrag/synthetic_data.py` with:

- `EXTRACTION_PROMPT`: Asks LLM to extract 3-5 key legal facts (area of law, legal issue, statutes, facts, outcome).
- `QUERY_GENERATION_PROMPT`: Asks LLM to write a search query from extracted facts.
- `build_extraction_prompt(text)`: Constructs fact extraction prompt with truncated text.
- `build_synthetic_pair(query_summary, candidate_text, candidate_id)`: Creates a training pair dict.
- `generate_synthetic_pairs(corpus_texts, labels, n_pairs)`: Full pipeline -- extracts facts from positive candidates, generates synthetic queries, saves to JSONL.

Key implementation details:
- Uses existing `OllamaClient` (already running deepseek-r1:8b)
- Samples from labeled positive documents for fact extraction
- Saves to `output/synthetic_pairs.jsonl` for caching
- Logs progress every 100 documents

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_synthetic_data.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/graphrag/synthetic_data.py tests/test_synthetic_data.py src/graphrag/config.py
git commit -m "feat: add synthetic training data generation module"
```

### Task 12: Integrate synthetic data into training pipeline

**Files:**
- Modify: `src/graphrag/run_pipeline_v2.py`
- Modify: `src/graphrag/finetune_biencoder.py`

- [ ] **Step 1: Add synthetic data loading to bi-encoder training**

In `finetune_biencoder.py`, modify `mine_hard_negatives()` to accept `synthetic_pairs` parameter.
Append synthetic triplets to the training data after mining real negatives.

- [ ] **Step 2: Wire synthetic data generation into run_train_pipeline()**

Add before Stage 3 (bi-encoder):

```python
    # Synthetic data augmentation
    synthetic_pairs = None
    if USE_SYNTHETIC_DATA:
        synthetic_path = OUTPUT_DIR / "synthetic_pairs.jsonl"
        if synthetic_path.exists():
            import json as json_mod
            synthetic_pairs = [
                json_mod.loads(line) for line in synthetic_path.read_text().splitlines()
                if line.strip()
            ]
            logger.info("Loaded %d synthetic pairs from cache", len(synthetic_pairs))
        else:
            from graphrag.synthetic_data import generate_synthetic_pairs
            synthetic_pairs = generate_synthetic_pairs(raw_corpus, labels)
```

- [ ] **Step 3: Commit**

```bash
git add src/graphrag/finetune_biencoder.py src/graphrag/run_pipeline_v2.py
git commit -m "feat: integrate synthetic data into training pipeline"
```

---

## Execution Timeline

Given current hardware state (both GPUs busy with cross-encoder training):

| Day | Time | Action |
|-----|------|--------|
| **Now** | Mar 13 ~05:00 UTC | Chunks 1-4 code-only tasks (config, modules, tests, integration) |
| **Mar 13 ~19:00 UTC** | Node 1 "smart" CE done | Run BGE-M3 encoding (Task 4, ~30 min GPU) |
| **Mar 13 ~20:00 UTC** | After BGE-M3 | Run full pipeline with BGE-M3 features -> A/B test CV F1 |
| **Mar 13 ~21:00 UTC** | | Run GNN reranker training (Task 7, ~10 min) |
| **Mar 13 ~22:00 UTC** | | Run pipeline with BGE-M3 + GNN -> A/B test |
| **Mar 14 evening** | Node 2 "longctx" CE done | Run reasoning reranker (slow, ~6h for 2001 queries x 30 candidates) |
| **Mar 15** | | Full pipeline with all enhancements -> final CV F1 |

**Critical path:** BGE-M3 encoding requires GPU. Everything else is code-only until Node 1 frees up.

---

## Expected Feature Expansion

Current: 22 base + 6 score distribution + 2 PPR = 30 features

After all chunks:
- +4 BGE-M3 features (m3_dense, m3_sparse, m3_colbert, m3_fused)
- +2 GNN features (gnn_score, gnn_rank)
- +2 Reasoning features (reasoning_score, reasoning_rank)
- = **38 total features**

Each feature group can be independently toggled via config flags for A/B testing.
