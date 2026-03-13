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
