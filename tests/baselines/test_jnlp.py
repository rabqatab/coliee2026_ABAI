"""Tests for JNLP 2025 baseline."""
from baselines.jnlp.model import JNLPBaseline
from baselines.jnlp.features import extract_jnlp_features, build_corpus_freq

import numpy as np


def test_jnlp_interface():
    model = JNLPBaseline()
    assert model.name() == "JNLP 2025 (BM25+SAILER+LightGBM)"


def test_jnlp_features(mini_corpus, mini_labels):
    """Test that feature extraction produces expected keys."""
    corpus_freq, total_terms = build_corpus_freq(mini_corpus)

    # Fake BM25 candidates
    bm25_candidates = {
        "q1.txt": [("d1.txt", 10.0), ("d3.txt", 8.0), ("d2.txt", 3.0)],
    }

    # Fake SAILER embeddings (random unit vectors)
    rng = np.random.RandomState(42)
    sailer_embeddings = {}
    for did in mini_corpus:
        emb = rng.randn(768).astype(np.float32)
        emb /= np.linalg.norm(emb)
        sailer_embeddings[did] = emb

    feats = extract_jnlp_features(
        "q1.txt", "d1.txt", mini_corpus, bm25_candidates,
        sailer_embeddings, corpus_freq, total_terms,
    )
    expected_keys = {
        "bm25_full", "bm25_para_max", "qld_score", "sailer_sim",
        "bm25_rank", "bm25_ratio", "query_len", "doc_len",
    }
    assert set(feats.keys()) == expected_keys
    assert feats["bm25_full"] == 10.0
    assert feats["bm25_rank"] == 1
    assert feats["bm25_ratio"] == 1.0
    assert feats["query_len"] > 0
    assert feats["doc_len"] > 0


def test_corpus_freq(mini_corpus):
    """Test corpus frequency building."""
    corpus_freq, total_terms = build_corpus_freq(mini_corpus)
    assert total_terms > 0
    assert len(corpus_freq) > 0
    # "immigration" appears in q1, d1, d3
    assert corpus_freq.get("immigration", 0) == 3
