"""Tests for TQM 2024 baseline."""
import numpy as np

from baselines.tqm.model import TQMBaseline
from baselines.tqm.features import FEATURE_NAMES, extract_tqm_features, build_tfidf


def test_tqm_interface():
    model = TQMBaseline()
    assert model.name() == "TQM 2024 (LTR Fusion)"


def test_feature_names():
    assert len(FEATURE_NAMES) == 10
    assert "bm25_score" in FEATURE_NAMES
    assert "dense_cosine" in FEATURE_NAMES
    assert "shared_top_k" in FEATURE_NAMES


def test_extract_features(mini_corpus, mini_labels):
    """Test feature extraction produces all 10 features with valid values."""
    bm25_candidates = {
        "q1.txt": [("d1.txt", 5.0), ("d3.txt", 4.0), ("d2.txt", 1.0)],
        "q2.txt": [("d2.txt", 6.0), ("d1.txt", 2.0)],
        "d1.txt": [("d3.txt", 3.0), ("q1.txt", 2.5)],
        "d2.txt": [("q2.txt", 4.0)],
        "d3.txt": [("d1.txt", 3.5), ("q1.txt", 2.0)],
    }
    # Fake embeddings (normalized)
    dim = 8
    embeddings = {}
    for doc_id in mini_corpus:
        vec = np.random.randn(dim).astype(np.float32)
        embeddings[doc_id] = vec / np.linalg.norm(vec)

    _, tfidf_vectors = build_tfidf(mini_corpus)

    feats = extract_tqm_features(
        "q1.txt", "d1.txt", mini_corpus,
        bm25_candidates, embeddings, tfidf_vectors,
    )
    assert set(feats.keys()) == set(FEATURE_NAMES)
    assert feats["bm25_score"] == 5.0
    assert feats["bm25_rank"] == 1.0
    assert 0.0 <= feats["tfidf_cosine"] <= 1.0
    assert -1.0 <= feats["dense_cosine"] <= 1.0
    assert feats["query_len"] > 0
    assert feats["doc_len"] > 0
    assert feats["len_ratio"] > 0
    assert 0.0 < feats["year_proximity"] <= 1.0


def test_build_tfidf(mini_corpus):
    vectorizer, vectors = build_tfidf(mini_corpus)
    assert len(vectors) == len(mini_corpus)
    assert vectorizer is not None
    for doc_id, vec in vectors.items():
        assert vec.shape[1] > 0  # sparse row vector has features
