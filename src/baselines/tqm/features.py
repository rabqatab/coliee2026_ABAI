"""Feature extraction for TQM LambdaMART ranker.

10 features per (query, candidate) pair, combining lexical, semantic,
and structural signals.
"""
import re

import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

# Feature names in stable order (used by model.py for consistency)
FEATURE_NAMES = [
    "bm25_score",
    "tfidf_cosine",
    "dense_cosine",
    "query_len",
    "doc_len",
    "len_ratio",
    "paragraph_count",
    "bm25_rank",
    "year_proximity",
    "shared_top_k",
]


def build_tfidf(
    corpus: dict[str, str],
) -> tuple[TfidfVectorizer, dict[str, spmatrix]]:
    """Build TF-IDF matrix for the corpus.

    Returns:
        (fitted vectorizer, {doc_id: sparse_row_vector})
    """
    doc_ids = sorted(corpus.keys())
    vectorizer = TfidfVectorizer(max_features=50000, sublinear_tf=True)
    matrix = vectorizer.fit_transform([corpus[d] for d in doc_ids])
    return vectorizer, {did: matrix[i] for i, did in enumerate(doc_ids)}


def extract_year(text: str) -> int:
    """Extract year from case text header (first 500 chars)."""
    match = re.search(r"\b(19|20)\d{2}\b", text[:500])
    return int(match.group()) if match else 2000


def extract_tqm_features(
    query_id: str,
    candidate_id: str,
    corpus: dict[str, str],
    bm25_candidates: dict[str, list[tuple[str, float]]],
    tqm_embeddings: dict[str, np.ndarray],
    tfidf_vectors: dict[str, spmatrix],
) -> dict[str, float]:
    """Extract 10 features for a single (query, candidate) pair.

    Features:
        1. bm25_score: BM25 score of candidate for query
        2. tfidf_cosine: TF-IDF cosine similarity
        3. dense_cosine: Bi-encoder embedding cosine similarity
        4. query_len: Number of whitespace tokens in query
        5. doc_len: Number of whitespace tokens in candidate
        6. len_ratio: doc_len / query_len
        7. paragraph_count: Number of non-empty paragraphs in candidate
        8. bm25_rank: Rank position of candidate in BM25 results (201 if absent)
        9. year_proximity: 1 / (1 + |year_q - year_d|)
        10. shared_top_k: Jaccard overlap of top-50 BM25 neighbors
    """
    q_text = corpus[query_id]
    d_text = corpus[candidate_id]

    # BM25 score and rank
    bm25_list = bm25_candidates.get(query_id, [])
    bm25_scores = {cid: s for cid, s in bm25_list}
    bm25_score = bm25_scores.get(candidate_id, 0.0)
    bm25_rank = next(
        (i + 1 for i, (cid, _) in enumerate(bm25_list) if cid == candidate_id),
        201,
    )

    # TF-IDF cosine similarity
    q_tfidf = tfidf_vectors.get(query_id)
    d_tfidf = tfidf_vectors.get(candidate_id)
    if q_tfidf is not None and d_tfidf is not None:
        tfidf_cos = float(sklearn_cosine(q_tfidf, d_tfidf)[0, 0])
    else:
        tfidf_cos = 0.0

    # Dense cosine similarity (bi-encoder)
    q_emb = tqm_embeddings.get(query_id)
    d_emb = tqm_embeddings.get(candidate_id)
    if q_emb is not None and d_emb is not None:
        dense_cos = float(np.dot(q_emb, d_emb))
    else:
        dense_cos = 0.0

    # Length features
    q_tokens = q_text.split()
    d_tokens = d_text.split()
    query_len = len(q_tokens)
    doc_len = len(d_tokens)
    len_ratio = doc_len / max(query_len, 1)

    # Paragraph count
    para_count = len([p for p in d_text.split("\n\n") if p.strip()])

    # Year proximity
    q_year = extract_year(q_text)
    d_year = extract_year(d_text)
    year_prox = 1.0 / (1.0 + abs(q_year - d_year))

    # Shared top-K neighbor overlap (top-50 BM25 neighbors)
    q_neighbors = set(cid for cid, _ in bm25_list[:50])
    d_list = bm25_candidates.get(candidate_id, [])
    d_neighbors = set(cid for cid, _ in d_list[:50])
    union = q_neighbors | d_neighbors
    shared = len(q_neighbors & d_neighbors) / max(len(union), 1)

    return {
        "bm25_score": bm25_score,
        "tfidf_cosine": tfidf_cos,
        "dense_cosine": dense_cos,
        "query_len": float(query_len),
        "doc_len": float(doc_len),
        "len_ratio": len_ratio,
        "paragraph_count": float(para_count),
        "bm25_rank": float(bm25_rank),
        "year_proximity": year_prox,
        "shared_top_k": shared,
    }
