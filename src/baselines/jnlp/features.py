"""Feature extraction for JNLP 2025 baseline."""
import math
import re
from collections import Counter

import numpy as np


def build_corpus_freq(corpus: dict[str, str]) -> tuple[dict[str, int], int]:
    """Build corpus-level term frequencies for QLD.

    Returns (corpus_freq, total_terms).
    """
    corpus_freq: dict[str, int] = Counter()
    total_terms = 0
    for text in corpus.values():
        words = text.lower().split()
        corpus_freq.update(words)
        total_terms += len(words)
    return dict(corpus_freq), total_terms


def qld_score(
    query_text: str,
    doc_text: str,
    corpus_freq: dict[str, int],
    total_terms: int,
    mu: float = 2000.0,
) -> float:
    """Query Likelihood with Dirichlet smoothing."""
    query_terms = query_text.lower().split()
    doc_terms = doc_text.lower().split()
    doc_len = len(doc_terms)
    doc_tf = Counter(doc_terms)

    score = 0.0
    for t in query_terms:
        tf = doc_tf.get(t, 0)
        cf = corpus_freq.get(t, 0)
        p_collection = cf / total_terms if total_terms > 0 else 1e-10
        numerator = tf + mu * p_collection
        denominator = doc_len + mu
        if numerator > 0 and denominator > 0:
            score += math.log(numerator / denominator)
    return score


def _word_overlap_score(query_text: str, doc_text: str) -> float:
    """Simple word overlap ratio for paragraph-level scoring."""
    q_words = set(query_text.lower().split())
    d_words = set(doc_text.lower().split())
    if not q_words:
        return 0.0
    return len(q_words & d_words) / len(q_words)


def extract_jnlp_features(
    query_id: str,
    candidate_id: str,
    corpus: dict[str, str],
    bm25_candidates: dict[str, list[tuple[str, float]]],
    sailer_embeddings: dict,
    corpus_freq: dict[str, int],
    total_terms: int,
) -> dict[str, float]:
    """Extract 8 features for a (query, candidate) pair."""
    query_text = corpus.get(query_id, "")
    doc_text = corpus.get(candidate_id, "")
    cands = bm25_candidates.get(query_id, [])

    # BM25 score and rank
    bm25_score = 0.0
    bm25_rank = 201
    max_bm25 = 0.0
    for rank_idx, (cid, sc) in enumerate(cands, 1):
        if rank_idx == 1:
            max_bm25 = sc
        if cid == candidate_id:
            bm25_score = sc
            bm25_rank = rank_idx

    # bm25_para_max: split query into paragraphs, score each against candidate
    paragraphs = [p.strip() for p in re.split(r"\n\n+", query_text) if p.strip()]
    if paragraphs:
        bm25_para_max = max(_word_overlap_score(p, doc_text) for p in paragraphs)
    else:
        bm25_para_max = 0.0

    # QLD
    qld = qld_score(query_text, doc_text, corpus_freq, total_terms)

    # SAILER similarity
    sailer_sim = 0.0
    q_emb = sailer_embeddings.get(query_id)
    d_emb = sailer_embeddings.get(candidate_id)
    if q_emb is not None and d_emb is not None:
        sailer_sim = float(np.dot(q_emb, d_emb))

    # Length features
    query_len = len(query_text.split())
    doc_len = len(doc_text.split())

    return {
        "bm25_full": bm25_score,
        "bm25_para_max": bm25_para_max,
        "qld_score": qld,
        "sailer_sim": sailer_sim,
        "bm25_rank": bm25_rank,
        "bm25_ratio": bm25_score / max_bm25 if max_bm25 > 0 else 0.0,
        "query_len": query_len,
        "doc_len": doc_len,
    }
