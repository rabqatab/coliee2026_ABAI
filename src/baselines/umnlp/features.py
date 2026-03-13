"""Feature extraction for UMNLP 2024 baseline.

Produces 8 features per (query, candidate) pair:
  1. proposition_sim     - cosine similarity of mean-pooled proposition embeddings
  2. judge_match         - binary: do query and candidate share a judge?
  3. judge_rarity_score  - IDF-weighted judge overlap
  4. quotation_overlap   - 5-gram overlap fraction
  5. para_max_sim        - max paragraph-level word overlap with query
  6. para_mean_sim       - mean paragraph-level word overlap with query
  7. bm25_score          - BM25 retrieval score for this pair
  8. doc_len_ratio       - candidate length / query length
"""
import numpy as np


def quotation_overlap(text1: str, text2: str, n: int = 5) -> float:
    """5-gram overlap fraction between two texts."""
    words1 = text1.lower().split()
    words2 = text2.lower().split()

    if len(words1) < n or len(words2) < n:
        return 0.0

    ngrams1 = set(tuple(words1[i:i + n]) for i in range(len(words1) - n + 1))
    ngrams2 = set(tuple(words2[i:i + n]) for i in range(len(words2) - n + 1))

    if not ngrams1:
        return 0.0

    return len(ngrams1 & ngrams2) / len(ngrams1)


def extract_umnlp_features(
    query_id: str,
    candidate_id: str,
    corpus: dict[str, str],
    bm25_candidates: dict[str, list[tuple[str, float]]],
    proposition_embeddings: dict[str, np.ndarray],
    judge_matcher,
) -> dict[str, float]:
    """Extract 8 features for UMNLP model."""
    q_text = corpus[query_id]
    d_text = corpus[candidate_id]

    # Proposition similarity
    q_emb = proposition_embeddings.get(query_id)
    d_emb = proposition_embeddings.get(candidate_id)
    if q_emb is not None and d_emb is not None:
        prop_sim = float(np.dot(q_emb, d_emb))
    else:
        prop_sim = 0.0

    # Judge matching
    judge_binary, judge_rarity = judge_matcher.match(query_id, candidate_id)

    # Quotation overlap
    quot_overlap = quotation_overlap(q_text, d_text)

    # Paragraph similarities (split doc into paragraphs, compute max and mean
    # word overlap with query)
    paras = [p.strip() for p in d_text.split('\n\n') if p.strip()]
    if paras:
        q_words = set(q_text.lower().split())
        para_sims = []
        for p in paras[:50]:  # limit to first 50 paragraphs
            p_words = set(p.lower().split())
            if q_words:
                sim = len(q_words & p_words) / len(q_words)
                para_sims.append(sim)
        para_max = max(para_sims) if para_sims else 0.0
        para_mean = float(np.mean(para_sims)) if para_sims else 0.0
    else:
        para_max = 0.0
        para_mean = 0.0

    # BM25 score
    bm25_list = bm25_candidates.get(query_id, [])
    bm25_scores = {cid: s for cid, s in bm25_list}
    bm25_score = bm25_scores.get(candidate_id, 0.0)

    # Length ratio
    q_len = len(q_text.split())
    d_len = len(d_text.split())
    len_ratio = d_len / max(q_len, 1)

    return {
        "proposition_sim": prop_sim,
        "judge_match": judge_binary,
        "judge_rarity_score": judge_rarity,
        "quotation_overlap": quot_overlap,
        "para_max_sim": para_max,
        "para_mean_sim": para_mean,
        "bm25_score": bm25_score,
        "doc_len_ratio": len_ratio,
    }
