"""Vector-based Pseudo Relevance Feedback for dense retrieval.

Uses top-K initially retrieved documents to expand the query embedding,
then re-retrieves with the expanded query. Lightweight and effective.

Ref: Li et al. (TOIS 2023) — "PRF with Deep Language Models and Dense Retrievers"
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


def expand_query_embeddings(
    query_ids: list[str],
    initial_results: dict[str, list[tuple[str, float]]],
    doc_ids: list[str],
    embeddings: np.ndarray,
    n_feedback: int = 3,
    query_weight: float = 0.7,
) -> np.ndarray:
    """Expand query embeddings using vector-based PRF.

    For each query:
    1. Take top-n_feedback documents from initial retrieval
    2. Average their embeddings
    3. Interpolate: expanded = query_weight * query + (1-query_weight) * avg_feedback

    Args:
        query_ids: List of query document IDs.
        initial_results: Initial retrieval results per query.
        doc_ids: All document IDs (aligned with embeddings rows).
        embeddings: Pre-computed embeddings (n_docs, embed_dim).
        n_feedback: Number of top documents for feedback.
        query_weight: Weight for original query in interpolation.

    Returns:
        Expanded query embeddings matrix (len(query_ids), embed_dim).
    """
    doc_idx = {did: i for i, did in enumerate(doc_ids)}
    expanded = np.zeros((len(query_ids), embeddings.shape[1]), dtype=np.float32)

    for i, qid in enumerate(query_ids):
        if qid not in doc_idx:
            continue

        q_emb = embeddings[doc_idx[qid]]

        results = initial_results.get(qid, [])
        feedback_ids = [cid for cid, _ in results[:n_feedback] if cid in doc_idx]

        if feedback_ids:
            fb_embs = embeddings[[doc_idx[cid] for cid in feedback_ids]]
            fb_avg = fb_embs.mean(axis=0)
            expanded[i] = query_weight * q_emb + (1 - query_weight) * fb_avg
        else:
            expanded[i] = q_emb

    # Normalize
    norms = np.linalg.norm(expanded, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    expanded = expanded / norms

    logger.info("PRF: expanded %d query embeddings (n_feedback=%d, query_weight=%.2f)",
                len(query_ids), n_feedback, query_weight)
    return expanded


def prf_retrieve(
    query_ids: list[str],
    expanded_queries: np.ndarray,
    doc_ids: list[str],
    embeddings: np.ndarray,
    top_k: int = 200,
) -> dict[str, list[tuple[str, float]]]:
    """Re-retrieve using PRF-expanded query embeddings.

    Args:
        query_ids: Query IDs.
        expanded_queries: Expanded query embeddings (n_queries, embed_dim).
        doc_ids: Document IDs aligned with embeddings.
        embeddings: Document embeddings.
        top_k: Results per query.

    Returns:
        {query_id: [(doc_id, score), ...]}
    """
    doc_idx = {did: i for i, did in enumerate(doc_ids)}

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    normed_corpus = embeddings / norms

    results = {}
    for i, qid in enumerate(query_ids):
        if qid not in doc_idx:
            continue

        sims = normed_corpus @ expanded_queries[i]
        sims[doc_idx[qid]] = -1.0  # exclude self

        top_indices = np.argpartition(-sims, top_k)[:top_k]
        top_indices = top_indices[np.argsort(-sims[top_indices])]

        results[qid] = [(doc_ids[idx], float(sims[idx])) for idx in top_indices]

    return results
