"""Multi-signal retrieval and fusion for case retrieval."""
import logging
from typing import Any

import numpy as np

from graphrag.config import ENTITY_WEIGHTS, RRF_K, BM25_TOP_K, STAGE1_TOP_K

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    rankings: dict[str, list[tuple[str, float]]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        rankings: dict mapping signal_name -> [(doc_id, score), ...] sorted descending
        k: smoothing parameter

    Returns:
        Fused ranking as [(doc_id, rrf_score), ...] sorted descending
    """
    scores: dict[str, float] = {}
    for signal_name, ranked_list in rankings.items():
        for rank, (doc_id, _) in enumerate(ranked_list, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: -x[1])


def weighted_entity_score(
    query_entities: dict[str, Any],
    candidate_entities: dict[str, Any],
) -> float:
    """Compute weighted Jaccard-like entity overlap score.

    Weights: statutes 0.35, concepts 0.30, tests 0.20, domain 0.10, judge 0.05
    """
    w = ENTITY_WEIGHTS
    score = 0.0

    # Set overlap for statutes, concepts, tests
    for key in ("statutes", "concepts", "tests"):
        q_set = set(query_entities.get(key, set()))
        c_set = set(candidate_entities.get(key, set()))
        if q_set or c_set:
            union = q_set | c_set
            intersection = q_set & c_set
            score += w[key] * (len(intersection) / len(union) if union else 0)

    # Domain match
    if query_entities.get("domain") == candidate_entities.get("domain"):
        if query_entities.get("domain") != "other":
            score += w["domain"]

    # Judge match
    if query_entities.get("judge") and query_entities.get("judge") == candidate_entities.get("judge"):
        score += w["judge"]

    return score


def signal_entity_graph(
    query_id: str,
    query_entities: dict[str, Any],
    corpus_entities: dict[str, dict[str, Any]],
    top_k: int = BM25_TOP_K,
) -> list[tuple[str, float]]:
    """Signal S2: Entity graph traversal scoring."""
    scores = []
    for doc_id, ents in corpus_entities.items():
        if doc_id == query_id:
            continue
        score = weighted_entity_score(query_entities, ents)
        if score > 0:
            scores.append((doc_id, score))
    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]


def signal_community(
    query_embedding: np.ndarray,
    community_embeddings: np.ndarray,
    community_members: dict[int, list[str]],
    top_k: int = BM25_TOP_K,
) -> list[tuple[str, float]]:
    """Signal S3: Community matching via embedding similarity.

    Finds nearest communities to query, returns member cases with scores.
    """
    # Cosine similarity to each community
    q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    c_norms = community_embeddings / (
        np.linalg.norm(community_embeddings, axis=1, keepdims=True) + 1e-10
    )
    sims = c_norms @ q_norm

    # Get top communities
    top_comm_indices = np.argsort(-sims)[:5]

    # Assign scores to member cases
    scores: dict[str, float] = {}
    for comm_idx in top_comm_indices:
        comm_score = float(sims[comm_idx])
        for case_id in community_members.get(int(comm_idx), []):
            doc_id = case_id.replace("case:", "") + ".txt"
            scores[doc_id] = max(scores.get(doc_id, 0), comm_score)

    result = sorted(scores.items(), key=lambda x: -x[1])
    return result[:top_k]


def signal_embedding(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    corpus_ids: list[str],
    query_id: str,
    top_k: int = BM25_TOP_K,
) -> list[tuple[str, float]]:
    """Signal S4: Dense embedding similarity."""
    q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    c_norms = corpus_embeddings / (
        np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-10
    )
    sims = c_norms @ q_norm

    top_indices = np.argsort(-sims)
    results = []
    for idx in top_indices:
        if corpus_ids[idx] == query_id:
            continue
        results.append((corpus_ids[idx], float(sims[idx])))
        if len(results) >= top_k:
            break
    return results


def retrieve_multi_signal(
    query_id: str,
    bm25_results: list[tuple[str, float]],
    entity_results: list[tuple[str, float]],
    community_results: list[tuple[str, float]],
    embedding_results: list[tuple[str, float]],
    top_k: int = STAGE1_TOP_K,
) -> list[tuple[str, float]]:
    """Stage 1: Fuse signals S1-S4 via RRF to get top-k candidates.

    These candidates are then passed to Stage 2 (reasoning chains).
    """
    rankings = {
        "bm25": bm25_results,
        "entity_graph": entity_results,
        "community": community_results,
        "embedding": embedding_results,
    }
    fused = reciprocal_rank_fusion(rankings)
    return fused[:top_k]
