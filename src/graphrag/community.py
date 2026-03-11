"""Community detection and summarization for the knowledge graph."""
import logging
from typing import Any

import igraph as ig
import leidenalg
import networkx as nx
import numpy as np

from graphrag.config import (
    COMMUNITY_EDGE_WEIGHTS,
    LEIDEN_RESOLUTION,
    LLM_MODEL,
)
from graphrag.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

COMMUNITY_SUMMARY_PROMPT = """You are a legal analyst. Given a group of Federal Court of Canada cases
that form a legal community, summarize in 2-3 sentences the shared legal themes, common statutes,
and typical case outcomes. Be specific and factual.

Cases in this community:

{case_descriptions}

Summary:"""


def build_case_similarity_graph(
    G: nx.Graph,
    bm25_neighbors: dict[str, list[tuple[str, float]]] | None = None,
) -> nx.Graph:
    """Project the knowledge graph to a weighted Case-only graph.

    Edge weight formula:
        0.30 * shared_statutes + 0.30 * shared_concepts +
        0.30 * bm25_score + 0.05 * same_judge + 0.05 * same_domain
    """
    case_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Case"]
    logger.info("Building case similarity graph for %d cases", len(case_nodes))

    # Precompute entity sets per case
    case_entities: dict[str, dict[str, set[str]]] = {}
    for case_id in case_nodes:
        neighbors = G[case_id]
        entities: dict[str, set[str]] = {
            "statutes": set(),
            "concepts": set(),
            "tests": set(),
            "judges": set(),
            "domains": set(),
        }
        for neighbor in neighbors:
            node_data = G.nodes[neighbor]
            ntype = node_data.get("type", "")
            if ntype == "Statute":
                entities["statutes"].add(neighbor)
            elif ntype == "LegalConcept":
                entities["concepts"].add(neighbor)
            elif ntype == "LegalTest":
                entities["tests"].add(neighbor)
            elif ntype == "Judge":
                entities["judges"].add(neighbor)
            elif ntype == "Domain":
                entities["domains"].add(neighbor)
        case_entities[case_id] = entities

    # Build weighted edges
    sim_graph = nx.Graph()
    sim_graph.add_nodes_from(case_nodes)

    # Normalize BM25 scores
    bm25_max = 1.0
    if bm25_neighbors:
        all_scores = [s for neighbors in bm25_neighbors.values() for _, s in neighbors]
        bm25_max = max(all_scores) if all_scores else 1.0

    w = COMMUNITY_EDGE_WEIGHTS
    processed = set()

    for case_a in case_nodes:
        ents_a = case_entities[case_a]
        for case_b in case_nodes:
            if case_a >= case_b:
                continue
            pair = (case_a, case_b)
            if pair in processed:
                continue
            processed.add(pair)

            ents_b = case_entities[case_b]

            shared_stat = len(ents_a["statutes"] & ents_b["statutes"])
            shared_conc = len(ents_a["concepts"] & ents_b["concepts"])
            same_judge = 1.0 if ents_a["judges"] & ents_b["judges"] else 0.0
            same_domain = 1.0 if ents_a["domains"] & ents_b["domains"] else 0.0

            # BM25 score
            bm25_score = 0.0
            doc_a = case_a.replace("case:", "")
            doc_b = case_b.replace("case:", "")
            if bm25_neighbors:
                for neighbor_id, score in bm25_neighbors.get(f"{doc_a}.txt", []):
                    if neighbor_id == f"{doc_b}.txt":
                        bm25_score = score / bm25_max
                        break

            # Compute weight
            weight = (
                w["shared_statutes"] * min(shared_stat / 3.0, 1.0)
                + w["shared_concepts"] * min(shared_conc / 3.0, 1.0)
                + w["bm25"] * bm25_score
                + w["same_judge"] * same_judge
                + w["same_domain"] * same_domain
            )

            if weight > 0.05:  # Threshold to avoid too many edges
                sim_graph.add_edge(case_a, case_b, weight=weight)

    logger.info(
        "Case similarity graph: %d nodes, %d edges",
        sim_graph.number_of_nodes(),
        sim_graph.number_of_edges(),
    )
    return sim_graph


def detect_communities(
    sim_graph: nx.Graph,
    resolution: float = LEIDEN_RESOLUTION,
) -> dict[str, int]:
    """Run Leiden algorithm on the case similarity graph.

    Returns dict mapping case_id -> community_id.
    """
    # Convert to igraph
    nodes = list(sim_graph.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    ig_graph = ig.Graph()
    ig_graph.add_vertices(len(nodes))
    edges = []
    weights = []
    for u, v, d in sim_graph.edges(data=True):
        edges.append((node_to_idx[u], node_to_idx[v]))
        weights.append(d.get("weight", 1.0))

    ig_graph.add_edges(edges)

    # Run Leiden
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=resolution,
        seed=42,
    )

    # Map back to case IDs
    communities = {}
    for comm_id, members in enumerate(partition):
        for idx in members:
            communities[nodes[idx]] = comm_id

    n_communities = len(partition)
    sizes = [len(m) for m in partition]
    logger.info(
        "Detected %d communities (min=%d, max=%d, median=%.0f)",
        n_communities,
        min(sizes),
        max(sizes),
        np.median(sizes),
    )
    return communities


def summarize_communities(
    communities: dict[str, int],
    extractions: dict[str, dict],
    client: OllamaClient,
    model: str = LLM_MODEL,
    max_cases_per_summary: int = 20,
) -> dict[int, str]:
    """Generate LLM summaries for each community's legal theme."""
    # Group cases by community
    comm_cases: dict[int, list[str]] = {}
    for case_id, comm_id in communities.items():
        comm_cases.setdefault(comm_id, []).append(case_id)

    summaries = {}
    for comm_id, case_ids in sorted(comm_cases.items()):
        # Build case descriptions
        descriptions = []
        for case_id in case_ids[:max_cases_per_summary]:
            doc_id = case_id.replace("case:", "")
            ext = extractions.get(doc_id, {})
            desc = (
                f"- {doc_id}: {ext.get('domain', 'unknown')} | "
                f"concepts: {', '.join(ext.get('concepts', [])[:3])} | "
                f"statutes: {', '.join(ext.get('statutes', [])[:3])} | "
                f"outcome: {ext.get('outcome', 'unknown')}"
            )
            descriptions.append(desc)

        prompt = COMMUNITY_SUMMARY_PROMPT.format(
            case_descriptions="\n".join(descriptions)
        )
        try:
            summary = client.generate(model, prompt, temperature=0.3, max_tokens=300)
            summaries[comm_id] = summary.strip()
        except Exception:
            logger.exception("Failed to summarize community %d", comm_id)
            summaries[comm_id] = f"Community {comm_id}: {len(case_ids)} cases"

        logger.info("Community %d (%d cases): %s...", comm_id, len(case_ids), summaries[comm_id][:80])

    return summaries
