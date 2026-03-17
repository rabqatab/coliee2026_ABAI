"""GNN Score Refinement: GAT-based reranking on a corpus graph.

Builds a document graph from semantic similarity (k-NN cosine) and entity overlap,
then trains a 2-layer GAT to refine retrieval scores by capturing cross-document
interactions.

Features produced per (query, candidate) pair:
  - gnn_score: GAT-refined relevance score
  - gnn_rank: Rank of candidate by gnn_score within query

References:
  - Di Francesco et al., "GNRR" (SIGIR 2024, arXiv:2406.11720)
  - Wendlinger et al., "The Missing Link" (DEXA 2025, arXiv:2506.22165)
"""
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from coliee_task1.config import (
    GNN_HIDDEN_DIM,
    GNN_NUM_LAYERS,
    GNN_HEADS,
    GNN_DROPOUT,
    GNN_LR,
    GNN_EPOCHS,
    GNN_K_NEIGHBORS,
    GNN_ENTITY_WEIGHT,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)


class GNNReranker(nn.Module):
    """2-layer GAT with a score prediction head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = GNN_HIDDEN_DIM,
        num_layers: int = GNN_NUM_LAYERS,
        heads: int = GNN_HEADS,
        dropout: float = GNN_DROPOUT,
    ):
        super().__init__()
        from torch_geometric.nn import GATConv

        self.convs = nn.ModuleList()
        # First layer
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
        # Additional layers
        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout)
            )

        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass. Returns one score per candidate node.

        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Graph edges (2, num_edges)
            candidate_mask: Boolean mask selecting candidate nodes

        Returns:
            Scores for candidate nodes (num_candidates,)
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Extract candidate node embeddings and score
        candidate_embs = x[candidate_mask]
        scores = self.score_head(candidate_embs).squeeze(-1)
        return scores


def build_corpus_graph(
    embeddings: np.ndarray,
    doc_ids: list[str],
    entity_overlaps: dict[tuple[str, str], float] | None = None,
    k: int = GNN_K_NEIGHBORS,
    entity_weight: float = GNN_ENTITY_WEIGHT,
) -> dict[int, list[int]]:
    """Build document graph from k-NN cosine similarity + entity overlap.

    Args:
        embeddings: (n_docs, dim) document embeddings (assumed normalized)
        doc_ids: Document IDs corresponding to embedding rows
        entity_overlaps: Optional {(doc_a, doc_b): overlap_score} for entity edges
        k: Number of nearest neighbors per document
        entity_weight: Not used for adjacency building, reserved for future edge weighting

    Returns:
        Adjacency list {node_idx: [neighbor_idx, ...]}
    """
    logger.info("Building corpus graph (k=%d, %d docs) ...", k, len(doc_ids))
    t0 = time.time()

    id_to_idx = {did: i for i, did in enumerate(doc_ids)}
    n = len(doc_ids)

    # Compute cosine similarity matrix (embeddings should be normalized)
    sim_matrix = embeddings @ embeddings.T

    adj: dict[int, list[int]] = {i: [] for i in range(n)}

    # k-NN edges (semantic similarity)
    for i in range(n):
        sims = sim_matrix[i].copy()
        sims[i] = -1.0  # exclude self
        top_k_idx = np.argpartition(sims, -k)[-k:]
        for j in top_k_idx:
            if j != i:
                adj[i].append(int(j))
                adj[int(j)].append(i)  # undirected

    # Entity overlap edges
    if entity_overlaps:
        for (doc_a, doc_b), score in entity_overlaps.items():
            if score > 0 and doc_a in id_to_idx and doc_b in id_to_idx:
                a_idx, b_idx = id_to_idx[doc_a], id_to_idx[doc_b]
                if b_idx not in adj[a_idx]:
                    adj[a_idx].append(b_idx)
                if a_idx not in adj[b_idx]:
                    adj[b_idx].append(a_idx)

    # Deduplicate
    for i in adj:
        adj[i] = list(set(adj[i]))

    n_edges = sum(len(v) for v in adj.values()) // 2
    logger.info("Corpus graph: %d nodes, %d edges (%.1fs)", n, n_edges, time.time() - t0)
    return adj


def build_query_subgraph(
    adj: dict[int, list[int]],
    node_features: np.ndarray,
    query_idx: int,
    candidate_idxs: list[int],
) -> dict:
    """Extract a local subgraph for a query: query + candidates + 1-hop neighbors.

    Returns:
        {
            "node_features": np.ndarray (num_subgraph_nodes, feat_dim),
            "edge_index": np.ndarray (2, num_edges),
            "candidate_mask": np.ndarray (num_subgraph_nodes,) bool,
            "query_mask": np.ndarray (num_subgraph_nodes,) bool,
            "num_nodes": int,
            "local_to_global": dict mapping local -> global indices,
            "global_to_local": dict mapping global -> local indices,
        }
    """
    # Collect nodes: query + candidates + 1-hop neighbors
    relevant_nodes = {query_idx} | set(candidate_idxs)
    for idx in list(relevant_nodes):
        if idx in adj:
            for neighbor in adj[idx]:
                relevant_nodes.add(neighbor)

    sorted_nodes = sorted(relevant_nodes)
    global_to_local = {g: l for l, g in enumerate(sorted_nodes)}
    local_to_global = {l: g for g, l in global_to_local.items()}

    n_sub = len(sorted_nodes)

    # Build subgraph features
    sub_features = node_features[sorted_nodes]

    # Build subgraph edges
    edges_src, edges_dst = [], []
    for g_idx in sorted_nodes:
        l_idx = global_to_local[g_idx]
        if g_idx in adj:
            for g_neighbor in adj[g_idx]:
                if g_neighbor in global_to_local:
                    edges_src.append(l_idx)
                    edges_dst.append(global_to_local[g_neighbor])

    if edges_src:
        edge_index = np.array([edges_src, edges_dst], dtype=np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    # Masks
    candidate_mask = np.zeros(n_sub, dtype=bool)
    for c_idx in candidate_idxs:
        if c_idx in global_to_local:
            candidate_mask[global_to_local[c_idx]] = True

    query_mask = np.zeros(n_sub, dtype=bool)
    if query_idx in global_to_local:
        query_mask[global_to_local[query_idx]] = True

    return {
        "node_features": sub_features,
        "edge_index": edge_index,
        "candidate_mask": candidate_mask,
        "query_mask": query_mask,
        "num_nodes": n_sub,
        "local_to_global": local_to_global,
        "global_to_local": global_to_local,
    }


def build_node_features(
    embeddings: np.ndarray,
    query_idx: int,
    retrieval_scores: dict[int, dict[str, float]],
    embed_dim: int = 32,
) -> np.ndarray:
    """Build node features: truncated embeddings + per-query retrieval scores.

    Args:
        embeddings: (n_docs, full_dim) document embeddings
        query_idx: Index of the query document
        retrieval_scores: {doc_idx: {"bm25": score, "biencoder": score, "crossencoder": score}}
        embed_dim: Truncated embedding dimension

    Returns:
        (n_docs, embed_dim + 3) feature matrix
    """
    # Truncate embeddings via PCA-like projection (just take first embed_dim dims)
    trunc_embs = embeddings[:, :embed_dim]

    # Add retrieval scores as node features (0 for docs without scores)
    n_docs = len(embeddings)
    score_feats = np.zeros((n_docs, 3), dtype=np.float32)
    for idx, scores in retrieval_scores.items():
        score_feats[idx, 0] = scores.get("bm25", 0.0)
        score_feats[idx, 1] = scores.get("biencoder", 0.0)
        score_feats[idx, 2] = scores.get("crossencoder", 0.0)

    return np.hstack([trunc_embs, score_feats]).astype(np.float32)


def train_gnn_reranker(
    adj: dict[int, list[int]],
    node_features: np.ndarray,
    train_queries: list[dict],
    epochs: int = GNN_EPOCHS,
    lr: float = GNN_LR,
    save_dir: Path | None = None,
) -> GNNReranker:
    """Train GNN reranker on training queries.

    Args:
        adj: Corpus graph adjacency list
        node_features: (n_docs, feat_dim) per-document features
        train_queries: List of {
            "query_idx": int,
            "candidate_idxs": list[int],
            "labels": list[int],  # 1 for positive, 0 for negative
        }
        epochs: Number of training epochs
        lr: Learning rate
        save_dir: Directory to save trained model

    Returns:
        Trained GNNReranker model
    """
    if save_dir is None:
        save_dir = MODELS_DIR / "gnn_reranker"
    save_dir.mkdir(parents=True, exist_ok=True)

    input_dim = node_features.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNNReranker(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info("Training GNN reranker (%d queries, %d epochs) ...", len(train_queries), epochs)
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_queries = 0

        for query_data in train_queries:
            query_idx = query_data["query_idx"]
            candidate_idxs = query_data["candidate_idxs"]
            labels = query_data["labels"]

            if not candidate_idxs:
                continue

            subgraph = build_query_subgraph(adj, node_features, query_idx, candidate_idxs)

            x = torch.tensor(subgraph["node_features"], dtype=torch.float32).to(device)
            edge_index = torch.tensor(subgraph["edge_index"], dtype=torch.long).to(device)
            candidate_mask = torch.tensor(subgraph["candidate_mask"], dtype=torch.bool).to(device)

            # Map labels to subgraph candidate order
            global_to_local = subgraph["global_to_local"]
            sub_labels = []
            for c_idx, label in zip(candidate_idxs, labels):
                if c_idx in global_to_local:
                    sub_labels.append(label)
            target = torch.tensor(sub_labels, dtype=torch.float32).to(device)

            if len(target) == 0:
                continue

            optimizer.zero_grad()
            scores = model(x, edge_index, candidate_mask)

            # Ensure shapes match
            min_len = min(len(scores), len(target))
            loss = F.binary_cross_entropy(scores[:min_len], target[:min_len])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_queries += 1

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / max(n_queries, 1)
            logger.info("  Epoch %d/%d: loss=%.4f", epoch + 1, epochs, avg_loss)

    # Save model
    torch.save(model.state_dict(), save_dir / "gnn_reranker.pt")
    logger.info("GNN training complete in %.1f seconds", time.time() - t0)

    return model


def gnn_rerank(
    model: GNNReranker,
    adj: dict[int, list[int]],
    node_features: np.ndarray,
    query_idxs: list[int],
    candidate_lists: dict[int, list[int]],
    doc_ids: list[str],
) -> dict[str, dict[str, float]]:
    """Run GNN inference to produce reranking scores.

    Returns:
        {query_id: {candidate_id: gnn_score}}
    """
    device = next(model.parameters()).device
    model.eval()

    results: dict[str, dict[str, float]] = {}

    with torch.no_grad():
        for q_idx in query_idxs:
            c_idxs = candidate_lists.get(q_idx, [])
            if not c_idxs:
                continue

            subgraph = build_query_subgraph(adj, node_features, q_idx, c_idxs)

            x = torch.tensor(subgraph["node_features"], dtype=torch.float32).to(device)
            edge_index = torch.tensor(subgraph["edge_index"], dtype=torch.long).to(device)
            candidate_mask = torch.tensor(subgraph["candidate_mask"], dtype=torch.bool).to(device)

            scores = model(x, edge_index, candidate_mask)

            # Map back to document IDs
            qid = doc_ids[q_idx]
            results[qid] = {}
            score_idx = 0
            for c_idx in c_idxs:
                if c_idx in subgraph["global_to_local"]:
                    results[qid][doc_ids[c_idx]] = float(scores[score_idx])
                    score_idx += 1

    return results
