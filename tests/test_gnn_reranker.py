"""Tests for GNN score refinement."""
import pytest
import numpy as np


def test_build_subgraph_extracts_candidates():
    """Subgraph should contain query + all candidates as nodes."""
    from coliee_task1.stages.gnn import build_query_subgraph

    # Mock corpus graph adjacency (5 nodes)
    adj = {0: [1, 2], 1: [0, 3], 2: [0, 4], 3: [1], 4: [2]}
    node_features = np.random.randn(5, 16).astype(np.float32)
    query_idx = 0
    candidate_idxs = [1, 2, 3]

    subgraph = build_query_subgraph(adj, node_features, query_idx, candidate_idxs)

    assert subgraph["num_nodes"] >= 4  # query + 3 candidates
    assert subgraph["node_features"].shape[0] == subgraph["num_nodes"]
    assert subgraph["edge_index"].shape[0] == 2  # (2, num_edges)


def test_gnn_forward_produces_scores():
    """GNN model forward pass should produce one score per candidate."""
    import torch
    from coliee_task1.stages.gnn import GNNReranker

    model = GNNReranker(input_dim=16, hidden_dim=32, num_layers=2, heads=2)
    # Simulate 5-node subgraph
    x = torch.randn(5, 16)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    candidate_mask = torch.tensor([False, True, True, True, False])

    scores = model(x, edge_index, candidate_mask)
    assert scores.shape == (3,)  # one score per candidate
