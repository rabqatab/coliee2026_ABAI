"""Simple GraphSAGE-like GNN using plain PyTorch (no PyG dependency).

This avoids the complex torch-geometric installation while reproducing
the core message-passing idea from CaseLink 2025.
"""
import random
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SimpleGraphSAGE(nn.Module):
    """Two-layer GraphSAGE with mean aggregation.

    Each layer concatenates a self-transform and a neighbor-mean-transform,
    then applies ReLU + dropout. The final output is L2-normalized.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        # Layer 1: aggregate neighbors + transform
        self.W1_self = nn.Linear(input_dim, hidden_dim)
        self.W1_neigh = nn.Linear(input_dim, hidden_dim)

        # Layer 2
        self.W2_self = nn.Linear(hidden_dim, output_dim)
        self.W2_neigh = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency_list: dict[int, list[int]],
    ) -> torch.Tensor:
        """Forward pass through 2-layer GraphSAGE.

        Args:
            node_features: (N, input_dim) tensor of initial node embeddings.
            adjacency_list: Mapping from node index to list of neighbor indices.

        Returns:
            (N, output_dim) tensor of L2-normalized GNN embeddings.
        """
        N = node_features.size(0)

        # Layer 1
        neigh_feats = self._mean_aggregate(node_features, adjacency_list, N)
        h = F.relu(self.W1_self(node_features) + self.W1_neigh(neigh_feats))
        h = self.dropout(h)

        # Layer 2
        neigh_feats2 = self._mean_aggregate(h, adjacency_list, N)
        out = self.W2_self(h) + self.W2_neigh(neigh_feats2)

        # L2 normalize
        out = F.normalize(out, p=2, dim=1)
        return out

    def _mean_aggregate(
        self,
        features: torch.Tensor,
        adjacency_list: dict[int, list[int]],
        N: int,
    ) -> torch.Tensor:
        """Mean aggregation of neighbor features.

        Samples up to 25 neighbors per node for efficiency on dense graphs.
        Isolated nodes use a self-loop (their own features).
        """
        device = features.device
        agg = torch.zeros(N, features.size(1), device=device)
        for i in range(N):
            neighbors = adjacency_list.get(i, [])
            if neighbors:
                # Sample up to 25 neighbors for efficiency
                if len(neighbors) > 25:
                    neighbors = random.sample(neighbors, 25)
                agg[i] = features[neighbors].mean(dim=0)
            else:
                agg[i] = features[i]  # self-loop if no neighbors
        return agg
