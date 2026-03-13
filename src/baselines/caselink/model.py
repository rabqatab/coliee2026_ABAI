"""CaseLink 2025 baseline — GNN over heterogeneous case graph.

Reproduces the core idea from the 2025 runner-up (arXiv 2505.20743):
    1. Build a heterogeneous graph (case-case citations + case-statute edges)
    2. Initialize node features with sentence-transformer embeddings
    3. Train a 2-layer GraphSAGE GNN with InfoNCE loss
    4. Use GNN output embeddings for cosine-similarity retrieval

Uses plain PyTorch instead of torch-geometric to avoid dependency issues.
"""
import logging
import random
from collections import defaultdict

import numpy as np
import torch

from baselines.caselink.gnn import SimpleGraphSAGE
from baselines.caselink.graph import CaseGraph
from baselines.caselink.node_features import NodeFeatureGenerator
from baselines.common.base_model import BaselineModel

logger = logging.getLogger(__name__)


class CaseLinkBaseline(BaselineModel):
    """CaseLink 2025 (GNN): graph-based case retrieval."""

    def __init__(
        self,
        hidden_dim: int = 256,
        epochs: int = 100,
        lr: float = 1e-3,
        temperature: float = 0.07,
        batch_size: int = 128,
    ):
        self._hidden_dim = hidden_dim
        self._epochs = epochs
        self._lr = lr
        self._temperature = temperature
        self._batch_size = batch_size
        self._graph = CaseGraph()
        self._feature_gen = NodeFeatureGenerator()
        self._gnn: SimpleGraphSAGE | None = None
        self._case_embeddings: dict[str, np.ndarray] | None = None

    def name(self) -> str:
        return "CaseLink 2025 (GNN)"

    def train(
        self,
        corpus: dict[str, str],
        train_queries: list[str],
        labels: dict[str, list[str]],
        bm25_candidates: dict[str, list[tuple[str, float]]] | None = None,
    ) -> None:
        """Train the GNN on the case graph.

        Steps:
            1. Build heterogeneous graph from corpus + labels
            2. Encode initial node features with sentence-transformer
            3. Build combined adjacency (case-case + case-statute-case)
            4. Train GraphSAGE with InfoNCE contrastive loss
            5. Store final GNN embeddings for retrieval
        """
        # 1. Build graph
        self._graph.build(corpus, labels, train_queries)

        # 2. Get initial node features
        case_embs = self._feature_gen.encode_cases(corpus)

        # Build node feature matrix (cases only)
        n_cases = len(self._graph.case_ids)
        input_dim = 384  # all-MiniLM-L6-v2 output dimension
        node_features = np.zeros((n_cases, input_dim), dtype=np.float32)
        for i, cid in enumerate(self._graph.case_ids):
            if cid in case_embs:
                node_features[i] = case_embs[cid]

        # 3. Build adjacency list (combine case-case and case-statute-case edges)
        adj: dict[int, set[int]] = defaultdict(set)
        for src, dst in self._graph.case_case_edges:
            adj[src].add(dst)

        # Add case-statute-case edges (cases connected via shared statutes)
        statute_cases: dict[int, set[int]] = defaultdict(set)
        for case_idx, stat_idx in self._graph.case_statute_edges:
            statute_cases[stat_idx].add(case_idx)
        for _stat_idx, cases in statute_cases.items():
            cases_list = list(cases)
            for c1 in cases_list:
                for c2 in cases_list:
                    if c1 != c2:
                        adj[c1].add(c2)

        adjacency_list = {k: list(v) for k, v in adj.items()}

        avg_neighbors = (
            np.mean([len(v) for v in adjacency_list.values()])
            if adjacency_list
            else 0
        )
        logger.info(
            "Adjacency: %d nodes with neighbors (avg %.1f neighbors)",
            len(adjacency_list),
            avg_neighbors,
        )

        # 4. Train GNN with InfoNCE loss
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Training GNN on device: %s", device)

        self._gnn = SimpleGraphSAGE(
            input_dim=input_dim,
            hidden_dim=self._hidden_dim,
            output_dim=self._hidden_dim,
        ).to(device)

        node_tensor = torch.from_numpy(node_features).to(device)
        optimizer = torch.optim.Adam(self._gnn.parameters(), lr=self._lr)

        # Build positive pairs from labels
        pos_pairs: list[tuple[int, int]] = []
        for qid in train_queries:
            if qid not in self._graph._case_to_idx:
                continue
            q_idx = self._graph._case_to_idx[qid]
            for did in labels.get(qid, []):
                if did in self._graph._case_to_idx:
                    d_idx = self._graph._case_to_idx[did]
                    pos_pairs.append((q_idx, d_idx))

        if not pos_pairs:
            logger.warning("No positive pairs for GNN training — skipping")
            self._case_embeddings = {
                cid: case_embs.get(cid, np.zeros(input_dim))
                for cid in self._graph.case_ids
            }
            return

        logger.info(
            "Training GNN: %d positive pairs, %d nodes, %d epochs",
            len(pos_pairs),
            n_cases,
            self._epochs,
        )

        self._gnn.train()
        for epoch in range(self._epochs):
            optimizer.zero_grad()

            # Forward pass
            out = self._gnn(node_tensor, adjacency_list)

            # InfoNCE loss on positive pairs with in-batch negatives
            actual_batch = min(self._batch_size, len(pos_pairs))
            batch = random.sample(pos_pairs, actual_batch)

            q_indices = torch.tensor([p[0] for p in batch], device=device)
            d_indices = torch.tensor([p[1] for p in batch], device=device)

            q_emb = out[q_indices]  # (B, dim)
            d_emb = out[d_indices]  # (B, dim)

            # Cosine similarity matrix with temperature scaling
            sim_matrix = torch.mm(q_emb, d_emb.t()) / self._temperature
            target = torch.arange(actual_batch, device=device)
            loss = torch.nn.functional.cross_entropy(sim_matrix, target)

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                logger.info(
                    "GNN epoch %d/%d  loss=%.4f",
                    epoch + 1,
                    self._epochs,
                    loss.item(),
                )

        # 5. Compute final embeddings for all cases
        self._gnn.eval()
        with torch.no_grad():
            final_embs = self._gnn(node_tensor, adjacency_list).cpu().numpy()

        self._case_embeddings = {}
        for i, cid in enumerate(self._graph.case_ids):
            self._case_embeddings[cid] = final_embs[i]

        logger.info(
            "GNN training complete — %d case embeddings stored",
            len(self._case_embeddings),
        )

    def predict(
        self,
        query_id: str,
        corpus: dict[str, str],
        bm25_candidates: list[tuple[str, float]] | None = None,
    ) -> list[tuple[str, float]]:
        """Rank candidates by cosine similarity of GNN embeddings.

        If bm25_candidates is provided, scores only those candidates.
        Otherwise scores all cases in the graph.

        Returns:
            Sorted list of (candidate_id, cosine_score), descending.
        """
        if self._case_embeddings is None or query_id not in self._case_embeddings:
            return []

        q_emb = self._case_embeddings[query_id]

        # Determine candidate set
        if bm25_candidates:
            cand_ids = [cid for cid, _ in bm25_candidates if cid != query_id]
        else:
            cand_ids = [cid for cid in self._case_embeddings if cid != query_id]

        results: list[tuple[str, float]] = []
        for cid in cand_ids:
            if cid in self._case_embeddings:
                score = float(np.dot(q_emb, self._case_embeddings[cid]))
                results.append((cid, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:200]
