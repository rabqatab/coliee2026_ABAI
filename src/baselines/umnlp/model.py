"""UMNLP 2024 baseline: Proposition extraction + Judge matching + NN classifier.

Reproduces the approach from Paper 27 (COLIEE 2024 runner-up).
Stages:
  1. Extract "propositions" (text windows around <FRAGMENT_SUPPRESSED> markers)
  2. Encode propositions with sentence-transformers (all-MiniLM-L6-v2)
  3. Extract judge names and compute IDF-weighted overlap
  4. Compute 5-gram quotation overlap
  5. Train a feedforward neural network on 8 features
"""
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baselines.common.base_model import BaselineModel
from baselines.umnlp.propositions import PropositionExtractor
from baselines.umnlp.judge_match import JudgeMatcher
from baselines.umnlp.features import extract_umnlp_features

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "proposition_sim",
    "judge_match",
    "judge_rarity_score",
    "quotation_overlap",
    "para_max_sim",
    "para_mean_sim",
    "bm25_score",
    "doc_len_ratio",
]


class PropositionNN(nn.Module):
    """Simple feedforward binary classifier over hand-crafted features."""

    def __init__(self, input_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """Returns raw logits (no sigmoid) for BCEWithLogitsLoss."""
        return self.net(x).squeeze(-1)

    def predict_proba(self, x):
        """Returns probabilities for inference."""
        return torch.sigmoid(self.forward(x))


class UMNLPBaseline(BaselineModel):
    """UMNLP 2024 runner-up approach for COLIEE Task 1."""

    def __init__(self):
        self._prop_extractor = PropositionExtractor()
        self._judge_matcher = JudgeMatcher()
        self._model: PropositionNN | None = None
        self._prop_embeddings: dict[str, np.ndarray] = {}

    def name(self) -> str:
        return "UMNLP 2024 (Propositions+NN)"

    def train(self, corpus, train_queries, labels, bm25_candidates=None):
        if bm25_candidates is None:
            raise ValueError("UMNLP baseline requires pre-computed BM25 candidates")

        # 1. Encode propositions for all docs
        self._prop_embeddings = self._prop_extractor.encode_propositions(corpus)

        # 2. Build judge index
        self._judge_matcher.build_index(corpus)

        # 3. Build training data (with negative subsampling)
        logger.info("Extracting training features for %d queries...", len(train_queries))
        rng = np.random.RandomState(42)
        max_neg_ratio = 10
        max_neg_per_query = 50

        X_rows = []
        y_rows = []
        for qi, qid in enumerate(train_queries):
            candidates = bm25_candidates.get(qid, [])
            positives = set(labels.get(qid, []))

            pos_pairs = []
            neg_pairs = []
            for cid, _ in candidates:
                if cid == qid:
                    continue
                feats = extract_umnlp_features(
                    qid, cid, corpus, bm25_candidates,
                    self._prop_embeddings, self._judge_matcher,
                )
                feat_vec = [feats[f] for f in FEATURE_NAMES]
                if cid in positives:
                    pos_pairs.append((feat_vec, 1.0))
                else:
                    neg_pairs.append((feat_vec, 0.0))

            # Subsample negatives
            n_neg = min(len(neg_pairs), max(len(pos_pairs) * max_neg_ratio, 1), max_neg_per_query)
            if len(neg_pairs) > n_neg:
                idx = rng.choice(len(neg_pairs), n_neg, replace=False)
                neg_pairs = [neg_pairs[i] for i in idx]

            for feat_vec, lbl in pos_pairs + neg_pairs:
                X_rows.append(feat_vec)
                y_rows.append(lbl)
            if (qi + 1) % 500 == 0:
                logger.info("  Features: %d/%d queries", qi + 1, len(train_queries))

        X = np.array(X_rows, dtype=np.float32)
        y = np.array(y_rows, dtype=np.float32)
        logger.info("Training UMNLP NN on %d samples (%d pos, %.2f%%)",
                     len(y), int(y.sum()), 100 * y.mean() if len(y) > 0 else 0)

        # 4. Train NN with class-weighted loss
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = PropositionNN(input_dim=len(FEATURE_NAMES)).to(device)

        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(dataset, batch_size=256, shuffle=True)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = n_neg / max(n_pos, 1)
        weight_tensor = torch.tensor([pos_weight], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

        self._model.train()
        for epoch in range(20):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = self._model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                logger.info("Epoch %d/%d loss=%.4f",
                            epoch + 1, 20, total_loss / max(len(loader), 1))

    def predict(self, query_id, corpus, bm25_candidates=None):
        if self._model is None:
            raise RuntimeError("Model not trained")
        if bm25_candidates is None:
            return []

        # bm25_candidates is a list for this query (per BaselineModel interface)
        cands = bm25_candidates if isinstance(bm25_candidates, list) else []
        if not cands:
            return []

        # Wrap into dict for feature extraction
        all_cands = {query_id: cands}

        X_rows = []
        cand_ids = []
        for cid, _ in cands:
            if cid == query_id:
                continue
            feats = extract_umnlp_features(
                query_id, cid, corpus, all_cands,
                self._prop_embeddings, self._judge_matcher,
            )
            X_rows.append([feats[f] for f in FEATURE_NAMES])
            cand_ids.append(cid)

        if not X_rows:
            return []

        device = next(self._model.parameters()).device
        X_tensor = torch.tensor(X_rows, dtype=torch.float32).to(device)

        self._model.training = False
        with torch.no_grad():
            scores = self._model.predict_proba(X_tensor).cpu().numpy()

        results = list(zip(cand_ids, scores.tolist()))
        results.sort(key=lambda x: x[1], reverse=True)
        return results
