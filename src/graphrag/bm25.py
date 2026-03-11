"""BM25 index for first-stage retrieval."""
import logging
import re
from typing import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """BM25 index over a corpus of documents."""

    def __init__(self):
        self._index: BM25Okapi | None = None
        self._doc_ids: list[str] = []

    def fit(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        """Build BM25 index."""
        self._doc_ids = list(doc_ids)
        tokenized = [tokenize(t) for t in texts]
        self._index = BM25Okapi(tokenized)
        logger.info("BM25 index built: %d documents", len(doc_ids))

    def query(self, text: str, top_k: int = 200) -> list[tuple[str, float]]:
        """Query the index. Returns list of (doc_id, score) sorted by score descending."""
        assert self._index is not None, "Index not built. Call fit() first."
        tokens = tokenize(text)
        scores = self._index.get_scores(tokens)
        top_indices = np.argsort(-scores)[:top_k]
        return [(self._doc_ids[i], float(scores[i])) for i in top_indices]

    def get_all_neighbors(self, texts: dict[str, str], top_k: int = 20) -> dict[str, list[tuple[str, float]]]:
        """Get top-k BM25 neighbors for every document in the corpus.

        Args:
            texts: dict mapping doc_id -> text (must match fit() doc_ids)
            top_k: number of neighbors per document

        Returns:
            dict mapping doc_id -> [(neighbor_id, score), ...]
        """
        assert self._index is not None
        neighbors = {}
        for doc_id in self._doc_ids:
            results = self.query(texts[doc_id], top_k=top_k + 1)
            # Filter out self
            neighbors[doc_id] = [(did, s) for did, s in results if did != doc_id][:top_k]
        return neighbors
