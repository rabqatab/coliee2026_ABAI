"""Abstract base class for all baseline models."""
from abc import ABC, abstractmethod


class BaselineModel(ABC):
    """Interface that every baseline implements."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable name for comparison tables."""
        ...

    @abstractmethod
    def train(
        self,
        corpus: dict[str, str],
        train_queries: list[str],
        labels: dict[str, list[str]],
        bm25_candidates: dict[str, list[tuple[str, float]]] | None = None,
    ) -> None:
        """Train/fit the model on training data."""
        ...

    @abstractmethod
    def predict(
        self,
        query_id: str,
        corpus: dict[str, str],
        bm25_candidates: list[tuple[str, float]] | None = None,
    ) -> list[tuple[str, float]]:
        """Return ranked list of (candidate_id, relevance_score).

        Must exclude query_id from results.
        """
        ...

    def predict_batch(
        self,
        query_ids: list[str],
        corpus: dict[str, str],
        bm25_candidates: dict[str, list[tuple[str, float]]] | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """Predict for multiple queries. Override for batch-optimized models."""
        results = {}
        for qid in query_ids:
            cands = bm25_candidates.get(qid) if bm25_candidates else None
            results[qid] = self.predict(qid, corpus, cands)
        return results
