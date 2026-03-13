"""Vanilla BM25 baseline — establishes the retrieval floor."""
from baselines.common.base_model import BaselineModel
from graphrag.bm25 import BM25Index


class VanillaBM25(BaselineModel):
    """Pure BM25 retrieval with threshold optimization."""

    def __init__(self, top_k: int = 200):
        self._top_k = top_k
        self._bm25: BM25Index | None = None

    def name(self) -> str:
        return "BM25 (vanilla)"

    def train(self, corpus, train_queries, labels, bm25_candidates=None):
        doc_ids = sorted(corpus.keys())
        self._bm25 = BM25Index()
        self._bm25.fit(doc_ids, [corpus[d] for d in doc_ids])

    def predict(self, query_id, corpus, bm25_candidates=None):
        results = self._bm25.query(corpus[query_id], top_k=self._top_k + 1)
        return [(did, s) for did, s in results if did != query_id][:self._top_k]
