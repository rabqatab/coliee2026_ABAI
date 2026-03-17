"""BM25 index for first-stage retrieval with multi-view support.

Uses scipy sparse matrices for vectorized BM25 scoring (~100x faster than
rank_bm25 for large corpora). Supports full-document and per-citation-context
retrieval with Reciprocal Rank Fusion (RRF).
"""
import logging
import re
from collections import Counter
from typing import Sequence

import numpy as np
from scipy import sparse

from coliee_task1.config import BM25_TOP_K, BM25_CONTEXT_TOP_K, RRF_K

logger = logging.getLogger(__name__)

# BM25 parameters
BM25_K1 = 1.5
BM25_B = 0.75


def tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return re.findall(r"\w+", text.lower())


def rrf_fuse(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = RRF_K,
    top_k: int = BM25_TOP_K,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over multiple ranked lists.

    For each document d across all lists:
        RRF(d) = sum(1 / (k + rank_i(d))) for each list i where d appears
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    sorted_results = sorted(scores.items(), key=lambda x: -x[1])
    return sorted_results[:top_k]


def convex_fuse(
    full_doc_scores: dict[str, float],
    context_scores: dict[str, float],
    alpha: float = 0.5,
    top_k: int = BM25_TOP_K,
) -> list[tuple[str, float]]:
    """Convex combination fusion of full-doc and context-window BM25 scores.

    score(d) = α · norm(full_doc_score(d)) + (1-α) · norm(context_score(d))

    Unlike RRF, preserves score magnitudes after normalization.
    """
    # Min-max normalize each score distribution
    def _minmax(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        vals = list(scores.values())
        lo, hi = min(vals), max(vals)
        rng = hi - lo
        if rng < 1e-10:
            return {k: 0.5 for k in scores}
        return {k: (v - lo) / rng for k, v in scores.items()}

    norm_full = _minmax(full_doc_scores)
    norm_ctx = _minmax(context_scores)

    # Combine
    all_docs = set(norm_full) | set(norm_ctx)
    combined = {}
    for doc_id in all_docs:
        s_full = norm_full.get(doc_id, 0.0)
        s_ctx = norm_ctx.get(doc_id, 0.0)
        combined[doc_id] = alpha * s_full + (1 - alpha) * s_ctx

    sorted_results = sorted(combined.items(), key=lambda x: -x[1])
    return sorted_results[:top_k]


def tune_convex_alpha(
    bm25_index: "BM25Index",
    query_texts: dict[str, str],
    context_texts: dict[str, list[str]],
    labels: dict[str, list[str]],
    top_k: int = BM25_TOP_K,
    alphas: list[float] | None = None,
) -> float:
    """Grid search for optimal convex combination alpha on training recall@K.

    Returns best alpha value.
    """
    if alphas is None:
        alphas = [i / 20 for i in range(21)]  # 0.0 to 1.0 in 0.05 steps

    best_alpha = 0.5
    best_recall = 0.0
    total_positives = sum(len(v) for v in labels.values())

    for alpha in alphas:
        hits = 0
        for qid, q_text in query_texts.items():
            gold = set(labels.get(qid, []))
            if not gold:
                continue

            # Full-doc scores
            full_results = bm25_index.query(q_text, top_k=top_k * 2)
            full_scores = {did: s for did, s in full_results if did != qid}

            # Context-window scores (max per candidate across windows)
            ctx_scores: dict[str, float] = {}
            for ctx in context_texts.get(qid, []):
                for did, s in bm25_index.query(ctx, top_k=BM25_CONTEXT_TOP_K):
                    if did != qid:
                        ctx_scores[did] = max(ctx_scores.get(did, 0.0), s)

            fused = convex_fuse(full_scores, ctx_scores, alpha=alpha, top_k=top_k)
            retrieved = {did for did, _ in fused}
            hits += len(gold & retrieved)

        recall = hits / total_positives if total_positives > 0 else 0.0
        if recall > best_recall:
            best_recall = recall
            best_alpha = alpha

    logger.info(
        "Convex alpha tuning: best alpha=%.2f, recall@%d=%.4f",
        best_alpha, top_k, best_recall,
    )
    return best_alpha


class BM25Index:
    """Fast BM25 index using scipy sparse matrices.

    Precomputes TF and IDF components so each query is a single
    sparse matrix multiplication — orders of magnitude faster than
    rank_bm25 for repeated queries on the same corpus.
    """

    def __init__(self):
        self._doc_ids: list[str] = []
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray | None = None
        self._tf_norm: sparse.csr_matrix | None = None  # (n_docs, n_terms)
        self._avgdl: float = 0.0
        self._doc_lens: np.ndarray | None = None

    def fit(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        """Build BM25 index from corpus."""
        self._doc_ids = list(doc_ids)
        n_docs = len(doc_ids)

        # Tokenize and build vocabulary
        tokenized = [tokenize(t) for t in texts]
        vocab: dict[str, int] = {}
        for tokens in tokenized:
            for t in tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)
        self._vocab = vocab
        n_terms = len(vocab)

        # Build TF matrix (sparse)
        doc_lens = np.array([len(t) for t in tokenized], dtype=np.float32)
        self._avgdl = float(doc_lens.mean()) if n_docs > 0 else 1.0
        self._doc_lens = doc_lens

        # CSR construction: collect (row, col, data) triplets
        rows, cols, data = [], [], []
        df = np.zeros(n_terms, dtype=np.float32)  # document frequency

        for doc_idx, tokens in enumerate(tokenized):
            tf = Counter(tokens)
            for term, count in tf.items():
                tid = vocab[term]
                rows.append(doc_idx)
                cols.append(tid)
                # BM25 TF normalization: tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl/avgdl))
                norm_tf = (count * (BM25_K1 + 1)) / (
                    count + BM25_K1 * (1 - BM25_B + BM25_B * doc_lens[doc_idx] / self._avgdl)
                )
                data.append(norm_tf)
                df[tid] += 1

        self._tf_norm = sparse.csr_matrix(
            (data, (rows, cols)), shape=(n_docs, n_terms), dtype=np.float32,
        )

        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        self._idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1).astype(np.float32)

        logger.info(
            "BM25 index built: %d documents, %d terms, avgdl=%.0f",
            n_docs, n_terms, self._avgdl,
        )

    def _score_tokens(self, tokens: list[str]) -> np.ndarray:
        """Score all documents for a tokenized query. Returns (n_docs,) array."""
        # Build query term vector (binary — which terms appear)
        term_ids = [self._vocab[t] for t in tokens if t in self._vocab]
        if not term_ids:
            return np.zeros(len(self._doc_ids), dtype=np.float32)

        # Gather IDF weights for query terms
        query_idf = self._idf[term_ids]  # (n_query_terms,)

        # Gather TF columns for query terms
        tf_slice = self._tf_norm[:, term_ids]  # (n_docs, n_query_terms) sparse

        # BM25 score = sum(idf * tf_norm) for each document
        scores = tf_slice.dot(query_idf)  # (n_docs,) dense

        return np.asarray(scores).flatten()

    def query(self, text: str, top_k: int = BM25_TOP_K) -> list[tuple[str, float]]:
        """Query the index. Returns list of (doc_id, score) sorted by score descending."""
        assert self._idf is not None, "Index not built. Call fit() first."
        tokens = tokenize(text)
        scores = self._score_tokens(tokens)
        top_indices = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [(self._doc_ids[i], float(scores[i])) for i in top_indices]

    def query_batch(
        self, texts: list[str], top_k: int = BM25_TOP_K,
    ) -> list[list[tuple[str, float]]]:
        """Query multiple texts at once. Returns list of ranked results per query."""
        return [self.query(t, top_k) for t in texts]

    def query_multiview(
        self,
        full_text: str,
        context_windows: list[str],
        full_top_k: int = BM25_TOP_K,
        context_top_k: int = BM25_CONTEXT_TOP_K,
        rrf_top_k: int = BM25_TOP_K,
        exclude_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """Multi-view BM25: fuse full-document and per-context-window queries via RRF."""
        ranked_lists = []

        # View 1: Full document
        full_results = self.query(full_text, top_k=full_top_k)
        if exclude_id:
            full_results = [(d, s) for d, s in full_results if d != exclude_id]
        ranked_lists.append(full_results)

        # View 2: Per-context-window queries
        for ctx in context_windows:
            ctx_results = self.query(ctx, top_k=context_top_k)
            if exclude_id:
                ctx_results = [(d, s) for d, s in ctx_results if d != exclude_id]
            ranked_lists.append(ctx_results)

        # Fuse all views
        fused = rrf_fuse(ranked_lists, top_k=rrf_top_k)

        return fused

    def query_multiview_convex(
        self,
        full_text: str,
        context_windows: list[str],
        alpha: float = 0.5,
        top_k: int = BM25_TOP_K,
        exclude_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """Multi-view BM25 with convex combination fusion (Option 3).

        Instead of RRF (rank-based), uses score-level fusion:
        score(d) = α · norm(full_doc(d)) + (1-α) · norm(max_context(d))
        """
        # Full-doc scores
        full_results = self.query(full_text, top_k=top_k * 2)
        full_scores = {
            did: s for did, s in full_results
            if did != exclude_id
        }

        # Context-window scores: max per candidate across all windows
        ctx_scores: dict[str, float] = {}
        for ctx in context_windows:
            for did, s in self.query(ctx, top_k=BM25_CONTEXT_TOP_K):
                if did != exclude_id:
                    ctx_scores[did] = max(ctx_scores.get(did, 0.0), s)

        return convex_fuse(full_scores, ctx_scores, alpha=alpha, top_k=top_k)

    def get_all_neighbors(self, texts: dict[str, str], top_k: int = 20) -> dict[str, list[tuple[str, float]]]:
        """Get top-k BM25 neighbors for every document in the corpus."""
        assert self._idf is not None
        neighbors = {}
        for doc_id in self._doc_ids:
            results = self.query(texts[doc_id], top_k=top_k + 1)
            neighbors[doc_id] = [(did, s) for did, s in results if did != doc_id][:top_k]
        return neighbors
