"""W1-C1: BM25 k1/b re-tuning to lift the first-stage recall ceiling.

Re-tunes BM25 k1 and b and measures pool recall@{50,100,200} of the multi-view
RRF candidate pool against gold, micro-averaged over all 2001 train queries.

Optimization: tokenize the corpus ONCE. The tokenization, document frequency,
IDF, and doc lengths are independent of (k1, b) -- only the norm_tf TF-saturation
term depends on them. So per grid point we only recompute the sparse tf_norm
matrix (cheap) and reuse the same vocab / idf / doc_lens / per-query tokens.

Run:  uv run python scripts/w1_c1_bm25_sweep.py
"""
import json
import logging
import pickle  # noqa: S403 -- internal cache only
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from coliee_task1.config import BM25_TOP_K, BM25_CONTEXT_TOP_K, RRF_K  # noqa: E402
from coliee_task1.stages.bm25 import tokenize, rrf_fuse  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("w1_c1")

STAGE1_PKL = REPO / "output" / "pipeline_cache" / "stage1.pkl"
LABELS_PATH = REPO / "data" / "task1" / "task1_train_labels_2026.json"
OUT_PATH = REPO / "output" / "w1" / "c1_full.json"

K1_GRID = [1.5, 4.5]
B_GRID = [0.75, 1.0]
QUERY_SUBSAMPLE = 0  # full set
RECALL_KS = [50, 100, 200]


def load_stage1():
    """Load stage1 cache, aliasing the legacy 'graphrag' module path."""
    import coliee_task1
    import coliee_task1.stages.citation_context as cc
    sys.modules.setdefault("graphrag", coliee_task1)
    sys.modules.setdefault("graphrag.stages", coliee_task1.stages)
    sys.modules.setdefault("graphrag.citation_context", cc)
    sys.modules.setdefault("graphrag.stages.citation_context", cc)
    with open(STAGE1_PKL, "rb") as f:
        raw_corpus, clean_corpus, contexts = pickle.load(f)  # noqa: S301
    return raw_corpus, clean_corpus, contexts


class PrebuiltBM25:
    """BM25 index that tokenizes the corpus once and recomputes only the
    norm_tf sparse matrix per (k1, b). Reuses vocab / idf / doc_lens.

    Query scoring matches BM25Index._score_tokens / .query exactly.
    """

    def __init__(self, doc_ids, texts):
        self.doc_ids = list(doc_ids)
        n_docs = len(doc_ids)

        t0 = time.time()
        tokenized = [tokenize(t) for t in texts]
        logger.info("Tokenized %d docs in %.1fs", n_docs, time.time() - t0)

        vocab: dict[str, int] = {}
        for tokens in tokenized:
            for tok in tokens:
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        self.vocab = vocab
        n_terms = len(vocab)

        doc_lens = np.array([len(t) for t in tokenized], dtype=np.float32)
        self.avgdl = float(doc_lens.mean()) if n_docs > 0 else 1.0
        self.doc_lens = doc_lens

        # Precompute the COO triplets that are INVARIANT to (k1, b): (row, col, count).
        rows, cols, counts = [], [], []
        df = np.zeros(n_terms, dtype=np.float32)
        for doc_idx, tokens in enumerate(tokenized):
            tf = Counter(tokens)
            for term, count in tf.items():
                tid = vocab[term]
                rows.append(doc_idx)
                cols.append(tid)
                counts.append(count)
                df[tid] += 1
        self._rows = np.asarray(rows, dtype=np.int32)
        self._cols = np.asarray(cols, dtype=np.int32)
        self._counts = np.asarray(counts, dtype=np.float32)
        self._row_doc_lens = doc_lens[self._rows]  # dl for each nonzero entry
        self.n_docs = n_docs
        self.n_terms = n_terms

        # IDF is independent of (k1, b).
        self.idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1).astype(np.float32)
        logger.info("Index prepped: %d docs, %d terms, avgdl=%.0f, nnz=%d",
                    n_docs, n_terms, self.avgdl, len(self._counts))

        self.tf_norm = None  # set by set_params

    def set_params(self, k1: float, b: float):
        """Recompute the norm_tf sparse matrix for given (k1, b)."""
        cnt = self._counts
        norm = (cnt * (k1 + 1)) / (
            cnt + k1 * (1 - b + b * self._row_doc_lens / self.avgdl)
        )
        self.tf_norm = sparse.csr_matrix(
            (norm, (self._rows, self._cols)),
            shape=(self.n_docs, self.n_terms),
            dtype=np.float32,
        )

    def _score_tokens(self, tokens):
        term_ids = [self.vocab[t] for t in tokens if t in self.vocab]
        if not term_ids:
            return np.zeros(self.n_docs, dtype=np.float32)
        query_idf = self.idf[term_ids]
        tf_slice = self.tf_norm[:, term_ids]
        scores = tf_slice.dot(query_idf)
        return np.asarray(scores).flatten()

    def query_tokens(self, tokens, top_k):
        scores = self._score_tokens(tokens)
        k = min(top_k, len(scores) - 1)
        top = np.argpartition(-scores, k)[:top_k]
        top = top[np.argsort(-scores[top])]
        return [(self.doc_ids[i], float(scores[i])) for i in top]


def main():
    t_start = time.time()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading stage1 cache ...")
    _, clean_corpus, contexts = load_stage1()

    labels = json.load(open(LABELS_PATH))
    query_ids = sorted(labels.keys())
    if QUERY_SUBSAMPLE and QUERY_SUBSAMPLE < len(query_ids):
        import random as _random
        query_ids = sorted(_random.Random(42).sample(query_ids, QUERY_SUBSAMPLE))
    total_pos = sum(len(labels[q]) for q in query_ids)
    logger.info("Queries=%d (subsample=%s), total gold positives=%d",
                len(query_ids), QUERY_SUBSAMPLE, total_pos)

    # Build index over full clean corpus (same as stage2: sorted doc ids).
    doc_ids = sorted(clean_corpus.keys())
    texts = [clean_corpus[did] for did in doc_ids]
    index = PrebuiltBM25(doc_ids, texts)

    # Precompute per-query token lists ONCE (full doc + each context window).
    logger.info("Pre-tokenizing queries ...")
    q_full_tokens = {}
    q_ctx_tokens = {}
    gold_sets = {}
    for qid in query_ids:
        if qid not in clean_corpus:
            q_full_tokens[qid] = []
            q_ctx_tokens[qid] = []
            gold_sets[qid] = set(labels.get(qid, []))
            continue
        q_full_tokens[qid] = tokenize(clean_corpus[qid])
        dc = contexts.get(qid)
        ctx_texts = [c.text for c in dc.contexts] if dc else []
        q_ctx_tokens[qid] = [tokenize(c) for c in ctx_texts]
        gold_sets[qid] = set(labels.get(qid, []))

    grid_results = {}
    for k1 in K1_GRID:
        for b in B_GRID:
            tg = time.time()
            index.set_params(k1, b)
            # Accumulate hits @ each recall K.
            hits = {rk: 0 for rk in RECALL_KS}
            for qid in query_ids:
                gold = gold_sets[qid]
                if not gold:
                    continue
                ft = q_full_tokens[qid]
                if not ft:
                    continue
                ranked_lists = []
                full_res = index.query_tokens(ft, top_k=BM25_TOP_K)
                full_res = [(d, s) for d, s in full_res if d != qid]
                ranked_lists.append(full_res)
                for ct in q_ctx_tokens[qid]:
                    ctx_res = index.query_tokens(ct, top_k=BM25_CONTEXT_TOP_K)
                    ctx_res = [(d, s) for d, s in ctx_res if d != qid]
                    ranked_lists.append(ctx_res)
                # Fuse to top-200 (RRF k=60), then slice for each recall K.
                fused = rrf_fuse(ranked_lists, k=RRF_K, top_k=BM25_TOP_K)
                pool_ids = [d for d, _ in fused]
                for rk in RECALL_KS:
                    pool = set(pool_ids[:rk])
                    hits[rk] += len(gold & pool)
            rec = {f"recall@{rk}": hits[rk] / total_pos for rk in RECALL_KS}
            rec["hits@200"] = hits[200]
            key = f"k1={k1}_b={b}"
            grid_results[key] = {"k1": k1, "b": b, **rec}
            logger.info(
                "%s -> R@50=%.4f R@100=%.4f R@200=%.4f (%d/%d) [%.0fs]",
                key, rec["recall@50"], rec["recall@100"], rec["recall@200"],
                hits[200], total_pos, time.time() - tg,
            )

    # Sanity check.
    base = grid_results["k1=1.5_b=0.75"]
    sanity_ok = abs(base["recall@200"] - 0.578) <= 0.01
    logger.info(
        "SANITY CHECK k1=1.5,b=0.75 recall@200=%.4f (expected ~0.578, ok=%s)",
        base["recall@200"], sanity_ok,
    )

    # Best by recall@200.
    best_key = max(grid_results, key=lambda kk: grid_results[kk]["recall@200"])
    best = grid_results[best_key]

    out = {
        "baseline_default": {"k1": 1.5, "b": 0.75, **{f"recall@{rk}": base[f"recall@{rk}"] for rk in RECALL_KS}},
        "sanity_check": {
            "expected_recall@200": 0.578,
            "actual_recall@200": base["recall@200"],
            "abs_diff": abs(base["recall@200"] - 0.578),
            "passed": bool(sanity_ok),
        },
        "best": {"config": best_key, **{kk: best[kk] for kk in ("k1", "b", "recall@50", "recall@100", "recall@200")}},
        "lift_over_baseline": {
            "abs_recall@200": best["recall@200"] - base["recall@200"],
            "rel_recall@200": (best["recall@200"] - base["recall@200"]) / base["recall@200"] if base["recall@200"] else 0.0,
        },
        "total_gold": total_pos,
        "n_queries": len(query_ids),
        "grid": grid_results,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Saved -> %s", OUT_PATH)

    # Print table.
    print("\n=== W1-C1 BM25 k1/b sweep: recall@200 (rows=k1, cols=b) ===")
    header = "k1\\b   " + "".join(f"{b:>10.2f}" for b in B_GRID)
    print(header)
    for k1 in K1_GRID:
        row = f"{k1:>5.1f} "
        for b in B_GRID:
            row += f"{grid_results[f'k1={k1}_b={b}']['recall@200']:>10.4f}"
        print(row)

    print("\n=== Full grid (recall@50 / @100 / @200) ===")
    print(f"{'config':<16}{'R@50':>9}{'R@100':>9}{'R@200':>9}")
    for k1 in K1_GRID:
        for b in B_GRID:
            g = grid_results[f"k1={k1}_b={b}"]
            print(f"{'k1=%s b=%s' % (k1, b):<16}{g['recall@50']:>9.4f}{g['recall@100']:>9.4f}{g['recall@200']:>9.4f}")

    print(f"\nSANITY (k1=1.5,b=0.75) recall@200 = {base['recall@200']:.4f} "
          f"(expected ~0.578, diff={abs(base['recall@200']-0.578):.4f}, "
          f"{'PASS' if sanity_ok else 'FAIL'})")
    print(f"BEST: {best_key} -> R@50={best['recall@50']:.4f} "
          f"R@100={best['recall@100']:.4f} R@200={best['recall@200']:.4f}")
    print(f"LIFT @200 over 0.578 baseline: "
          f"abs={best['recall@200']-base['recall@200']:+.4f} "
          f"rel={100*(best['recall@200']-base['recall@200'])/base['recall@200']:+.2f}%")
    print(f"\nTotal runtime: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
