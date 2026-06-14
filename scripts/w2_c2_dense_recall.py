"""W2-C2: dense first-stage retrieval to lift the recall ceiling.

Our BM25-RRF first stage retrieves only ~57.8% of gold into the top-200 pool.
This experiment measures how much a strong instruction-tuned dense embedder
lifts standalone Recall@{50,100,200} and how much an RRF fusion of the dense
ranking with the cached BM25-RRF ranking lifts it further.

Three rankings are compared, all evaluated against gold with micro-averaged
recall:
    1. bm25_rrf  -- the cached BM25-RRF baseline (from stage2 rrf_results)
    2. dense     -- cosine top-200 from the dense embedder, query id excluded
    3. fused     -- RRF (k=60) of dense and bm25_rrf rankings

Pooling is detected per model:
    * e5-mistral (default) -> last-token pooling, query instruction prefix
    * bge / gte / generic  -> CLS or mean pooling (sentence-transformers)

Hardware note (GB10 / Blackwell SM121): FP8 kernels are broken; use bf16 on
GPU. Do NOT enable FP8 or flash-attn-fp8.

Full GPU run (default e5-mistral-7b):
    HF_HOME=/mnt/nfs/ssd1/huggingface_cache \
    uv run python scripts/w2_c2_dense_recall.py

Local CPU smoke (tiny bge model, ~75 docs / 15 queries):
    uv run python scripts/w2_c2_dense_recall.py --smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle  # noqa: S403 -- internal cache only
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("w2_c2")

STAGE1_PKL = REPO / "output" / "pipeline_cache" / "stage1.pkl"
STAGE2_PKL = REPO / "output" / "pipeline_cache" / "stage2.pkl"
LABELS_PATH = REPO / "data" / "task1" / "task1_train_labels_2026.json"
OUT_PATH = REPO / "output" / "w2" / "c2_dense_recall.json"

DEFAULT_MODEL = "intfloat/e5-mistral-7b-instruct"
SMOKE_MODEL = "BAAI/bge-small-en-v1.5"

RECALL_KS = [50, 100, 200]
TOP_K = 200
RRF_K = 60

# Instruction prepended to QUERIES ONLY for instruction-tuned models.
QUERY_INSTRUCTION = (
    "Instruct: Given a legal case, retrieve prior cases it cites\nQuery: "
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
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


def load_bm25_rrf():
    """Return rrf_results = {query_id: [(cand_id, score), ...]}."""
    with open(STAGE2_PKL, "rb") as f:
        rrf_results, _bm25_raw, _bm25_rrf, _ctx_feats = pickle.load(f)  # noqa: S301
    return rrf_results


def load_labels():
    with open(LABELS_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------
def detect_pooling(model_name: str) -> str:
    """Return 'last' for instruction LLM embedders (e5-mistral etc.),
    else 'mean' (bge / gte / generic sentence-transformers)."""
    n = model_name.lower()
    if "e5-mistral" in n or "e5mistral" in n:
        return "last"
    # Most LLM-based embedders use last-token pooling.
    if any(t in n for t in ("mistral", "qwen", "llama", "gritlm", "sfr-embedding")):
        return "last"
    return "mean"


def uses_query_instruction(model_name: str, pooling: str) -> bool:
    """Instruction-tuned LLM embedders take a query instruction prefix."""
    return pooling == "last"


def last_token_pool(last_hidden, attention_mask):
    """Last non-pad token pooling (handles left- and right-padding)."""
    import torch

    left_padded = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padded:
        return last_hidden[:, -1]
    seq_len = attention_mask.sum(dim=1) - 1
    bs = last_hidden.shape[0]
    return last_hidden[torch.arange(bs, device=last_hidden.device), seq_len]


def mean_pool(last_hidden, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def embed_texts(
    texts: list[str],
    model_name: str,
    pooling: str,
    is_query: bool,
    max_length: int,
    batch_size: int,
    device: str,
    use_instruction: bool,
) -> np.ndarray:
    """Embed texts with a HF model, L2-normalized. Returns (n, d) float32."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    if is_query and use_instruction:
        texts = [QUERY_INSTRUCTION + t for t in texts]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()

    embs: list[np.ndarray] = []
    n = len(texts)
    t0 = time.time()
    for start in range(0, n, batch_size):
        batch = texts[start : start + batch_size]
        if pooling == "last":
            # e5-mistral recipe: truncate to max_len-1, APPEND eos, then pad.
            # The embedding forms at the EOS position; omitting it -> degenerate (near-random).
            bd = tokenizer(
                batch, max_length=max_length - 1, truncation=True,
                padding=False, add_special_tokens=True, return_attention_mask=False,
            )
            bd["input_ids"] = [ids + [tokenizer.eos_token_id] for ids in bd["input_ids"]]
            enc = tokenizer.pad(
                bd, padding=True, return_attention_mask=True, return_tensors="pt",
            ).to(device)
        else:
            enc = tokenizer(
                batch,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
        with torch.no_grad():
            out = model(**enc)
            hidden = out.last_hidden_state
            if pooling == "last":
                pooled = last_token_pool(hidden, enc["attention_mask"])
            else:
                pooled = mean_pool(hidden, enc["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        embs.append(pooled.float().cpu().numpy())
        done = min(start + batch_size, n)
        if done % (batch_size * 20) == 0 or done == n:
            logger.info("  embedded %d/%d (%.1fs)", done, n, time.time() - t0)
    return np.vstack(embs).astype(np.float32)


# ---------------------------------------------------------------------------
# Retrieval + fusion
# ---------------------------------------------------------------------------
def dense_topk(
    query_embs: np.ndarray,
    corpus_embs: np.ndarray,
    corpus_ids: list[str],
    query_ids: list[str],
    top_k: int,
) -> dict[str, list[tuple[str, float]]]:
    """Cosine top-k per query (embeddings already L2-normalized so cosine ==
    dot product). Excludes the query's own id from its ranking."""
    id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
    results: dict[str, list[tuple[str, float]]] = {}
    for qi, qid in enumerate(query_ids):
        sims = corpus_embs @ query_embs[qi]  # (n_corpus,)
        self_idx = id_to_idx.get(qid)
        if self_idx is not None:
            sims[self_idx] = -np.inf
        # top_k + 1 guards against the (excluded) self appearing; then trim.
        k = min(top_k, sims.shape[0])
        part = np.argpartition(-sims, k - 1)[:k]
        order = part[np.argsort(-sims[part])]
        results[qid] = [
            (corpus_ids[i], float(sims[i])) for i in order if np.isfinite(sims[i])
        ]
    return results


def rrf_fuse_two(
    ranking_a: list[tuple[str, float]],
    ranking_b: list[tuple[str, float]],
    k: int,
    top_k: int,
) -> list[tuple[str, float]]:
    """RRF over two ranked lists -> fused [(id, score)], truncated to top_k."""
    scores: dict[str, float] = {}
    for ranked in (ranking_a, ranking_b):
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])[:top_k]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def micro_recall_at_ks(
    rankings: dict[str, list[tuple[str, float]]],
    labels: dict[str, list[str]],
    query_ids: list[str],
    ks: list[int],
) -> dict[str, float]:
    """Micro-averaged recall@k: sum(hits@k) / sum(gold) over queries."""
    out: dict[str, float] = {}
    for k in ks:
        total_gold = 0
        total_hit = 0
        for qid in query_ids:
            gold = set(labels.get(qid, []))
            if not gold:
                continue
            topk_ids = [cid for cid, _ in rankings.get(qid, [])[:k]]
            total_gold += len(gold)
            total_hit += len(gold.intersection(topk_ids))
        out[f"R@{k}"] = total_hit / total_gold if total_gold else 0.0
    return out


# ---------------------------------------------------------------------------
# Smoke subset selection
# ---------------------------------------------------------------------------
def select_smoke_subset(labels, clean_corpus, rrf_results, n_queries=15, n_corpus=60):
    """Pick a small set of queries (with gold present in corpus) and a corpus
    restricted to those queries + their gold + their BM25-RRF pool, capped so
    the embed->topk->recall->RRF path runs on CPU in well under 3 min."""
    chosen_q: list[str] = []
    for qid in labels:
        if qid not in clean_corpus:
            continue
        gold = [g for g in labels[qid] if g in clean_corpus]
        if gold:
            chosen_q.append(qid)
        if len(chosen_q) >= n_queries:
            break

    corpus_ids: list[str] = []
    seen = set()

    def add(cid):
        if cid in clean_corpus and cid not in seen:
            seen.add(cid)
            corpus_ids.append(cid)

    # Always include queries and their gold first so recall can be non-zero.
    for qid in chosen_q:
        add(qid)
        for g in labels[qid]:
            add(g)
    # Pad with BM25-RRF pool members (distractors) up to the cap.
    for qid in chosen_q:
        for cid, _ in rrf_results.get(qid, []):
            if len(corpus_ids) >= max(n_corpus, len(corpus_ids)):
                break
            add(cid)
        if len(corpus_ids) >= n_corpus and len(corpus_ids) >= len(chosen_q) * 2:
            break
    return chosen_q, corpus_ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL", DEFAULT_MODEL),
        help="HF model id (env MODEL overrides default).",
    )
    parser.add_argument("--smoke", action="store_true", help="tiny CPU validation run")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda|cpu (auto if unset)")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    import torch

    model_name = SMOKE_MODEL if args.smoke else args.model
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() and not args.smoke else "cpu"

    pooling = detect_pooling(model_name)
    use_instr = uses_query_instruction(model_name, pooling)

    # Defaults: e5-mistral wants 4096 ctx; bge tiny model uses 512.
    if args.max_length is not None:
        max_length = args.max_length
    elif pooling == "last":
        max_length = 4096
    else:
        max_length = 512
    if args.batch_size is not None:
        batch_size = args.batch_size
    elif args.smoke:
        batch_size = 8
    elif pooling == "last":
        batch_size = 8
    else:
        batch_size = 64

    logger.info(
        "model=%s pooling=%s instr=%s device=%s max_len=%d bs=%d smoke=%s",
        model_name, pooling, use_instr, device, max_length, batch_size, args.smoke,
    )

    _raw, clean_corpus, _ctx = load_stage1()
    rrf_results = load_bm25_rrf()
    labels = load_labels()

    if args.smoke:
        query_ids, corpus_ids = select_smoke_subset(
            labels, clean_corpus, rrf_results, n_queries=15, n_corpus=60
        )
        logger.info("smoke subset: %d queries, %d corpus docs",
                    len(query_ids), len(corpus_ids))
    else:
        query_ids = [q for q in labels if q in clean_corpus]
        corpus_ids = list(clean_corpus.keys())
        logger.info("full run: %d queries, %d corpus docs",
                    len(query_ids), len(corpus_ids))

    # Embed corpus (passages, no instruction) and queries (with instruction).
    corpus_texts = [clean_corpus[c] for c in corpus_ids]
    query_texts = [clean_corpus[q] for q in query_ids]

    logger.info("Embedding %d corpus passages ...", len(corpus_texts))
    corpus_embs = embed_texts(
        corpus_texts, model_name, pooling, is_query=False,
        max_length=max_length, batch_size=batch_size, device=device,
        use_instruction=use_instr,
    )
    logger.info("Embedding %d queries ...", len(query_texts))
    query_embs = embed_texts(
        query_texts, model_name, pooling, is_query=True,
        max_length=max_length, batch_size=batch_size, device=device,
        use_instruction=use_instr,
    )

    # Dense retrieval.
    dense_rankings = dense_topk(query_embs, corpus_embs, corpus_ids, query_ids, TOP_K)

    # BM25-RRF baseline rankings (restricted to evaluated query set).
    bm25_rankings = {q: rrf_results.get(q, []) for q in query_ids}

    # Fused = RRF(dense, bm25_rrf).
    fused_rankings = {
        q: rrf_fuse_two(dense_rankings.get(q, []), bm25_rankings.get(q, []),
                        k=RRF_K, top_k=TOP_K)
        for q in query_ids
    }

    baseline_recall = micro_recall_at_ks(bm25_rankings, labels, query_ids, RECALL_KS)
    dense_recall = micro_recall_at_ks(dense_rankings, labels, query_ids, RECALL_KS)
    fused_recall = micro_recall_at_ks(fused_rankings, labels, query_ids, RECALL_KS)

    logger.info("BM25-RRF baseline: %s", baseline_recall)
    logger.info("Dense-only:        %s", dense_recall)
    logger.info("Fused (RRF k=60):  %s", fused_recall)

    result = {
        "model": model_name,
        "mode": "smoke" if args.smoke else "full",
        "device": device,
        "pooling": pooling,
        "query_instruction_used": use_instr,
        "max_length": max_length,
        "n_queries": len(query_ids),
        "n_corpus": len(corpus_ids),
        "rrf_k": RRF_K,
        "top_k": TOP_K,
        "recall_ks": RECALL_KS,
        "bm25_rrf_baseline": baseline_recall,
        "dense_only": dense_recall,
        "fused": fused_recall,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
