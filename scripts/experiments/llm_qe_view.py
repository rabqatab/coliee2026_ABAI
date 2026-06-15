"""LLM query expansion as a recall-ceiling view (Corpus-Steered / Exp4Fuse / HUKB style).

Hypothesis: for a query case with its citations suppressed, an instruction-tuned
generation LLM can name the legal issues / statutes / holdings / doctrinal terms
that the *cited* cases likely address. Using that generated text as an extra BM25
query ("expansion view") and RRF-fusing it with the existing BM25-RRF pool should
recover golds that the lexical pool alone misses -- i.e. lift the recall ceiling.

Pipeline (generation -> BM25 view -> RRF -> recall):
  1. For each query, build the prompt from the (truncated) query case text.
  2. Generate the expansion text (real LLM at GPU time, or a MOCK generator on CPU).
  3. Query a BM25 index over the corpus with the expansion text -> top-200 view.
  4. RRF-fuse that view with stage2 `rrf_results` (the BM25-RRF pool).
  5. Report recall@{50,100,200} for: bm25_rrf (reference), qe_view alone, fused;
     and count golds recovered by the fused pool that the BM25-RRF pool missed.

Real generation reuses `coliee_task1.stages.query_expansion.expand_queries_llm`
(Qwen/Qwen2.5-7B-Instruct, bf16) -- that path is GPU-only and is NOT exercised by
--smoke. --smoke uses a mock, embed-free, CPU-only generator to validate the
plumbing end to end in < 2 min.

BM25 index + tokenizer reuse `coliee_task1.stages.bm25` (BM25Index, tokenize).

Reference sanity number: BM25-RRF recall@200 = 0.578.

Examples
--------
  # CPU smoke (mock generator, no model download, < 2 min):
  uv run python scripts/experiments/llm_qe_view.py --smoke --limit 200

  # Real GPU run, first subset diagnostic (Qwen2.5-7B-Instruct, bf16):
  uv run python scripts/experiments/llm_qe_view.py \
      --model Qwen/Qwen2.5-7B-Instruct --limit 200 --out llm_qe_view_qwen200
"""
import argparse
import pickle
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from coliee_task1.stages.bm25 import BM25Index, rrf_fuse, tokenize  # noqa: E402

from common import (  # noqa: E402
    LABELS_PATH,
    PIPELINE_CACHE,
    recall_at_k,
    write_result,
)

import json  # noqa: E402

# --- Generation prompt -----------------------------------------------------
# Distinct from query_expansion.EXPANSION_PROMPT (which asks for a free-text
# pseudo-document a la Query2Doc). Here we want a recall-ceiling *view*: a terse,
# keyword-dense list of the issues/statutes/holdings/terms the CITED cases likely
# address -- no prose, so it acts as a strong BM25 bag of legal terms.
QE_VIEW_PROMPT = """You are a legal research assistant. The following is an excerpt of a court \
decision in which citations to prior cases have been suppressed. List the legal \
issues, statutes and regulations, doctrinal tests, and key legal terms that the \
cited prior cases most likely address. Output a concise comma-separated list of \
terms and short phrases only -- no sentences, no explanation, no numbering.

Case excerpt:
{excerpt}

Likely-cited legal issues, statutes, doctrines, and terms (comma-separated):"""

MAX_EXCERPT_WORDS = 500
QE_MAX_NEW_TOKENS = 160
QE_VIEW_TOP_K = 200
RECALL_KS = (50, 100, 200)


# --- stage cache loaders ---------------------------------------------------
def _register_graphrag_aliases() -> None:
    """stage1/stage2 were pickled under the old `graphrag.*` package path.

    Register aliases so unpickling resolves the classes to coliee_task1.* (same
    trick as scripts/w2_c2_diag.py).
    """
    import coliee_task1
    import coliee_task1.stages.citation_context as cc

    sys.modules.setdefault("graphrag", coliee_task1)
    sys.modules.setdefault("graphrag.stages", coliee_task1.stages)
    sys.modules.setdefault("graphrag.citation_context", cc)
    sys.modules.setdefault("graphrag.stages.citation_context", cc)


def load_stage1():
    """Return (clean_corpus, contexts) from stage1.pkl (3-tuple)."""
    _register_graphrag_aliases()
    with open(PIPELINE_CACHE / "stage1.pkl", "rb") as f:
        _raw, clean, contexts = pickle.load(f)
    return clean, contexts


def load_rrf_results() -> dict:
    """Return rrf_results {qid: [(cid, score)]} from stage2.pkl (4-tuple, idx 0)."""
    _register_graphrag_aliases()
    with open(PIPELINE_CACHE / "stage2.pkl", "rb") as f:
        return pickle.load(f)[0]


# --- generators ------------------------------------------------------------
def mock_expand(
    query_texts: dict[str, str],
    contexts: dict,
    max_terms: int = 60,
) -> dict[str, str]:
    """Embed-free, CPU-only stand-in for the LLM generator (for --smoke).

    Produces a keyword-dense "expansion" with NO access to gold labels, using
    only signals the real LLM is also given (the query text + its own citation
    contexts):
      - the citation-context windows around <FRAGMENT_SUPPRESSED> markers
        (these surround the suppressed citations -- the strongest local signal
        for what the cited cases are about), plus
      - the top corpus-level TF terms from the query body, restricted to
        salient (>=5 char) alphabetic tokens.

    This deliberately mirrors the real flow (text -> terms -> BM25 query) so the
    plumbing it exercises is the same; it is NOT meant to match LLM quality.
    """
    from collections import Counter

    expanded: dict[str, str] = {}
    for qid, text in query_texts.items():
        parts: list[str] = []

        # (1) citation-context windows -- local signal around suppressed cites
        dc = contexts.get(qid)
        if dc is not None and getattr(dc, "contexts", None):
            for c in dc.contexts:
                parts.append(c.text)

        # (2) salient body terms by frequency (TF), excluding the marker token
        body = text.replace("FRAGMENT_SUPPRESSED", " ")
        toks = [t for t in re.findall(r"[a-z]+", body.lower()) if len(t) >= 5]
        common = [w for w, _ in Counter(toks).most_common(max_terms)]
        parts.append(" ".join(common))

        expanded[qid] = " ".join(parts).strip() or text
    return expanded


def llm_expand(
    query_texts: dict[str, str],
    model_name: str,
) -> dict[str, str]:
    """Real generation via HuggingFace transformers (Qwen2.5-7B-Instruct, bf16).

    GPU-only. Built on top of the existing stage module's model-loading recipe
    but with the recall-ceiling-view prompt (QE_VIEW_PROMPT) and returning the
    GENERATED text only (not query + generation), since we want a standalone
    expansion view to fuse, not an enriched original query.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[llm] loading {model_name} (bf16, device_map=auto)...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    use_chat = hasattr(tok, "apply_chat_template") and tok.chat_template is not None

    expanded: dict[str, str] = {}
    t0 = time.time()
    for i, (qid, text) in enumerate(query_texts.items()):
        excerpt = " ".join(text.split()[:MAX_EXCERPT_WORDS])
        user = QE_VIEW_PROMPT.format(excerpt=excerpt)
        if use_chat:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True,
            )
        else:
            prompt = user

        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=QE_MAX_NEW_TOKENS,
                do_sample=False, temperature=None, top_p=None,
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        expanded[qid] = gen.strip() or text

        if (i + 1) % 25 == 0:
            print(f"  [llm] {i + 1}/{len(query_texts)} ({time.time() - t0:.0f}s)", flush=True)

    print(f"[llm] generation done: {len(expanded)} queries ({time.time() - t0:.0f}s)", flush=True)
    return expanded


# --- recall / fusion helpers ----------------------------------------------
def recall_pool(pool: dict, labels: dict, qids: list[str], ks=RECALL_KS) -> dict:
    """Micro recall@k over a {qid: [(cid, score)]} pool vs labels.

    recall_at_k handles truncation to k internally and returns {"micro","macro"};
    we restrict labels to the evaluated qids and report the micro value.
    """
    rel = {q: labels.get(q, []) for q in qids}
    out = {}
    for k in ks:
        out[f"R@{k}"] = float(recall_at_k(pool, rel, k)["micro"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                    help="HF model id for the real generator (GPU).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Use only the first N queries (with gold present).")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU mock generator -- validates the plumbing, no model.")
    ap.add_argument("--out", default="llm_qe_view",
                    help="Output name under output/experiments/<out>.json")
    args = ap.parse_args()

    print("loading stage caches + labels...", flush=True)
    clean, contexts = load_stage1()
    rrf_results = load_rrf_results()
    labels = json.load(open(LABELS_PATH))

    # Queries with at least one gold present in the corpus (consistent with
    # w2_c2_diag.py). Sorted for determinism, then optionally limited.
    qids = [
        q for q in sorted(labels)
        if q in clean and any(g in clean for g in labels[q])
    ]
    if args.limit is not None:
        qids = qids[: args.limit]
    print(f"corpus={len(clean)} queries={len(qids)} mode={'SMOKE' if args.smoke else 'LLM'}",
          flush=True)

    query_texts = {q: clean[q] for q in qids}

    # --- generate expansion text -------------------------------------------
    t0 = time.time()
    if args.smoke:
        print("[smoke] mock generator (citation-contexts + TF terms, CPU)...", flush=True)
        expansions = mock_expand(query_texts, contexts)
    else:
        expansions = llm_expand(query_texts, args.model)
    gen_secs = time.time() - t0

    # --- BM25 index over the corpus ----------------------------------------
    # Reuse BM25Index + tokenize from the pipeline. The cached multi-view index
    # is not serialized, so we (re)build a full-doc index here -- same tokenizer
    # and same BM25 params as the pipeline, so the qe_view is comparable.
    print("building BM25 index over corpus...", flush=True)
    cids = sorted(clean.keys())
    bm = BM25Index()
    bm.fit(cids, [clean[c] for c in cids])

    # --- query with expansion text -> qe_view ------------------------------
    print("querying BM25 with expansions -> qe_view...", flush=True)
    qe_view: dict[str, list[tuple[str, float]]] = {}
    empty_exp = 0
    for q in qids:
        exp = expansions.get(q, "")
        if not tokenize(exp):
            empty_exp += 1
            qe_view[q] = []
            continue
        res = bm.query(exp, top_k=QE_VIEW_TOP_K + 1)
        qe_view[q] = [(c, s) for c, s in res if c != q][:QE_VIEW_TOP_K]

    # --- RRF-fuse qe_view with the BM25-RRF pool ---------------------------
    print("RRF-fusing qe_view with bm25_rrf...", flush=True)
    fused: dict[str, list[tuple[str, float]]] = {}
    bm25_rrf = {q: rrf_results.get(q, []) for q in qids}
    for q in qids:
        fused[q] = rrf_fuse([bm25_rrf[q], qe_view[q]], top_k=QE_VIEW_TOP_K)

    # --- recall @ k --------------------------------------------------------
    r_bm25 = recall_pool(bm25_rrf, labels, qids)
    r_qe = recall_pool(qe_view, labels, qids)
    r_fused = recall_pool(fused, labels, qids)

    # --- new golds recovered by fused beyond bm25_rrf top-200 --------------
    recovered = 0
    total_gold = 0
    queries_helped = 0
    for q in qids:
        gold = set(labels.get(q, [])) & set(clean.keys())
        total_gold += len(gold)
        base = {c for c, _ in bm25_rrf[q][:QE_VIEW_TOP_K]}
        fus = {c for c, _ in fused[q][:QE_VIEW_TOP_K]}
        new = (gold & fus) - base
        recovered += len(new)
        if new:
            queries_helped += 1

    lift200 = r_fused["R@200"] - r_bm25["R@200"]

    result = {
        "config": {
            "smoke": args.smoke,
            "model": "MOCK(contexts+TF)" if args.smoke else args.model,
            "n_queries": len(qids),
            "n_corpus": len(cids),
            "qe_view_top_k": QE_VIEW_TOP_K,
            "rrf_k": 60,
            "empty_expansions": empty_exp,
            "generation_seconds": round(gen_secs, 1),
        },
        "recall": {
            "bm25_rrf": r_bm25,
            "qe_view_alone": r_qe,
            "fused": r_fused,
        },
        "lift": {
            "R@50": r_fused["R@50"] - r_bm25["R@50"],
            "R@100": r_fused["R@100"] - r_bm25["R@100"],
            "R@200": lift200,
        },
        "new_golds": {
            "total_gold_in_corpus": total_gold,
            "recovered_beyond_bm25_rrf_top200": recovered,
            "queries_helped": queries_helped,
        },
    }

    path = write_result(args.out, result, script=Path(__file__).name)
    print(json.dumps(result, indent=2, default=float), flush=True)
    print(f"\nwrote {path}", flush=True)
    print(f"SANITY: bm25_rrf R@200={r_bm25['R@200']:.3f} "
          f"(reference 0.578) | fused R@200={r_fused['R@200']:.3f} "
          f"(+{lift200:.3f}) | recovered {recovered} new golds "
          f"across {queries_helped} queries", flush=True)


if __name__ == "__main__":
    main()
