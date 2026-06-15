"""U-CREAT-style event-extraction retrieval view (Joshi et al., ACL 2023).

Builds a structured-but-lexical BM25 retrieval view over per-document EVENTS
(SVO triples + verb/argument tokens), then measures whether fusing it (RRF) with
the existing multi-view BM25-RRF pool lifts the first-stage recall ceiling.

Pipeline
  1. Event extraction (CPU, spaCy en_core_web_sm, dependency parse):
       per sentence -> for each verb, take its lemma + subject head(s) + object
       head(s) (nsubj/nsubjpass/dobj/pobj/attr/dative). Each (subj, verb, obj)
       SVO triple plus loose verb+argument tokens are emitted. A document's
       "event document" is the concatenation of all its event tokens.
     Docs are capped at 40k chars (keeps ~81% of corpus characters; only ~18% of
     docs are truncated) and parsed with multiprocessing for speed.
  2. BM25 index over the 9,556 event-documents (reuses BM25Index).
  3. Query side = the query case's own event-document. Retrieve top-200/query.
  4. Metrics: event-view-only recall@{50,100,200}; RRF fusion of the event view
     with the cached BM25-RRF pool (stage2 rrf_results) recall@{50,100,200}.
     Reports golds recovered vs BM25-RRF (golds the fused pool@200 finds that the
     BM25-RRF pool@200 missed).

Baseline sanity: BM25-RRF recall@200 (micro) = 0.5777 over 2001 queries / 8251 gold.

Run:    uv run python scripts/experiments/ucreat_event_view.py
Writes: output/experiments/ucreat_event_view.json
"""
import json
import logging
import pickle  # noqa: S403 -- internal pipeline cache only
import sys
import time
from pathlib import Path

import spacy

from common import LABELS_PATH, PIPELINE_CACHE, write_result
from coliee_task1.config import RRF_K
from coliee_task1.stages.bm25 import BM25Index, rrf_fuse
from coliee_task1.utils.metrics import recall_at_k

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ucreat_event_view")

STAGE1_PKL = PIPELINE_CACHE / "stage1.pkl"
STAGE2_PKL = PIPELINE_CACHE / "stage2.pkl"
EVENT_CACHE = PIPELINE_CACHE / "ucreat_event_docs.pkl"

CHAR_CAP = 40_000          # truncate long docs before parsing
N_PROCESS = 16             # spaCy multiprocessing workers (machine has 20 CPUs)
SPACY_BATCH = 16
EVENT_TOP_K = 200
RECALL_KS = [50, 100, 200]

SUBJ_DEPS = {"nsubj", "nsubjpass", "csubj"}
OBJ_DEPS = {"dobj", "obj", "pobj", "attr", "dative", "oprd"}


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


def _arg_token(tok):
    """Lowercased head text for an argument token (skip pronouns/punct/stop)."""
    if tok.is_stop or tok.is_punct or tok.like_num or len(tok.text) < 2:
        return None
    if tok.pos_ in ("PRON",):
        return None
    return tok.lemma_.lower() if tok.lemma_ else tok.text.lower()


def extract_events(doc) -> str:
    """Return the space-joined event document for one parsed spaCy Doc.

    For each VERB token: collect subject heads and object heads from its
    children. Emit SVO triples 'subj_verb_obj' as single tokens (structured
    signal) plus the loose 'subj', 'verb', 'obj' argument tokens (lexical
    backoff so partial-overlap events still match under BM25).
    """
    out = []
    for tok in doc:
        if tok.pos_ not in ("VERB", "AUX"):
            continue
        verb = tok.lemma_.lower()
        if not verb or len(verb) < 2:
            continue
        subs, objs = [], []
        for ch in tok.children:
            if ch.dep_ in SUBJ_DEPS:
                a = _arg_token(ch)
                if a:
                    subs.append(a)
            elif ch.dep_ in OBJ_DEPS:
                a = _arg_token(ch)
                if a:
                    objs.append(a)
            elif ch.dep_ == "prep":  # follow preposition to its object
                for gc in ch.children:
                    if gc.dep_ == "pobj":
                        a = _arg_token(gc)
                        if a:
                            objs.append(a)
        if not subs and not objs:
            continue
        # loose argument + verb tokens (lexical backoff)
        out.append(verb)
        out.extend(subs)
        out.extend(objs)
        # structured SVO triples (joined so they are distinct BM25 terms)
        for s in subs or [""]:
            for o in objs or [""]:
                if s or o:
                    out.append(f"{s}-{verb}-{o}".strip("-"))
    return " ".join(out)


def build_event_docs(clean_corpus, doc_ids):
    """Parse every doc and return {doc_id: event_document}. Cached to disk."""
    if EVENT_CACHE.exists():
        logger.info("Loading cached event docs from %s", EVENT_CACHE)
        with open(EVENT_CACHE, "rb") as f:
            return pickle.load(f)  # noqa: S301

    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    nlp.max_length = CHAR_CAP + 1000
    texts = [clean_corpus[d][:CHAR_CAP] for d in doc_ids]

    t0 = time.time()
    event_docs = {}
    n = 0
    for did, doc in zip(
        doc_ids,
        nlp.pipe(texts, batch_size=SPACY_BATCH, n_process=N_PROCESS),
    ):
        event_docs[did] = extract_events(doc)
        n += 1
        if n % 500 == 0:
            logger.info("parsed %d/%d (%.0fs)", n, len(doc_ids), time.time() - t0)
    logger.info("Event extraction done: %d docs in %.0fs", len(doc_ids), time.time() - t0)

    with open(EVENT_CACHE, "wb") as f:
        pickle.dump(event_docs, f)
    return event_docs


def main():
    t_start = time.time()
    logger.info("Loading stage1 cache ...")
    _, clean_corpus, _ = load_stage1()
    doc_ids = sorted(clean_corpus.keys())

    labels = json.load(open(LABELS_PATH))
    query_ids = sorted(labels.keys())
    total_gold = sum(len(labels[q]) for q in query_ids)
    logger.info("corpus=%d queries=%d total_gold=%d", len(doc_ids), len(query_ids), total_gold)

    # --- baseline BM25-RRF pool (sanity) ---
    rrf_results = pickle.load(open(STAGE2_PKL, "rb"))[0]  # noqa: S301
    base_recall = {k: recall_at_k(rrf_results, labels, k) for k in RECALL_KS}
    base_200 = base_recall[200]["micro"]
    sanity_ok = abs(base_200 - 0.578) <= 0.01
    logger.info("BASELINE BM25-RRF recall@200 micro=%.4f (expected ~0.578, ok=%s)",
                base_200, sanity_ok)

    # --- event extraction + event BM25 index ---
    t_extract = time.time()
    event_docs = build_event_docs(clean_corpus, doc_ids)
    extract_secs = time.time() - t_extract
    empties = sum(1 for d in doc_ids if not event_docs[d].strip())
    avg_ev_tokens = sum(len(event_docs[d].split()) for d in doc_ids) / len(doc_ids)
    logger.info("event docs: avg %.0f tokens/doc, %d empty", avg_ev_tokens, empties)

    logger.info("Building event BM25 index ...")
    idx = BM25Index()
    idx.fit(doc_ids, [event_docs[d] for d in doc_ids])

    # --- event-view-only ranking (query = query case's event document) ---
    logger.info("Querying event view (%d queries) ...", len(query_ids))
    event_view = {}
    t_q = time.time()
    for i, qid in enumerate(query_ids):
        qtext = event_docs.get(qid, "")
        if not qtext.strip():
            event_view[qid] = []
            continue
        res = idx.query(qtext, top_k=EVENT_TOP_K + 1)
        event_view[qid] = [(d, s) for d, s in res if d != qid][:EVENT_TOP_K]
        if (i + 1) % 500 == 0:
            logger.info("  queried %d/%d (%.0fs)", i + 1, len(query_ids), time.time() - t_q)
    event_recall = {k: recall_at_k(event_view, labels, k) for k in RECALL_KS}

    # --- RRF fusion of event view with BM25-RRF pool ---
    fused = {}
    for qid in query_ids:
        fused[qid] = rrf_fuse(
            [rrf_results.get(qid, []), event_view.get(qid, [])],
            k=RRF_K, top_k=EVENT_TOP_K,
        )
    fused_recall = {k: recall_at_k(fused, labels, k) for k in RECALL_KS}

    # --- golds recovered by fusion vs BM25-RRF (at @200) ---
    new_golds = 0
    new_golds_by_query = {}
    event_only_hits_200 = 0   # golds the event view alone has in top-200
    for qid in query_ids:
        gold = set(labels.get(qid, []))
        if not gold:
            continue
        base_pool = {d for d, _ in rrf_results.get(qid, [])[:200]}
        fused_pool = {d for d, _ in fused[qid][:200]}
        ev_pool = {d for d, _ in event_view.get(qid, [])[:200]}
        recovered = gold & fused_pool - base_pool
        if recovered:
            new_golds += len(recovered)
            new_golds_by_query[qid] = sorted(recovered)
        event_only_hits_200 += len((gold & ev_pool) - base_pool)

    payload = {
        "extraction_method": (
            "spaCy en_core_web_sm dependency parse; per-verb SVO triples "
            "(subj/obj heads via nsubj*/dobj/pobj/attr/dative/prep>pobj) + loose "
            "verb/argument lemma tokens; 40k char cap; multiprocessing"
        ),
        "corpus_scope": "FULL: 9556 docs, all 2001 train queries, 8251 gold",
        "char_cap": CHAR_CAP,
        "avg_event_tokens_per_doc": avg_ev_tokens,
        "empty_event_docs": empties,
        "extraction_seconds": extract_secs,
        "sanity_check": {
            "expected_recall@200": 0.578,
            "actual_bm25_rrf_recall@200_micro": base_200,
            "abs_diff": abs(base_200 - 0.578),
            "passed": bool(sanity_ok),
        },
        "baseline_bm25_rrf": {
            f"recall@{k}": base_recall[k] for k in RECALL_KS
        },
        "event_view_only": {
            f"recall@{k}": event_recall[k] for k in RECALL_KS
        },
        "fused_event_plus_bm25rrf": {
            f"recall@{k}": fused_recall[k] for k in RECALL_KS
        },
        "lift_micro": {
            f"recall@{k}": fused_recall[k]["micro"] - base_recall[k]["micro"]
            for k in RECALL_KS
        },
        "golds_recovered_vs_bm25rrf_at200": {
            "n_new_golds": new_golds,
            "n_queries_with_new_golds": len(new_golds_by_query),
            "event_view_alone_new_golds_in_top200": event_only_hits_200,
            "total_gold": total_gold,
            "frac_of_total_gold": new_golds / total_gold,
        },
        "total_gold": total_gold,
        "n_queries": len(query_ids),
        "n_corpus": len(doc_ids),
        "runtime_seconds": time.time() - t_start,
    }

    path = write_result("ucreat_event_view", payload, script="experiments/ucreat_event_view.py")
    logger.info("Saved -> %s", path)

    def line(tag, d):
        return (f"{tag:<22} R@50={d[50]['micro']:.4f} "
                f"R@100={d[100]['micro']:.4f} R@200={d[200]['micro']:.4f}")

    print("\n=== U-CREAT event view (micro recall) ===")
    print(line("BM25-RRF baseline", base_recall))
    print(line("event-view-only", event_recall))
    print(line("fused (RRF)", fused_recall))
    print(f"\nlift@200 (fused-base) = {payload['lift_micro']['recall@200']:+.4f}")
    print(f"new golds recovered @200 = {new_golds} "
          f"({new_golds/total_gold*100:.2f}% of {total_gold}) "
          f"across {len(new_golds_by_query)} queries")
    print(f"SANITY BM25-RRF R@200={base_200:.4f} "
          f"({'PASS' if sanity_ok else 'FAIL'})")
    print(f"extraction {extract_secs:.0f}s; total {payload['runtime_seconds']:.0f}s")


if __name__ == "__main__":
    main()
