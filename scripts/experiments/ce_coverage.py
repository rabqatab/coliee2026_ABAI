"""CE coverage expansion experiment (ADACUR-motivated).

Motivation (from error analysis): ~50% of gold misses are reranking-side, and
the fine-tuned cross-encoder currently scores only ~50 of the 200 pool
candidates per query ("smart" mode, CROSSENCODER_TOP_K). This script tests the
hypothesis: if the *same* fine-tuned CE scores ALL 200 pool candidates (full
coverage), how many reranking-miss golds (gold in pool but unscored / scored ~0
under partial coverage) get a high CE score, and what is the meta-learner test
delta-F1 when the crossencoder_score / crossencoder_rank features are rebuilt
with full coverage?

REUSE (no truncation reimplementation): this script imports and calls
`crossencoder_rerank` from `coliee_task1.stages.crossencoder` -- the exact
selective-truncation input builder (head/tail + top citation-context windows +
top paragraphs, 500-word budget) used by the production pipeline (stage 4).

INPUTS:
  - CE checkpoint:   output/models_v2/crossencoder/final  (DeBERTa-v3-large, smart)
  - Candidate pool:  output/pipeline_cache/stage2.pkl  (4-tuple; rrf_results = [0])
  - Cached ~50 CE:   output/pipeline_cache/stage4.pkl  ({qid: {cid: score}})
  - Corpus+context:  output/pipeline_cache/stage1.pkl  (raw, clean, contexts)
  - Feature matrix:  output/feature_matrix.parquet  (meta-learner harness)
  - Labels:          data/task1/task1_train_labels_2026.json

MODES:
  --smoke   CPU validation in <3 min: tiny CE (base microsoft/deberta-v3-large
            randomly-init head OR --model), ~5 queries x few candidates. Only
            validates the input-build -> score -> coverage-merge -> metric
            plumbing; numbers are meaningless.
  (default) Full subset run: score ALL 200 pool candidates with the fine-tuned
            CE for --limit queries, rebuild features, report recovery + dF1.

Run (CPU smoke):
  uv run python scripts/experiments/ce_coverage.py --smoke
Run (GPU subset -- DO NOT run here; use nvcr docker, see module footer):
  uv run python scripts/experiments/ce_coverage.py --device cuda --limit 100
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# common.py lives alongside this file and sets up sys.path to src/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CALIB_LGBM_PARAMS,
    REPO,
    load_feature_matrix,
    pred_global,
    prf,
    select_features,
)

PIPELINE_CACHE = REPO / "output" / "pipeline_cache"
LABELS_PATH = REPO / "data" / "task1" / "task1_train_labels_2026.json"
DEFAULT_MODEL = REPO / "output" / "models_v2" / "crossencoder" / "final"
SMOKE_MODEL = "microsoft/deberta-v3-large"


# ---------------------------------------------------------------------------
# Cache loaders (graphrag-alias unpickle trick, as in scripts/w2_c2_diag.py)
# ---------------------------------------------------------------------------
def _install_graphrag_aliases():
    """The pickles reference the old `graphrag` package paths for the
    DocumentContexts / CitationContext classes. Alias them to the current
    coliee_task1 modules so unpickling resolves."""
    import coliee_task1
    import coliee_task1.stages.citation_context as cc
    sys.modules.setdefault("graphrag", coliee_task1)
    sys.modules.setdefault("graphrag.stages", coliee_task1.stages)
    sys.modules.setdefault("graphrag.citation_context", cc)
    sys.modules.setdefault("graphrag.stages.citation_context", cc)


def load_stage1():
    _install_graphrag_aliases()
    with open(PIPELINE_CACHE / "stage1.pkl", "rb") as f:
        raw, clean, contexts = pickle.load(f)
    return raw, clean, contexts


def load_rrf():
    with open(PIPELINE_CACHE / "stage2.pkl", "rb") as f:
        s2 = pickle.load(f)
    return s2[0]  # rrf_results = {qid: [(cid, score), ...]} (200/query)


def load_cached_ce():
    with open(PIPELINE_CACHE / "stage4.pkl", "rb") as f:
        s4 = pickle.load(f)
    # stage4.pkl is {qid: {cid: score}} (smart, ~50/query)
    if isinstance(s4, dict) and "crossencoder_scores" in s4:
        return s4["crossencoder_scores"]
    return s4


# ---------------------------------------------------------------------------
# Full-coverage CE scoring (reuses crossencoder_rerank input builder)
# ---------------------------------------------------------------------------
def score_full_coverage(qids, rrf, clean, contexts, model, tokenizer,
                        max_cand, max_length, batch_size, device_str):
    """Score the top `max_cand` pool candidates per query with the fine-tuned
    CE in smart mode, reusing the production input builder.

    Returns {qid: {cid: score}} -- same shape as the cached stage4 dict.
    """
    from coliee_task1.stages.crossencoder import crossencoder_rerank

    out = {}
    t0 = time.time()
    for i, qid in enumerate(qids):
        pool = rrf.get(qid, [])[:max_cand]
        if not pool:
            out[qid] = {}
            continue
        query_text = clean.get(qid, "")
        candidates = [(cid, clean.get(cid, "")) for cid, _ in pool]

        # Smart-mode context construction -- identical to pipeline.py stage 4.
        dc = contexts.get(qid)
        q_ctx = [c.text for c in dc.contexts] if dc and dc.contexts else None
        c_ctx = {}
        for cid, _ in pool:
            dc_c = contexts.get(cid)
            if dc_c and dc_c.contexts:
                c_ctx[cid] = [c.text for c in dc_c.contexts]

        reranked = crossencoder_rerank(
            model, tokenizer, query_text, candidates,
            max_length=max_length,
            batch_size=batch_size,
            mode="smart",
            query_contexts=q_ctx,
            candidate_contexts=c_ctx,
        )
        out[qid] = {cid: float(s) for cid, s in reranked}
        if (i + 1) % 10 == 0:
            print(f"  CE full-coverage: {i + 1}/{len(qids)} queries "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return out


# ---------------------------------------------------------------------------
# Recovery analysis
# ---------------------------------------------------------------------------
def recovery_analysis(qids, rrf, labels, cached_ce, full_ce, high_thr,
                      partial_zero=1e-4):
    """For golds that are in the pool but were reranking-misses under partial
    coverage (not scored, or scored ~0 by the cached ~50 CE), count how many now
    get a high CE score (>= high_thr) under full coverage.

    A 'reranking-miss gold' = gold present in the 200-pool but either absent
    from the cached CE dict for that query or with cached score <= partial_zero.
    """
    pool_golds = 0
    miss_golds = 0          # reranking-miss golds (in pool, partial-unscored/~0)
    recovered = 0           # of those, now CE >= high_thr under full coverage
    miss_full_scores = []   # full-coverage CE scores for the miss golds

    for qid in qids:
        gold = set(labels.get(qid, []))
        if not gold:
            continue
        pool_cids = {cid for cid, _ in rrf.get(qid, [])}
        cce = cached_ce.get(qid, {})
        fce = full_ce.get(qid, {})
        for g in gold:
            if g not in pool_cids:
                continue  # retrieval-side miss, out of scope here
            pool_golds += 1
            cached_s = cce.get(g, 0.0)
            if cached_s > partial_zero:
                continue  # already covered with a real score under partial CE
            miss_golds += 1
            full_s = fce.get(g, 0.0)
            miss_full_scores.append(full_s)
            if full_s >= high_thr:
                recovered += 1

    return {
        "pool_golds": pool_golds,
        "reranking_miss_golds": miss_golds,
        "recovered_high_ce": recovered,
        "recovery_rate": (recovered / miss_golds) if miss_golds else 0.0,
        "high_ce_threshold": high_thr,
        "miss_gold_full_ce_mean": float(np.mean(miss_full_scores)) if miss_full_scores else 0.0,
        "miss_gold_full_ce_p90": float(np.percentile(miss_full_scores, 90)) if miss_full_scores else 0.0,
    }


# ---------------------------------------------------------------------------
# Meta-learner dF1 on the subset (cached-coverage vs full-coverage CE features)
# ---------------------------------------------------------------------------
def _rebuild_ce_features(df, ce_scores):
    """Return a copy of df with crossencoder_score / crossencoder_rank
    overwritten from `ce_scores` = {qid: {cid: score}}.

    Mirrors the feature-matrix convention: unscored candidates get
    crossencoder_score = 0.0; crossencoder_rank is the per-query descending rank
    by CE score (1 = best). Other features are untouched.
    """
    df = df.copy()
    new_score = np.zeros(len(df), dtype=float)
    qid_arr = df.query_id.values
    cid_arr = df.candidate_id.values
    for idx in range(len(df)):
        d = ce_scores.get(qid_arr[idx])
        if d:
            new_score[idx] = d.get(cid_arr[idx], 0.0)
    df["crossencoder_score"] = new_score
    # per-query descending rank by CE score
    df["crossencoder_rank"] = (
        df.groupby("query_id")["crossencoder_score"]
        .rank(ascending=False, method="first")
        .astype(float)
    )
    return df


def _split_train_test(qids_sorted, frac=0.8):
    n = len(qids_sorted)
    i = int(n * frac)
    return set(qids_sorted[:i]), set(qids_sorted[i:])


def meta_learner_df1(df_subset, feats, frac=0.8):
    """Train LGBM on the subset train split, sweep a global threshold on the
    train split, evaluate micro-F1 on the test split. Returns the F1 + threshold.

    NOTE: this is a self-contained 2-way split over the SUBSET only -- it is a
    relative comparison between cached-coverage and full-coverage CE feature
    columns on the identical split, not the authoritative 3-way number.
    """
    import lightgbm as lgb

    qids = sorted(df_subset.query_id.unique())
    train_q, test_q = _split_train_test(qids, frac)
    tr = df_subset[df_subset.query_id.isin(train_q)]
    te = df_subset[df_subset.query_id.isin(test_q)]

    model = lgb.train(
        CALIB_LGBM_PARAMS,
        lgb.Dataset(tr[feats].values, tr.label.values),
        num_boost_round=300,
        callbacks=[lgb.log_evaluation(0)],
    )

    def _qs_lab(d, scores):
        qs, lab = {}, {}
        for qid, cid, sc, y in zip(d.query_id.values, d.candidate_id.values, scores, d.label.values):
            qs.setdefault(qid, []).append((cid, float(sc)))
            if y == 1:
                lab.setdefault(qid, []).append(cid)
        for qid in qs:
            lab.setdefault(qid, [])
            qs[qid].sort(key=lambda x: -x[1])
        return qs, lab

    tr_qs, tr_lab = _qs_lab(tr, model.predict(tr[feats].values))
    te_qs, te_lab = _qs_lab(te, model.predict(te[feats].values))

    best = (-1.0, 0.5)
    for t in np.arange(0.01, 1.0, 0.01):
        f = prf(pred_global(tr_qs, float(t)), tr_lab)["f1"]
        if f > best[0]:
            best = (f, float(t))
    t_star = best[1]
    res = prf(pred_global(te_qs, t_star), te_lab)
    res["threshold"] = t_star
    res["n_test_queries"] = len(test_q)
    return res


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_ce_model(model_ref, device_str, smoke):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device(device_str)
    tokenizer = AutoTokenizer.from_pretrained(str(model_ref))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_ref), num_labels=2,
    )
    # Match production: DeBERTa-v3 runs in fp32 (XSoftmax fp16 NaN guard); on
    # cuda the recommended docker command uses bf16 autocast at call sites, but
    # the saved checkpoint is fp32 so we load it as-is.
    model = model.to(device).float().eval()
    return model, tokenizer, device


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    ap.add_argument("--limit", type=int, default=100,
                    help="number of queries to process (subset)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL),
                    help="CE checkpoint dir or HF id")
    ap.add_argument("--max-cand", type=int, default=200,
                    help="full-coverage candidate cap per query")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--high-thr", type=float, default=0.5,
                    help="CE score threshold counted as 'recovered'")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU plumbing smoke test (tiny, ~5 queries)")
    ap.add_argument("--out", default=str(REPO / "output" / "experiments" / "ce_coverage.json"))
    args = ap.parse_args()

    if args.smoke:
        args.device = "cpu"
        args.model = SMOKE_MODEL          # base DeBERTa (random head) -- plumbing only
        args.limit = 5
        args.max_cand = 6                 # tiny candidate set
        args.batch_size = 4
        print("[SMOKE] CPU plumbing test: base DeBERTa (random head), "
              "5 queries x 6 candidates. Numbers are MEANINGLESS.", flush=True)

    t0 = time.time()
    print(f"loading caches (device={args.device}, limit={args.limit}, "
          f"max_cand={args.max_cand})...", flush=True)
    raw, clean, contexts = load_stage1()
    rrf = load_rrf()
    cached_ce = load_cached_ce()
    labels = json.load(open(LABELS_PATH))

    # Subset: queries that exist in rrf + clean, with at least one in-pool gold
    # (so recovery analysis has signal), then take the first --limit.
    qids = []
    for q in sorted(rrf.keys()):
        if q not in clean:
            continue
        gold = set(labels.get(q, []))
        pool_cids = {c for c, _ in rrf.get(q, [])}
        if gold & pool_cids:
            qids.append(q)
        if len(qids) >= args.limit:
            break
    print(f"subset queries: {len(qids)} (caches loaded in {time.time() - t0:.0f}s)",
          flush=True)

    # Load CE model
    print(f"loading CE model: {args.model}", flush=True)
    model, tokenizer, device = load_ce_model(args.model, args.device, args.smoke)
    print(f"  model on {device}, num_labels={model.config.num_labels}, "
          f"type={model.config.model_type}", flush=True)

    # Full-coverage scoring (reuses crossencoder_rerank input builder)
    print("scoring full coverage...", flush=True)
    full_ce = score_full_coverage(
        qids, rrf, clean, contexts, model, tokenizer,
        max_cand=args.max_cand, max_length=args.max_length,
        batch_size=args.batch_size, device_str=args.device,
    )
    n_full = int(np.mean([len(full_ce[q]) for q in qids])) if qids else 0
    n_cached = int(np.mean([len(cached_ce.get(q, {})) for q in qids])) if qids else 0
    print(f"  avg scored/query: cached={n_cached}  full={n_full}", flush=True)

    # (a) Recovery of reranking-miss golds
    rec = recovery_analysis(qids, rrf, labels, cached_ce, full_ce, args.high_thr)
    print("recovery:", json.dumps(rec, indent=2), flush=True)

    # (b) Meta-learner dF1 on the subset: cached-coverage vs full-coverage CE
    df = load_feature_matrix()
    feats = select_features(df)
    df_sub = df[df.query_id.isin(set(qids))].copy()
    print(f"feature-matrix subset rows: {len(df_sub)} "
          f"({df_sub.query_id.nunique()} queries)", flush=True)

    df_cached = _rebuild_ce_features(df_sub, cached_ce)
    df_full = _rebuild_ce_features(df_sub, full_ce)

    skip_meta = df_sub.query_id.nunique() < 8
    if skip_meta:
        print("[SMOKE] <8 subset queries -> skipping LGBM dF1 (80/20 split needs "
              ">=2 test queries); feature-rebuild plumbing still validated above.",
              flush=True)
        baseline = {"note": "skipped (smoke)"}
        full = {"note": "skipped (smoke)"}
        delta_f1 = None
    else:
        baseline = meta_learner_df1(df_cached, feats)
        full = meta_learner_df1(df_full, feats)
        delta_f1 = full["f1"] - baseline["f1"]
        print(f"meta-learner: cached F1={baseline['f1']:.4f}  "
              f"full F1={full['f1']:.4f}  dF1={delta_f1:+.4f}", flush=True)

    payload = {
        "_meta": {
            "smoke": args.smoke,
            "device": args.device,
            "model": args.model,
            "limit": args.limit,
            "max_cand": args.max_cand,
            "n_subset_queries": len(qids),
            "avg_scored_cached": n_cached,
            "avg_scored_full": n_full,
            "runtime_s": round(time.time() - t0, 1),
        },
        "recovery": rec,
        "meta_learner": {
            "cached_coverage": baseline,
            "full_coverage": full,
            "delta_f1": delta_f1,
        },
    }

    out_path = Path(args.out)
    if args.smoke:
        out_path = out_path.with_name("ce_coverage_smoke.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {out_path}", flush=True)
    print(json.dumps(payload, indent=2, default=float), flush=True)
    if args.smoke:
        print("\n[SMOKE PASS] CE-input-build -> score -> coverage-merge -> "
              "feature-rebuild -> metric plumbing OK.", flush=True)


# ---------------------------------------------------------------------------
# RECOMMENDED GPU SUBSET RUN (nvcr docker; bf16; DO NOT run base CE on GPU here)
# ---------------------------------------------------------------------------
# docker run --rm --gpus all \
#   -v /home/alphabridge/Research/coliee2026:/workspace -w /workspace \
#   nvcr.io/nvidia/pytorch:24.10-py3 bash -lc '
#     pip install -q transformers sentencepiece pandas lightgbm && \
#     python scripts/experiments/ce_coverage.py \
#       --device cuda --limit 100 \
#       --model output/models_v2/crossencoder/final --batch-size 32'
# (DeBERTa-v3-large checkpoint is fp32; on GB10 wrap the score call in
#  torch.autocast("cuda", dtype=torch.bfloat16) if memory/throughput needs it.)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
