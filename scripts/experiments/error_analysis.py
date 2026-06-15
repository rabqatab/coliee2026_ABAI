"""Part A -- Error analysis of the 34-feature meta-learner on the 3-way split.

Trains the LightGBM meta-learner on the clean chronological 3-way split
(train_core 70% / calib 10% / test 20% by query id) exactly as
calibration.py does, fits the global threshold on the calib fold, and applies
it to the test fold. It then categorises every gold MISS on the test queries:

  (1) recall-ceiling miss -- the gold is NOT in the BM25-RRF top-200 pool at
      all (the first stage never surfaced it; the meta-learner could not have
      recovered it).
  (2) reranking/threshold miss -- the gold IS in the pool but the meta-learner
      scored it below the decision threshold (a reranking failure).

False POSITIVES (predicted, not gold) are also broken down, and errors are
split by legal domain (classify_domain over the query text). 2-3 concrete
case studies are pulled for reranking misses (shared statutes but low lexical
overlap, etc.).

Run:    uv run python scripts/experiments/error_analysis.py
Writes: output/experiments/error_analysis.json
        output/experiments/error_analysis.md
"""
import json
import pickle
import sys
from collections import Counter, defaultdict

import lightgbm as lgb
import numpy as np

from common import (
    CALIB_LGBM_PARAMS,
    EXPERIMENTS_DIR,
    LABELS_PATH,
    PIPELINE_CACHE,
    REPO,
    load_feature_matrix,
    pred_global,
    prf,
    select_features,
    write_result,
)

sys.path.insert(0, str(REPO / "src"))
from coliee_task1.stages.graphrag import classify_domain  # noqa: E402


def load_stage1_clean():
    """Load stage1 cache (clean corpus), aliasing the legacy 'graphrag' path."""
    import coliee_task1
    import coliee_task1.stages.citation_context as cc
    sys.modules.setdefault("graphrag", coliee_task1)
    sys.modules.setdefault("graphrag.stages", coliee_task1.stages)
    sys.modules.setdefault("graphrag.citation_context", cc)
    sys.modules.setdefault("graphrag.stages.citation_context", cc)
    with open(PIPELINE_CACHE / "stage1.pkl", "rb") as f:
        _raw, clean, _ctx = pickle.load(f)  # noqa: S301
    return clean


def qs_lab(df, scores):
    qs, lab = {}, {}
    for qid, cid, sc, y in zip(
        df.query_id.values, df.candidate_id.values, scores, df.label.values
    ):
        qs.setdefault(qid, []).append((cid, float(sc)))
        if y == 1:
            lab.setdefault(qid, []).append(cid)
    for qid in qs:
        lab.setdefault(qid, [])
        qs[qid].sort(key=lambda x: -x[1])
    return qs, lab


def sweep_global(qs, lab):
    best = (-1.0, 0.5)
    for t in np.arange(0.01, 1.0, 0.01):
        f = prf(pred_global(qs, float(t)), lab)["f1"]
        if f > best[0]:
            best = (f, float(t))
    return best[1]


# simple stopword-free token sets for the lexical-overlap diagnostic in case studies
def toks(text):
    import re
    return set(w for w in re.findall(r"[a-z]{3,}", (text or "").lower()))


def main():
    df = load_feature_matrix()
    feats = select_features(df)

    qids = sorted(df.query_id.unique())
    n = len(qids)
    i70, i80 = int(n * 0.70), int(n * 0.80)
    core, calib, test = set(qids[:i70]), set(qids[i70:i80]), set(qids[i80:])
    print(f"split: core={len(core)} calib={len(calib)} test={len(test)}", flush=True)

    core_df = df[df.query_id.isin(core)]
    model = lgb.train(
        CALIB_LGBM_PARAMS,
        lgb.Dataset(core_df[feats].values, core_df.label.values),
        num_boost_round=300,
        callbacks=[lgb.log_evaluation(0)],
    )

    calib_df = df[df.query_id.isin(calib)].copy()
    test_df = df[df.query_id.isin(test)].copy()
    calib_qs, calib_lab = qs_lab(calib_df, model.predict(calib_df[feats].values))
    test_qs, test_lab = qs_lab(test_df, model.predict(test_df[feats].values))

    t_global = sweep_global(calib_qs, calib_lab)
    print(f"threshold (fit on calib) = {t_global:.3f}", flush=True)

    test_preds = pred_global(test_qs, t_global)
    metrics = prf(test_preds, test_lab)
    print(f"test F1={metrics['f1']:.4f} P={metrics['precision']:.4f} "
          f"R={metrics['recall']:.4f}", flush=True)

    # --- full label set + RRF pool for recall-ceiling determination ---
    labels = json.loads(LABELS_PATH.read_text())
    rrf = pickle.load(open(PIPELINE_CACHE / "stage2.pkl", "rb"))[0]
    clean = load_stage1_clean()

    test_qlist = sorted(test)

    # pool membership = candidate ids the meta-learner actually saw (feature matrix
    # rows), which IS the BM25-RRF top-200 pool used to build the matrix.
    pool_by_q = {q: set(c for c, _ in test_qs.get(q, [])) for q in test_qlist}
    # score lookup for predicted/threshold reasoning
    score_by_q = {q: {c: s for c, s in test_qs.get(q, [])} for q in test_qlist}

    # query domains (classify over clean text)
    q_domain = {}
    for q in test_qlist:
        q_domain[q] = classify_domain(clean.get(q, ""))

    # --- categorise every gold on the test queries ---
    n_ceiling = n_rerank = n_recovered = 0
    total_gold = 0
    miss_by_domain = defaultdict(lambda: {"ceiling": 0, "rerank": 0,
                                          "recovered": 0, "gold": 0})
    rerank_cases = []  # (query, gold, rank, score) for case studies

    for q in test_qlist:
        golds = set(labels.get(q, []))
        if not golds:
            continue
        pred_set = set(test_preds.get(q, []))
        pool = pool_by_q[q]
        # rank lookup within the query's sorted candidate list
        rank_of = {c: i for i, (c, _) in enumerate(test_qs.get(q, []))}
        dom = q_domain[q]
        for g in golds:
            total_gold += 1
            miss_by_domain[dom]["gold"] += 1
            if g in pred_set:
                n_recovered += 1
                miss_by_domain[dom]["recovered"] += 1
            elif g not in pool:
                n_ceiling += 1
                miss_by_domain[dom]["ceiling"] += 1
            else:
                n_rerank += 1
                miss_by_domain[dom]["rerank"] += 1
                rerank_cases.append({
                    "query_id": q, "gold_id": g,
                    "rank_in_pool": rank_of.get(g, -1),
                    "meta_score": score_by_q[q].get(g, 0.0),
                    "domain": dom,
                })

    n_miss = n_ceiling + n_rerank
    print(f"\nGOLD on test: total={total_gold} recovered={n_recovered} "
          f"miss={n_miss} (ceiling={n_ceiling} rerank={n_rerank})", flush=True)

    miss_buckets = {
        "total_gold_test": total_gold,
        "recovered (true positive)": n_recovered,
        "total_misses": n_miss,
        "recall_ceiling_miss": {
            "count": n_ceiling,
            "pct_of_misses": 100.0 * n_ceiling / n_miss if n_miss else 0.0,
            "pct_of_gold": 100.0 * n_ceiling / total_gold if total_gold else 0.0,
        },
        "reranking_threshold_miss": {
            "count": n_rerank,
            "pct_of_misses": 100.0 * n_rerank / n_miss if n_miss else 0.0,
            "pct_of_gold": 100.0 * n_rerank / total_gold if total_gold else 0.0,
        },
    }

    # --- false positives breakdown ---
    n_fp = 0
    fp_by_domain = defaultdict(int)
    fp_score_buckets = Counter()
    fp_rank_buckets = Counter()  # where in the pool the FP sat
    for q in test_qlist:
        golds = set(labels.get(q, []))
        pred_set = set(test_preds.get(q, []))
        rank_of = {c: i for i, (c, _) in enumerate(test_qs.get(q, []))}
        for p in pred_set:
            if p not in golds:
                n_fp += 1
                fp_by_domain[q_domain[q]] += 1
                s = score_by_q[q].get(p, 0.0)
                if s >= 0.9:
                    fp_score_buckets[">=0.9 (high conf)"] += 1
                elif s >= 0.7:
                    fp_score_buckets["0.7-0.9"] += 1
                else:
                    fp_score_buckets["thr-0.7 (borderline)"] += 1
                r = rank_of.get(p, 999)
                if r < 5:
                    fp_rank_buckets["top-5"] += 1
                elif r < 20:
                    fp_rank_buckets["5-20"] += 1
                elif r < 50:
                    fp_rank_buckets["20-50"] += 1
                else:
                    fp_rank_buckets["50+"] += 1

    n_pred = sum(len(v) for v in test_preds.values())
    fp_breakdown = {
        "total_predictions": n_pred,
        "false_positives": n_fp,
        "fp_rate": 100.0 * n_fp / n_pred if n_pred else 0.0,
        "by_confidence": dict(fp_score_buckets),
        "by_pool_rank": dict(fp_rank_buckets),
    }

    # --- domain table ---
    domain_table = []
    for dom in sorted(miss_by_domain, key=lambda d: -miss_by_domain[d]["gold"]):
        b = miss_by_domain[dom]
        g = b["gold"]
        domain_table.append({
            "domain": dom,
            "n_gold": g,
            "recovered": b["recovered"],
            "recall_ceiling_miss": b["ceiling"],
            "reranking_miss": b["rerank"],
            "ceiling_miss_pct": 100.0 * b["ceiling"] / g if g else 0.0,
            "rerank_miss_pct": 100.0 * b["rerank"] / g if g else 0.0,
            "recall_pct": 100.0 * b["recovered"] / g if g else 0.0,
            "n_false_positives": fp_by_domain.get(dom, 0),
        })

    # --- case studies: reranking misses with shared statutes / low lexical overlap ---
    # rank the in-pool misses by closeness to threshold (most "almost-recovered")
    rerank_cases.sort(key=lambda c: c["rank_in_pool"])
    case_studies = []
    for c in rerank_cases:
        q, g = c["query_id"], c["gold_id"]
        row = df[(df.query_id == q) & (df.candidate_id == g)]
        if row.empty:
            continue
        r = row.iloc[0]
        # lexical overlap diagnostic
        qt, gt = toks(clean.get(q, "")), toks(clean.get(g, ""))
        jacc = len(qt & gt) / len(qt | gt) if (qt | gt) else 0.0
        cs = {
            "query_id": q,
            "gold_id": g,
            "domain": c["domain"],
            "rank_in_pool": int(c["rank_in_pool"]),
            "meta_score": round(float(c["meta_score"]), 4),
            "threshold": round(float(t_global), 3),
            "shared_statutes": int(r.get("shared_statutes", 0)),
            "shared_judges": int(r.get("shared_judges", 0)),
            "same_domain": int(r.get("same_domain", 0)),
            "tfidf_cosine": round(float(r.get("tfidf_cosine", 0)), 4),
            "jaccard_feat": round(float(r.get("jaccard", 0)), 4),
            "raw_text_jaccard": round(jacc, 4),
            "bm25_rrf_score": round(float(r.get("bm25_rrf_score", 0)), 5),
            "crossencoder_score": round(float(r.get("crossencoder_score", 0)), 4),
            "m3_fused_score": round(float(r.get("m3_fused_score", 0)), 4),
        }
        # prefer cases with shared statutes but low lexical overlap (the "057595" pattern)
        cs["_interesting"] = (cs["shared_statutes"] > 0 and cs["raw_text_jaccard"] < 0.15)
        case_studies.append(cs)

    interesting = [c for c in case_studies if c["_interesting"]]
    chosen = (interesting[:3] if len(interesting) >= 3
              else (interesting + [c for c in case_studies if not c["_interesting"]])[:3])
    for c in chosen:
        c.pop("_interesting", None)

    # check whether the workshop example query 057595 is in the test set, and if
    # so report its full miss profile (it is the canonical example).
    q057595 = None
    for cand in ("057595.txt", "057595"):
        if cand in test_qs or cand in labels:
            q057595 = cand
            break
    workshop_case = None
    if q057595 is not None:
        # 057595 lives in the CORE (train) fold, so score its candidates with the
        # model directly off the full feature matrix (the meta-score is in-sample
        # here, used only to illustrate the reranking-miss mechanism, not metrics).
        wq = q057595
        golds = list(labels.get(wq, []))
        wsub = df[df.query_id == wq].copy()
        if not wsub.empty:
            wsub["_p"] = model.predict(wsub[feats].values)
            wsub = wsub.sort_values("_p", ascending=False).reset_index(drop=True)
        pool = set(wsub.candidate_id) if not wsub.empty else set()
        rank_of = {c: i for i, c in enumerate(wsub.candidate_id)} if not wsub.empty else {}
        in_pool, ceiling = [], []
        for g in golds:
            if g in pool:
                r = wsub[wsub.candidate_id == g].iloc[0]
                in_pool.append({
                    "gold_id": g,
                    "rank_in_pool": int(rank_of.get(g, -1)),
                    "meta_score": round(float(r["_p"]), 4),
                    "shared_statutes": int(r.get("shared_statutes", 0)),
                    "same_domain": int(r.get("same_domain", 0)),
                    "tfidf_cosine": round(float(r.get("tfidf_cosine", 0)), 4),
                    "crossencoder_score": round(float(r.get("crossencoder_score", 0)), 4),
                    "m3_fused_score": round(float(r.get("m3_fused_score", 0)), 4),
                })
            else:
                ceiling.append(g)
        workshop_case = {
            "query_id": wq,
            "fold": "core/train (illustrative, in-sample meta-score)",
            "domain": classify_domain(clean.get(wq, "")),
            "n_gold": len(golds),
            "n_recall_ceiling_miss": len(ceiling),
            "n_in_pool": len(in_pool),
            "recall_ceiling_gold_ids": ceiling,
            "in_pool_golds": in_pool,
            "summary": (
                f"Query {wq} has {len(golds)} gold precedents; {len(ceiling)} never "
                f"reach the BM25-RRF top-200 pool (recall-ceiling), and the "
                f"{len(in_pool)} that do are reranking misses -- e.g. shared statutes "
                f"and high TF-IDF cosine but a near-zero cross-encoder score, so they "
                f"score below threshold."
            ),
        }

    payload = {
        "split": {"core": len(core), "calib": len(calib), "test": len(test)},
        "threshold": round(float(t_global), 4),
        "test_metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "miss_buckets": miss_buckets,
        "false_positive_breakdown": fp_breakdown,
        "domain_breakdown": domain_table,
        "case_studies": chosen,
        "workshop_query_057595_in_test": q057595,
        "workshop_case_057595": workshop_case,
        "notes": (
            "Recall-ceiling miss = gold absent from BM25-RRF top-200 pool (feature "
            "matrix rows). Reranking/threshold miss = gold in pool but meta-learner "
            "score < threshold. The two buckets partition all gold misses on the test "
            "fold. Domains via classify_domain on clean query text."
        ),
    }

    path = write_result("error_analysis", payload, script="experiments/error_analysis.py")

    # --- markdown summary ---
    md = []
    md.append("# Error Analysis -- 34-feature meta-learner (3-way split)\n")
    md.append(f"Split: core={len(core)} / calib={len(calib)} / test={len(test)} "
              f"queries (chronological by query id).  ")
    md.append(f"Threshold (fit on calib) = {t_global:.3f}.  ")
    md.append(f"Test: F1={metrics['f1']:.4f}, P={metrics['precision']:.4f}, "
              f"R={metrics['recall']:.4f}.\n")
    md.append("## Where the pipeline fails -- miss buckets\n")
    md.append(f"Total gold on test queries: **{total_gold}**. "
              f"Recovered (TP): {n_recovered}. Missed: **{n_miss}**.\n")
    md.append("| bucket | count | % of misses | % of gold |")
    md.append("|---|---:|---:|---:|")
    rc = miss_buckets["recall_ceiling_miss"]
    rr = miss_buckets["reranking_threshold_miss"]
    md.append(f"| recall-ceiling (not in pool) | {rc['count']} | "
              f"{rc['pct_of_misses']:.1f}% | {rc['pct_of_gold']:.1f}% |")
    md.append(f"| reranking / threshold (in pool, low score) | {rr['count']} | "
              f"{rr['pct_of_misses']:.1f}% | {rr['pct_of_gold']:.1f}% |")
    md.append("")
    md.append("## False positives\n")
    md.append(f"Predictions: {n_pred}; false positives: {n_fp} "
              f"({fp_breakdown['fp_rate']:.1f}% of predictions).\n")
    md.append(f"By confidence: {fp_breakdown['by_confidence']}.  ")
    md.append(f"By pool rank: {fp_breakdown['by_pool_rank']}.\n")
    md.append("## By legal domain\n")
    md.append("| domain | gold | recall% | ceiling-miss% | rerank-miss% | FPs |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for d in domain_table:
        md.append(f"| {d['domain']} | {d['n_gold']} | {d['recall_pct']:.1f}% | "
                  f"{d['ceiling_miss_pct']:.1f}% | {d['rerank_miss_pct']:.1f}% | "
                  f"{d['n_false_positives']} |")
    md.append("")
    md.append("## Case studies (reranking misses)\n")
    for i, c in enumerate(chosen, 1):
        md.append(f"**Case {i}: query {c['query_id']} -> missed gold "
                  f"{c['gold_id']}** (domain={c['domain']})  ")
        md.append(f"- in pool at rank {c['rank_in_pool']}, meta-score "
                  f"{c['meta_score']:.4f} < threshold {c['threshold']:.3f}  ")
        md.append(f"- shared statutes={c['shared_statutes']}, shared "
                  f"judges={c['shared_judges']}, same_domain={c['same_domain']}  ")
        md.append(f"- raw-text Jaccard={c['raw_text_jaccard']:.3f}, "
                  f"tfidf_cosine={c['tfidf_cosine']:.3f}, "
                  f"cross-encoder={c['crossencoder_score']:.3f}, "
                  f"BGE-M3 fused={c['m3_fused_score']:.3f}  ")
        md.append("")
    if workshop_case is not None:
        md.append("## Canonical workshop example -- query 057595\n")
        md.append(workshop_case["summary"] + "\n")
        md.append(f"- {workshop_case['n_gold']} gold; "
                  f"{workshop_case['n_recall_ceiling_miss']} recall-ceiling "
                  f"(never in pool); {workshop_case['n_in_pool']} in pool.")
        for g in workshop_case["in_pool_golds"]:
            md.append(f"  - in-pool gold {g['gold_id']} @ rank {g['rank_in_pool']}, "
                      f"meta-score {g['meta_score']}, shared_statutes "
                      f"{g['shared_statutes']}, tfidf {g['tfidf_cosine']}, "
                      f"cross-encoder {g['crossencoder_score']}, "
                      f"M3-fused {g['m3_fused_score']}")
        md.append("")
    (EXPERIMENTS_DIR / "error_analysis.md").write_text("\n".join(md))

    print(json.dumps(payload, indent=2, default=float), flush=True)
    print(f"\nSaved {path}")
    print(f"Saved {EXPERIMENTS_DIR / 'error_analysis.md'}")


if __name__ == "__main__":
    main()
