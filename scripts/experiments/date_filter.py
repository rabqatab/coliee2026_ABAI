"""Part B -- temporal / date precedent filter (BJPWH / mezza style).

A cited precedent must PREDATE the citing case. We extract each case's decision
year via regex over the clean corpus text (Canadian neutral-citation patterns
like "2015 FC 725", with bracket-year and max-year fallbacks), then on the
BM25-RRF candidate pool we DROP candidates whose year > the query year (keeping
undated candidates to protect recall). We measure:

  (i)  pool recall@200 before vs after the filter (Delta recall);
  (ii) the meta-learner test F1/P/R before vs after, on the same clean 3-way
       chronological split as calibration.py / error_analysis.py;
  (iii) a SELECTIVE variant (BJPWH finding: filtering the dense / non-lexical
       signal helps more than filtering BM25). We only drop a date-violating
       candidate when it is NOT lexically supported -- i.e. it was surfaced by
       the dense / cross-encoder / graph stages rather than by full-doc BM25
       (bm25_raw). Lexical (BM25) hits are kept regardless of year.

Year extraction cascade (decision year proxy = latest dated event the opinion
references, which for a judgment is its own date):
  1. neutral citation year   "2015 FC 725"   (most reliable)
  2. bracket year            "[2015] 1 FCR"
  3. max plausible year      any 1950-2029 token

Run:    uv run python scripts/experiments/date_filter.py
Writes: output/experiments/date_filter.json
"""
import json
import pickle
import re
import sys
from collections import Counter

import lightgbm as lgb
import numpy as np

from common import (
    CALIB_LGBM_PARAMS,
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

NEUTRAL = re.compile(
    r"\b(19[5-9]\d|20[0-2]\d)\s+"
    r"(?:SCC|FC|FCA|FCT|CA|ONCA|ONSC|ONCJ|BCCA|BCSC|ABCA|ABQB|ABPC|QCCA|QCCS|"
    r"SKCA|SKQB|MBCA|MBQB|NSCA|NSSC|NBCA|NBQB|NLCA|NLTD|PECA|YKCA|NWTSC|TCC|"
    r"CHRT|CanLII)\b"
)
BRACKET = re.compile(r"\[(19[5-9]\d|20[0-2]\d)\]")
ANYYEAR = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")


def extract_year(text: str):
    """Return (year, source) or (None, 'none')."""
    t = text or ""
    m = NEUTRAL.findall(t)
    if m:
        return max(int(x) for x in m), "neutral"
    m = BRACKET.findall(t)
    if m:
        return max(int(x) for x in m), "bracket"
    m = ANYYEAR.findall(t)
    if m:
        return max(int(x) for x in m), "maxyear"
    return None, "none"


def load_stage1_clean():
    import coliee_task1
    import coliee_task1.stages.citation_context as cc
    sys.modules.setdefault("graphrag", coliee_task1)
    sys.modules.setdefault("graphrag.stages", coliee_task1.stages)
    sys.modules.setdefault("graphrag.citation_context", cc)
    sys.modules.setdefault("graphrag.stages.citation_context", cc)
    with open(PIPELINE_CACHE / "stage1.pkl", "rb") as f:  # noqa: S301 -- internal cache
        _raw, clean, _ctx = pickle.load(f)
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


def recall_at_k(pool_by_q, labels, qids, k=200):
    tot = found = 0
    for q in qids:
        golds = set(labels.get(q, []))
        if not golds:
            continue
        top = set(list(pool_by_q.get(q, []))[:k])
        tot += len(golds)
        found += len(golds & top)
    return (found / tot if tot else 0.0), tot, found


def main():
    df = load_feature_matrix()
    feats = select_features(df)
    labels = json.loads(LABELS_PATH.read_text())
    clean = load_stage1_clean()
    rrf, bm25_raw, _bm25_rrf, _ctx = pickle.load(
        open(PIPELINE_CACHE / "stage2.pkl", "rb")  # noqa: S301 -- internal cache
    )

    # --- (0) extract years for the whole corpus ---
    year, src_count = {}, Counter()
    for d, t in clean.items():
        y, s = extract_year(t)
        src_count[s] += 1
        if y is not None:
            year[d] = y
    coverage = len(year) / len(clean)
    print(f"year extraction: {len(year)}/{len(clean)} = {coverage:.3f} "
          f"({dict(src_count)})", flush=True)

    qids_all = sorted(labels.keys())

    # --- (i) pool recall@200 before/after filter (full-RRF pool) ---
    pool_base = {q: [c for c, _ in rrf.get(q, [])] for q in qids_all}

    def violates(q, c):
        """True if candidate c is dated strictly AFTER query q (so it cannot be
        a precedent). Undated query or candidate -> never filtered (protect recall)."""
        qy, cy = year.get(q), year.get(c)
        return (qy is not None) and (cy is not None) and (cy > qy)

    pool_full = {q: [c for c, _ in rrf.get(q, []) if not violates(q, c)]
                 for q in qids_all}

    # selective: drop date-violating candidate ONLY if NOT lexically supported
    # (not present in full-doc BM25 raw results for that query).
    pool_sel = {}
    for q in qids_all:
        lex = set(bm25_raw.get(q, {}).keys())
        keep = []
        for c, _ in rrf.get(q, []):
            if violates(q, c) and c not in lex:
                continue
            keep.append(c)
        pool_sel[q] = keep

    r_base, tot, f_base = recall_at_k(pool_base, labels, qids_all)
    r_full, _, f_full = recall_at_k(pool_full, labels, qids_all)
    r_sel, _, f_sel = recall_at_k(pool_sel, labels, qids_all)
    print(f"recall@200  base={r_base:.4f}  full-filter={r_full:.4f}  "
          f"selective={r_sel:.4f}", flush=True)

    # how many candidates dropped + how many GOLD violators (would-be future cites)
    n_drop_full = sum(len(pool_base[q]) - len(pool_full[q]) for q in qids_all)
    n_drop_sel = sum(len(pool_base[q]) - len(pool_sel[q]) for q in qids_all)
    gold_violators = sum(
        1 for q in qids_all for g in labels.get(q, [])
        if violates(q, g)
    )
    gold_in_pool = sum(
        1 for q in qids_all for g in set(labels.get(q, []))
        if g in set(pool_base[q][:200])
    )
    gold_in_pool_viol = sum(
        1 for q in qids_all for g in set(labels.get(q, []))
        if g in set(pool_base[q][:200]) and violates(q, g)
    )

    # --- (ii) meta-learner test F1/P/R before/after, on the 3-way split ---
    qids = sorted(df.query_id.unique())
    n = len(qids)
    i70, i80 = int(n * 0.70), int(n * 0.80)
    core, calib, test = set(qids[:i70]), set(qids[i70:i80]), set(qids[i80:])

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

    def apply_filter(qs, mode):
        """Remove date-violating candidates from a {qid:[(cid,score)]} dict.
        mode='full' drops all violators; mode='selective' keeps lexical (BM25) hits."""
        out = {}
        for q, cs in qs.items():
            lex = set(bm25_raw.get(q, {}).keys())
            kept = []
            for c, s in cs:
                if violates(q, c) and (mode == "full" or c not in lex):
                    continue
                kept.append((c, s))
            out[q] = kept
        return out

    def eval_meta(filter_mode):
        cq = calib_qs if filter_mode is None else apply_filter(calib_qs, filter_mode)
        tq = test_qs if filter_mode is None else apply_filter(test_qs, filter_mode)
        t = sweep_global(cq, calib_lab)
        m = prf(pred_global(tq, t), test_lab)
        return {**{k: round(float(v), 4) for k, v in m.items()},
                "threshold": round(float(t), 3)}

    meta_base = eval_meta(None)
    meta_full = eval_meta("full")
    meta_sel = eval_meta("selective")
    print(f"meta test  base F1={meta_base['f1']}  full F1={meta_full['f1']}  "
          f"selective F1={meta_sel['f1']}", flush=True)

    # year coverage restricted to query/candidate populations actually used
    q_dated = sum(1 for q in qids_all if q in year)
    payload = {
        "year_extraction": {
            "corpus_size": len(clean),
            "dated": len(year),
            "coverage": round(coverage, 4),
            "source_mix": dict(src_count),
            "queries_dated": q_dated,
            "queries_total": len(qids_all),
            "year_range": [int(min(year.values())), int(max(year.values()))],
            "median_year": int(np.median(list(year.values()))),
        },
        "pool_recall_at_200": {
            "baseline": round(r_base, 4),
            "full_filter": round(r_full, 4),
            "selective_filter": round(r_sel, 4),
            "delta_full": round(r_full - r_base, 4),
            "delta_selective": round(r_sel - r_base, 4),
            "total_gold": tot,
            "candidates_dropped_full": n_drop_full,
            "candidates_dropped_selective": n_drop_sel,
            "gold_pairs_dated_after_query (future-cite label noise)": gold_violators,
            "gold_in_pool_top200": gold_in_pool,
            "gold_in_pool_top200_that_violate_date": gold_in_pool_viol,
        },
        "meta_learner_test": {
            "baseline": meta_base,
            "full_filter": meta_full,
            "selective_filter": meta_sel,
            "delta_f1_full": round(meta_full["f1"] - meta_base["f1"], 4),
            "delta_f1_selective": round(meta_sel["f1"] - meta_base["f1"], 4),
        },
        "split": {"core": len(core), "calib": len(calib), "test": len(test)},
        "notes": (
            "violates(q,c) = year[c] > year[q] (both dated). Undated kept to protect "
            "recall. Full filter drops all date-violating pool candidates; selective "
            "filter (BJPWH) keeps full-doc-BM25 (lexical) hits and only drops "
            "date-violating dense/CE/graph-surfaced candidates. Meta test uses the same "
            "clean 3-way chronological split as calibration.py; threshold re-swept on "
            "calib for each filter mode."
        ),
    }

    path = write_result("date_filter", payload, script="experiments/date_filter.py")
    print(json.dumps(payload, indent=2, default=float), flush=True)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
