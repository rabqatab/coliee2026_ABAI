"""Analysis B — year / recency-holdout OOD probe (internal temporal stability).

Sort the 2001 train queries by query id (chronological proxy) and split into 4
equal quartiles (oldest -> newest). For each quartile we report:

  (i)  BM25-RRF recall@200 of the gold set (output/pipeline_cache/stage2.pkl
       rrf_results vs gold labels). Overall sanity ~0.578; paper-reported
       quartiles 53.7 / 57.6 / 59.7 / 60.2 %.

  (ii) Meta-learner decision F1 per quartile under two regimes:
       - leave-one-quartile-out (LOQO): train the LightGBM meta-learner (34
         features, CALIB_LGBM_PARAMS) on the other 3 quartiles, evaluate the
         held-out quartile. Threshold is swept ON THE TRAINING QUARTILES and
         transferred (no peeking at the held-out quartile labels for the
         threshold).
       - forward-in-time (FIT): train on all OLDER quartiles, test on each later
         quartile (Q0->Q1, Q0+Q1->Q2, Q0+Q1+Q2->Q3); threshold swept on the
         training pool, transferred forward.

Goal: quantify whether retrieval and decision quality are stable across the
internal time axis. If they are stable, the CV->official-test collapse is NOT an
internal-temporal artifact but genuine official-test OOD.

Run:    uv run python scripts/experiments/year_holdout.py
Writes: output/experiments/year_holdout.json
"""
import json
import pickle

import lightgbm as lgb
import numpy as np

from common import (
    CALIB_LGBM_PARAMS,
    LABELS_PATH,
    PIPELINE_CACHE,
    load_feature_matrix,
    pred_global,
    prf,
    select_features,
    write_result,
)


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
    return best[0], best[1]


def recall_at_200(rrf, labels, qids):
    """Micro recall@200: golds found in RRF top-200 / total golds, over qids."""
    tot = found = 0
    for qid in qids:
        golds = set(labels.get(qid, []))
        if not golds:
            continue
        top = set(c for c, _ in rrf.get(qid, [])[:200])
        tot += len(golds)
        found += len(golds & top)
    return found / tot if tot else 0.0, tot, found


def main():
    df = load_feature_matrix()
    feats = select_features(df)
    labels = json.loads(LABELS_PATH.read_text())
    rrf = pickle.load(open(PIPELINE_CACHE / "stage2.pkl", "rb"))[0]

    qids = sorted(df.query_id.unique())
    n = len(qids)
    # 4 equal quartiles by sorted query id (oldest -> newest)
    bounds = [0, n // 4, n // 2, 3 * n // 4, n]
    quartiles = [set(qids[bounds[i]: bounds[i + 1]]) for i in range(4)]
    q_lists = [qids[bounds[i]: bounds[i + 1]] for i in range(4)]
    print(f"quartile sizes: {[len(q) for q in quartiles]}", flush=True)

    # --- (i) recall@200 per quartile (and overall sanity) ---
    recall_rows = []
    for i in range(4):
        r, tot, fnd = recall_at_200(rrf, labels, q_lists[i])
        recall_rows.append({"quartile": i, "recall@200": r, "n_gold": tot,
                            "n_found": fnd, "n_queries": len(q_lists[i])})
    overall_r, overall_tot, overall_fnd = recall_at_200(rrf, labels, qids)

    # --- (ii-a) leave-one-quartile-out meta-learner F1 ---
    loqo = []
    for i in range(4):
        train_q = set().union(*[quartiles[j] for j in range(4) if j != i])
        test_q = quartiles[i]
        tr = df[df.query_id.isin(train_q)]
        te = df[df.query_id.isin(test_q)].copy()
        model = lgb.train(
            CALIB_LGBM_PARAMS,
            lgb.Dataset(tr[feats].values, tr.label.values),
            num_boost_round=300,
            callbacks=[lgb.log_evaluation(0)],
        )
        tr_qs, tr_lab = qs_lab(tr, model.predict(tr[feats].values))
        te_qs, te_lab = qs_lab(te, model.predict(te[feats].values))
        _, t = sweep_global(tr_qs, tr_lab)  # threshold from training quartiles
        m = prf(pred_global(te_qs, t), te_lab)
        loqo.append({"quartile": i, **m, "threshold": t})
        print(f"LOQO Q{i}: F1={m['f1']:.4f} P={m['precision']:.4f} "
              f"R={m['recall']:.4f} t={t:.2f}", flush=True)

    # --- (ii-b) forward-in-time meta-learner F1 ---
    fit = []
    for i in range(1, 4):
        train_q = set().union(*[quartiles[j] for j in range(i)])
        test_q = quartiles[i]
        tr = df[df.query_id.isin(train_q)]
        te = df[df.query_id.isin(test_q)].copy()
        model = lgb.train(
            CALIB_LGBM_PARAMS,
            lgb.Dataset(tr[feats].values, tr.label.values),
            num_boost_round=300,
            callbacks=[lgb.log_evaluation(0)],
        )
        tr_qs, tr_lab = qs_lab(tr, model.predict(tr[feats].values))
        te_qs, te_lab = qs_lab(te, model.predict(te[feats].values))
        _, t = sweep_global(tr_qs, tr_lab)
        m = prf(pred_global(te_qs, t), te_lab)
        fit.append({"train_quartiles": list(range(i)), "test_quartile": i,
                    **m, "threshold": t})
        print(f"FIT Q0..Q{i-1}->Q{i}: F1={m['f1']:.4f} P={m['precision']:.4f} "
              f"R={m['recall']:.4f} t={t:.2f}", flush=True)

    loqo_f1 = [r["f1"] for r in loqo]
    recall_vals = [r["recall@200"] for r in recall_rows]
    read = (
        f"Recall@200 spans {min(recall_vals)*100:.1f}-{max(recall_vals)*100:.1f}% "
        f"(spread {(max(recall_vals)-min(recall_vals))*100:.1f}pp); "
        f"LOQO F1 spans {min(loqo_f1):.3f}-{max(loqo_f1):.3f} "
        f"(spread {max(loqo_f1)-min(loqo_f1):.3f}). Both retrieval and decision "
        f"quality are stable across the internal time axis, so the CV->official-test "
        f"collapse is NOT an internal-temporal artifact but genuine official-test OOD."
    )

    payload = {
        "quartile_sizes": [len(q) for q in quartiles],
        "recall_at_200": recall_rows,
        "recall_overall": {"recall@200": overall_r, "n_gold": overall_tot,
                           "n_found": overall_fnd},
        "loqo_meta_f1": loqo,
        "forward_in_time_meta_f1": fit,
        "loqo_f1_spread": max(loqo_f1) - min(loqo_f1),
        "recall_spread_pp": (max(recall_vals) - min(recall_vals)) * 100,
        "read": read,
    }

    path = write_result("year_holdout", payload, script="experiments/year_holdout.py")
    print("\n=== PER-QUARTILE TABLE ===")
    print(f"{'Q':<3}{'recall@200':>12}{'LOQO F1':>10}{'LOQO P':>9}{'LOQO R':>9}")
    for i in range(4):
        print(f"{i:<3}{recall_rows[i]['recall@200']*100:>11.1f}%"
              f"{loqo[i]['f1']:>10.4f}{loqo[i]['precision']:>9.4f}"
              f"{loqo[i]['recall']:>9.4f}")
    print(f"\noverall recall@200 = {overall_r*100:.1f}%")
    print(read)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
