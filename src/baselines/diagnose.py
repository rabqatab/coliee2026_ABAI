"""Diagnostic script: check score distributions, threshold sweep, gold positive coverage."""
import json
import logging
import numpy as np

from baselines.common.data_loader import load_dataset
from baselines.common.bm25_index import build_shared_bm25
from coliee_task1.utils.metrics import micro_f1, scores_to_predictions

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()


def check_gold_coverage(dataset, bm25_candidates):
    """How many gold positives are in BM25 top-200?"""
    for split_name, qids in [("train", dataset.train_queries), ("val", dataset.val_queries)]:
        total_gold = 0
        found_gold = 0
        for qid in qids:
            golds = set(dataset.labels.get(qid, []))
            cand_ids = {c for c, _ in bm25_candidates.get(qid, [])}
            total_gold += len(golds)
            found_gold += len(golds & cand_ids)
        logger.info(
            "[%s] Gold coverage: %d/%d = %.1f%% of gold positives in BM25 top-200",
            split_name, found_gold, total_gold, 100 * found_gold / max(total_gold, 1),
        )


def analyze_model_scores(model_name, model_cls, dataset, bm25_candidates):
    """Train model, get score distributions, try wide threshold sweep."""
    logger.info("\n" + "=" * 70)
    logger.info("  DIAGNOSING: %s", model_name)
    logger.info("=" * 70)

    model = model_cls()
    model.train(dataset.corpus, dataset.train_queries, dataset.labels, bm25_candidates)

    # Get train scores
    train_scores = model.predict_batch(dataset.train_queries, dataset.corpus, bm25_candidates)
    val_scores = model.predict_batch(dataset.val_queries, dataset.corpus, bm25_candidates)

    for name, scores in [("train", train_scores), ("val", val_scores)]:
        all_scores = []
        for qid, cands in scores.items():
            for _, s in cands:
                all_scores.append(s)
        arr = np.array(all_scores) if all_scores else np.array([0.0])
        logger.info("[%s] Score distribution: min=%.4f, p25=%.4f, median=%.4f, p75=%.4f, max=%.4f, mean=%.4f",
                    name, arr.min(), np.percentile(arr, 25), np.median(arr),
                    np.percentile(arr, 75), arr.max(), arr.mean())

    # Wide threshold sweep
    train_labels = {q: dataset.labels[q] for q in dataset.train_queries if q in dataset.labels}
    val_labels = {q: dataset.labels[q] for q in dataset.val_queries if q in dataset.labels}

    # Get score range from train scores
    all_train_scores = [s for cands in train_scores.values() for _, s in cands]
    if all_train_scores:
        score_min = min(all_train_scores)
        score_max = max(all_train_scores)
        # Sweep across actual score range
        thresholds = np.linspace(max(score_min, -10), score_max, 200)
    else:
        thresholds = np.arange(0.01, 1.0, 0.01)

    best_train_f1 = -1
    best_train_t = 0
    best_val_f1 = -1
    best_val_t = 0
    for t in thresholds:
        t = float(t)
        train_preds = scores_to_predictions(train_scores, t)
        train_m = micro_f1(train_preds, train_labels)
        if train_m["f1"] > best_train_f1:
            best_train_f1 = train_m["f1"]
            best_train_t = t

        val_preds = scores_to_predictions(val_scores, t)
        val_m = micro_f1(val_preds, val_labels)
        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            best_val_t = t
            best_val_metrics = val_m

    logger.info("[WIDE SWEEP] Best train: t=%.4f F1=%.4f", best_train_t, best_train_f1)
    logger.info("[WIDE SWEEP] Best val:   t=%.4f F1=%.4f P=%.4f R=%.4f",
                best_val_t, best_val_f1,
                best_val_metrics.get("precision", 0), best_val_metrics.get("recall", 0))

    # Also check: what's the oracle val F1 (threshold optimized on val itself)?
    logger.info("[ORACLE] Val F1 with val-optimal threshold")

    # Check per-query prediction counts at best train threshold
    train_preds = scores_to_predictions(train_scores, best_train_t)
    val_preds_best = scores_to_predictions(val_scores, best_train_t)
    n_preds = [len(v) for v in val_preds_best.values()]
    logger.info("[VAL @ train_t=%.4f] Queries with 0 preds: %d/%d, avg preds/query: %.1f",
                best_train_t, sum(1 for x in n_preds if x == 0), len(n_preds),
                np.mean(n_preds) if n_preds else 0)


def main():
    dataset = load_dataset()
    _, bm25_candidates = build_shared_bm25(dataset.corpus)

    check_gold_coverage(dataset, bm25_candidates)

    # Only diagnose the fast baselines (skip CaseLink/UMNLP which take forever)
    from baselines.bm25.model import VanillaBM25
    analyze_model_scores("BM25 (vanilla)", VanillaBM25, dataset, bm25_candidates)

    from baselines.jnlp.model import JNLPBaseline
    analyze_model_scores("JNLP 2025", JNLPBaseline, dataset, bm25_candidates)

    from baselines.tqm.model import TQMBaseline
    analyze_model_scores("TQM 2024", TQMBaseline, dataset, bm25_candidates)


if __name__ == "__main__":
    main()
