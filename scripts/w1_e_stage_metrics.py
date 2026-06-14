"""W1 experiment E: per-stage intermediate retrieval metrics.

For each pipeline stage, rank each query's BM25-RRF candidate pool by that
stage's score and compute Recall@{5,10,50,100,200}, NDCG@{5,10}, and MAP
against gold, micro- and macro-averaged over the 2001 training queries.

Reviewer 1 asked for intermediate metrics across stages: this quantifies how
ranking quality evolves stage to stage.

Run:
    uv run python scripts/w1_e_stage_metrics.py
"""
import json
import pickle
from pathlib import Path

from coliee_task1.utils.metrics import (
    mean_average_precision,
    ndcg_at_k,
    recall_at_k,
)

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "output" / "pipeline_cache"
LABELS_PATH = REPO / "data" / "task1" / "task1_train_labels_2026.json"
OUT_PATH = REPO / "output" / "w1" / "e_stage_metrics.json"

RECALL_KS = [5, 10, 50, 100, 200]
NDCG_KS = [5, 10]


def load_pickle(name: str):
    with open(CACHE / name, "rb") as f:
        return pickle.load(f)


def build_stage_rankings():
    """Return {stage_name: {query_id: [(candidate_id, score), ...]}}.

    Every stage is restricted to the BM25-RRF candidate pool per query for
    comparability. Candidates missing from a stage's score dict are dropped
    from that stage's ranking (they cannot be ranked by it).
    """
    rrf_results, _bm25_raw, bm25_rrf, _ctx_feats = load_pickle("stage2.pkl")
    biencoder = load_pickle("stage3.pkl")
    m3 = load_pickle("stage3_m3.pkl")
    crossenc = load_pickle("stage4.pkl")
    gnn_oof = load_pickle("stage5_5_oof.pkl")

    # Candidate pool per query = BM25-RRF top-200 (order preserved).
    pool = {qid: [cid for cid, _ in cands] for qid, cands in rrf_results.items()}

    stages: dict[str, dict[str, list[tuple[str, float]]]] = {}

    # Stage 1: BM25-RRF. Use bm25_rrf scores; identical to rrf_results scores.
    stages["bm25_rrf"] = {
        qid: [(cid, bm25_rrf[qid][cid]) for cid in cands if cid in bm25_rrf.get(qid, {})]
        for qid, cands in pool.items()
    }

    # Stage 2: Bi-encoder.
    stages["biencoder"] = {
        qid: [(cid, biencoder[qid][cid]) for cid in cands if cid in biencoder.get(qid, {})]
        for qid, cands in pool.items()
    }

    # Stage 3: BGE-M3 (fused signal).
    m3_fused = {qid: v.get("fused", {}) for qid, v in m3.items()}
    stages["bge_m3_fused"] = {
        qid: [(cid, m3_fused[qid][cid]) for cid in cands if cid in m3_fused.get(qid, {})]
        for qid, cands in pool.items()
    }

    # Stage 4: Cross-encoder (smart mode scores only a subset of the pool).
    stages["crossencoder"] = {
        qid: [(cid, crossenc[qid][cid]) for cid in cands if cid in crossenc.get(qid, {})]
        for qid, cands in pool.items()
    }

    # Stage 5: GNN out-of-fold.
    stages["gnn_oof"] = {
        qid: [(cid, gnn_oof[qid][cid]) for cid in cands if cid in gnn_oof.get(qid, {})]
        for qid, cands in pool.items()
    }

    return stages, pool


def compute_metrics(ranked, labels):
    row = {}
    for k in RECALL_KS:
        r = recall_at_k(ranked, labels, k)
        row[f"recall@{k}"] = r
    for k in NDCG_KS:
        n = ndcg_at_k(ranked, labels, k)
        row[f"ndcg@{k}"] = n
    row["map"] = mean_average_precision(ranked, labels)
    return row


def fmt(v):
    return f"{v:.4f}"


def print_table(results, avg="micro"):
    cols = (
        [f"R@{k}" for k in RECALL_KS]
        + [f"NDCG@{k}" for k in NDCG_KS]
        + ["MAP"]
    )
    keys = (
        [f"recall@{k}" for k in RECALL_KS]
        + [f"ndcg@{k}" for k in NDCG_KS]
        + ["map"]
    )
    header = f"{'stage':<16}" + "".join(f"{c:>10}" for c in cols)
    print(f"\n=== {avg.upper()}-averaged ===")
    print(header)
    print("-" * len(header))
    for stage, row in results.items():
        line = f"{stage:<16}"
        for key in keys:
            line += f"{fmt(row[key][avg]):>10}"
        print(line)


def main():
    print("Loading labels and caches...")
    labels = json.loads(LABELS_PATH.read_text())
    # Drop queries with no gold (none expected for train, but be safe).
    labels = {q: g for q, g in labels.items() if g}

    stages, pool = build_stage_rankings()

    # Diagnostics: pool coverage per stage.
    print(f"Queries with gold labels: {len(labels)}")
    print(f"Queries in BM25-RRF pool: {len(pool)}")
    for stage, ranked in stages.items():
        avg_cands = sum(len(v) for v in ranked.values()) / max(len(ranked), 1)
        print(f"  {stage:<16} avg pool-candidates ranked/query = {avg_cands:.1f}")

    results = {}
    for stage, ranked in stages.items():
        results[stage] = compute_metrics(ranked, labels)

    print_table(results, avg="micro")
    print_table(results, avg="macro")

    # Sanity checks.
    print("\n=== SANITY CHECKS ===")
    r200 = results["bm25_rrf"]["recall@200"]["micro"]
    check1 = abs(r200 - 0.578) < 0.02
    print(
        f"[1] BM25-RRF Recall@200 (micro) = {r200:.4f} "
        f"(expected ~0.578) -> {'PASS' if check1 else 'CHECK'}"
    )
    ce_ndcg10 = results["crossencoder"]["ndcg@10"]["micro"]
    other_max = max(
        results[s]["ndcg@10"]["micro"] for s in results if s != "crossencoder"
    )
    check2 = ce_ndcg10 >= other_max
    best = max(results, key=lambda s: results[s]["ndcg@10"]["micro"])
    print(
        f"[2] Cross-encoder NDCG@10 (micro) = {ce_ndcg10:.4f}, "
        f"max other = {other_max:.4f}, highest stage = '{best}' -> "
        f"{'PASS' if check2 else 'CHECK'}"
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_queries": len(labels),
        "recall_ks": RECALL_KS,
        "ndcg_ks": NDCG_KS,
        "stages_order": list(stages.keys()),
        "results": results,
        "sanity_checks": {
            "bm25_rrf_recall@200_micro": float(r200),
            "bm25_rrf_recall@200_pass": bool(check1),
            "crossencoder_ndcg@10_micro": float(ce_ndcg10),
            "crossencoder_highest_ndcg@10": bool(check2),
            "highest_ndcg@10_stage": best,
        },
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
