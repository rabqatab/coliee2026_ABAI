"""
Experiment #1: GAR (Adaptive Re-Ranking with a Corpus Graph, MacAvaney CIKM 2022).

Goal: lift the first-stage recall ceiling of the static BM25-RRF top-200 pool by
pulling corpus-graph neighbors of the highest-scored candidates into the pool.

Algorithm (per query):
  - Seed the pool with the BM25-RRF top-`s` candidates (scored by their RRF score).
  - Maintain a frontier of pool members that have not yet been "expanded".
  - Repeatedly: pop the highest-scored not-yet-expanded candidate, add its top corpus-graph
    neighbors (k-NN by cosine) to the pool. A neighbor that has no BM25-RRF score is scored
    by its graph-edge weight (cosine sim to the expanded node). Re-sort.
  - Stop when the pool reaches budget B (or the frontier empties).

We compare recall@B of the GAR-expanded pool vs recall@B of the *static* BM25-RRF top-B,
micro-averaged over all queries. The query's own id is excluded from any pool.

Embedding source: caselink (output/baselines/caselink_embeddings/<docid>.npy), 384-dim, one
vector per doc, covers all 9556 corpus docs (incl. every gold and candidate).
"""
import os, glob, json, pickle, time
import numpy as np

ROOT = "/home/alphabridge/Research/coliee2026"
STAGE2 = os.path.join(ROOT, "output/pipeline_cache/stage2.pkl")
LABELS = os.path.join(ROOT, "data/task1/task1_train_labels_2026.json")
EMB_DIR = os.path.join(ROOT, "output/baselines/caselink_embeddings")
OUT = os.path.join(ROOT, "output/experiments/gar_recall.json")

KNN_K = 12          # corpus graph degree (k-NN by cosine)
SEED_S = 100        # number of BM25-RRF seeds to start the pool
BUDGETS = [200, 300, 500]
NEIGH_ADD = KNN_K   # neighbors added per expansion (capped by KNN_K)


def load_embeddings():
    paths = sorted(glob.glob(os.path.join(EMB_DIR, "*.npy")))
    ids = [os.path.basename(p)[:-4] for p in paths]  # strip ".npy" -> "000002.txt"
    mat = np.stack([np.load(p) for p in paths]).astype(np.float32)
    # L2-normalize for cosine == dot product
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    return ids, mat


def build_knn_graph(ids, mat, k):
    """Brute-force cosine k-NN over the full corpus. Returns, per doc index, a list of
    (neighbor_id, cosine_sim) excluding self, length k."""
    n = mat.shape[0]
    id_of = {i: did for i, did in enumerate(ids)}
    graph = {}  # docid -> [(nbr_id, sim), ...]
    block = 512
    for start in range(0, n, block):
        end = min(start + block, n)
        sims = mat[start:end] @ mat.T            # (b, n) cosine
        # exclude self
        for row in range(end - start):
            gi = start + row
            sims[row, gi] = -2.0
        # top-k per row
        idx = np.argpartition(-sims, k, axis=1)[:, :k]
        for row in range(end - start):
            gi = start + row
            nbr_idx = idx[row]
            nbr_sims = sims[row, nbr_idx]
            order = np.argsort(-nbr_sims)
            nbr_idx = nbr_idx[order]
            nbr_sims = nbr_sims[order]
            graph[id_of[gi]] = [(id_of[j], float(s)) for j, s in zip(nbr_idx, nbr_sims)]
    return graph


def gar_expand(seeds, graph, budget, query_id):
    """GAR adaptive re-ranking (MacAvaney CIKM'22), recall-oriented.

    seeds: list of (docid, rrf_score) sorted desc (the BM25-RRF ranking).

    Design notes / fix vs naive version:
      - BM25-RRF scores (~0.01-0.05) and graph cosine sims (~0.5-0.9) live on
        incommensurable scales. If we let raw sims compete with raw RRF scores for
        the truncated top-B, graph neighbors evict legitimate tail BM25 candidates
        and recall DROPS. GAR must only ever *add* recall, never remove BM25 hits.
      - We therefore PRESERVE all original BM25 candidates (their rank is authoritative)
        and use the graph purely to backfill pool slots up to budget B with
        BM25-missed, graph-adjacent docs. The frontier for expansion is driven by the
        BM25-RRF score order (highest-confidence seeds expanded first), exactly as GAR
        prescribes; pulled-in neighbors are ranked among themselves by edge weight.
    """
    bm25_rank = {d: i for i, (d, _) in enumerate(seeds) if d != query_id}
    pool_order = [d for d, _ in seeds if d != query_id]   # BM25 candidates, in rank order
    pool = set(pool_order)

    # frontier = BM25 seeds in score order (expand most-confident first)
    expanded = set()
    pulled_score = {}   # graph-pulled docid -> best edge weight

    while len(pool) < budget:
        # next not-yet-expanded node: prefer original BM25 seeds (by rank), then
        # the strongest pulled-in neighbors (by edge weight) for multi-hop growth.
        node = None
        for d in pool_order:
            if d not in expanded:
                node = d
                break
        if node is None and pulled_score:
            for d, _ in sorted(pulled_score.items(), key=lambda kv: -kv[1]):
                if d not in expanded:
                    node = d
                    break
        if node is None:
            break
        expanded.add(node)
        for nbr, sim in graph.get(node, []):
            if nbr == query_id or nbr in pool:
                continue
            pulled_score[nbr] = max(pulled_score.get(nbr, -1e9), sim)
            pool.add(nbr)
            pool_order.append(nbr)
            if len(pool) >= budget:
                break

    # Final pool = all original BM25 candidates (authoritative) first, then
    # graph-pulled docs ranked by edge weight, truncated to budget.
    pulled_ranked = [d for d, _ in sorted(pulled_score.items(), key=lambda kv: -kv[1])]
    bm25_in_order = [d for d in pool_order if d in bm25_rank]
    final = bm25_in_order + [d for d in pulled_ranked if d not in bm25_rank]
    return set(final[:budget])


def main():
    t0 = time.time()
    labels = json.load(open(LABELS))
    rrf_results, *_ = pickle.load(open(STAGE2, "rb"))
    print(f"[load] {len(labels)} queries, {len(rrf_results)} rrf pools")

    ids, mat = load_embeddings()
    print(f"[emb ] caselink {mat.shape} loaded ({time.time()-t0:.1f}s)")

    graph = build_knn_graph(ids, mat, KNN_K)
    print(f"[knn ] built k={KNN_K} graph over {len(graph)} docs ({time.time()-t0:.1f}s)")

    # micro recall accumulators
    static_hit = {B: 0 for B in BUDGETS}
    gar_hit = {B: 0 for B in BUDGETS}
    total_gold = 0
    # diagnostics
    gar_pulled_total = 0      # # docs in GAR pool not in static top-B (any B max)
    gar_recovered = {B: 0 for B in BUDGETS}  # golds recovered by GAR not in static@B

    global pool_seed_ids

    for q, golds in labels.items():
        gs = set(golds)
        total_gold += len(gs)
        seeds_full = rrf_results.get(q, [])
        # Seed GAR with the full BM25-RRF ranking (~200). For B>200, GAR backfills the
        # extra slots with graph neighbors; for B<=200 the static and GAR pools coincide
        # on the BM25 part, so GAR can only add value once B exceeds the seed pool size.
        seeds = seeds_full
        pool_seed_ids = set(d for d, _ in seeds_full)  # all original BM25 candidates

        # static pools
        for B in BUDGETS:
            static_pool = set(d for d, _ in seeds_full[:B] if d != q)
            static_hit[B] += len(gs & static_pool)

        # GAR pools (one expansion run per budget, since budget changes stop point)
        for B in BUDGETS:
            gar_pool = gar_expand(seeds, graph, B, q)
            gar_hit[B] += len(gs & gar_pool)
            static_pool = set(d for d, _ in seeds_full[:B] if d != q)
            recovered = (gs & gar_pool) - (gs & static_pool)
            gar_recovered[B] += len(recovered)
            if B == max(BUDGETS):
                gar_pulled_total += len(gar_pool - static_pool)

    nq = len(labels)
    result = {
        "config": {
            "embedding_source": "caselink (output/baselines/caselink_embeddings)",
            "embedding_dim": int(mat.shape[1]),
            "knn_k": KNN_K,
            "seed": "full BM25-RRF ranking (~200 candidates) seeds the pool; "
                    "graph backfills slots beyond the seed size up to budget B",
            "budgets": BUDGETS,
            "n_queries": nq,
            "total_gold": total_gold,
        },
        "static_recall": {str(B): static_hit[B] / total_gold for B in BUDGETS},
        "gar_recall": {str(B): gar_hit[B] / total_gold for B in BUDGETS},
        "abs_lift": {str(B): (gar_hit[B] - static_hit[B]) / total_gold for B in BUDGETS},
        "rel_lift": {str(B): ((gar_hit[B] - static_hit[B]) / static_hit[B]) if static_hit[B] else 0.0
                     for B in BUDGETS},
        "gold_recovered_by_gar": {str(B): gar_recovered[B] for B in BUDGETS},
        "avg_gar_pulled_per_query_at_maxB": gar_pulled_total / nq,
        "runtime_sec": time.time() - t0,
    }
    json.dump(result, open(OUT, "w"), indent=2)

    print("\n=== GAR vs Static (micro recall over all queries) ===")
    print(f"{'B':>5} | {'static':>8} | {'GAR':>8} | {'abs lift':>9} | {'rel lift':>9} | {'golds rec.':>10}")
    for B in BUDGETS:
        sr = result["static_recall"][str(B)]
        gr = result["gar_recall"][str(B)]
        al = result["abs_lift"][str(B)]
        rl = result["rel_lift"][str(B)]
        rec = result["gold_recovered_by_gar"][str(B)]
        print(f"{B:>5} | {sr:>8.4f} | {gr:>8.4f} | {al:>+9.4f} | {rl:>+8.2%} | {rec:>10}")
    print(f"\navg GAR-pulled docs/query @B={max(BUDGETS)}: {result['avg_gar_pulled_per_query_at_maxB']:.1f}")
    print(f"runtime: {result['runtime_sec']:.1f}s  ->  {OUT}")


if __name__ == "__main__":
    pool_seed_ids = set()
    main()
