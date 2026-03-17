"""GraphRAG Lite: regex-based entity graph + Leiden community features.

Builds a bipartite document-entity graph using regex extraction only
(no LLM needed), projects it to a document-document similarity graph,
runs multi-resolution Leiden community detection, and computes community
co-membership features for the meta-learner.

Based on the SAP Practical GraphRAG finding that regex achieves 94% of
LLM extraction quality, and the E2GraphRAG architecture for fast indexing.
"""
import logging
from collections import defaultdict
from dataclasses import dataclass, field

import igraph as ig
import leidenalg
import numpy as np

from coliee_task1.config import (
    BIPARTITE_WEIGHTS,
    LEIDEN_RESOLUTIONS,
    DOMAIN_KEYWORDS,
)
from coliee_task1.utils.extract_regex import extract_statutes, extract_judges, extract_outcome
from coliee_task1.utils.normalize import normalize_statute, normalize_judge

logger = logging.getLogger(__name__)


@dataclass
class EntityRecord:
    """Extracted entities for a single document."""
    doc_id: str
    statutes: list[str] = field(default_factory=list)
    judges: list[str] = field(default_factory=list)
    domain: str = "other"
    outcome: str | None = None


def classify_domain(text: str) -> str:
    """Classify a case into a legal domain using keyword matching."""
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in text_lower)
    if not scores or max(scores.values()) == 0:
        return "other"
    return max(scores, key=scores.get)


def extract_entities_regex(doc_id: str, text: str) -> EntityRecord:
    """Extract all entity types from a document using regex only."""
    statutes = [normalize_statute(s) for s in extract_statutes(text)]
    judges = [normalize_judge(j) for j in extract_judges(text)]
    domain = classify_domain(text)
    outcome = extract_outcome(text)
    return EntityRecord(
        doc_id=doc_id,
        statutes=sorted(set(statutes)),
        judges=sorted(set(judges)),
        domain=domain,
        outcome=outcome,
    )


def build_entity_records(corpus: dict[str, str]) -> dict[str, EntityRecord]:
    """Extract entities from all documents in the corpus."""
    records = {}
    for doc_id, text in corpus.items():
        records[doc_id] = extract_entities_regex(doc_id, text)
    logger.info("Extracted entities from %d documents", len(records))

    # Stats
    n_statutes = sum(len(r.statutes) for r in records.values())
    n_judges = sum(len(r.judges) for r in records.values())
    domains = defaultdict(int)
    for r in records.values():
        domains[r.domain] += 1
    logger.info(
        "  Total statutes: %d, judges: %d, domain dist: %s",
        n_statutes, n_judges, dict(domains),
    )
    return records


def build_bipartite_graph(
    records: dict[str, EntityRecord],
    weights: dict[str, float] | None = None,
) -> ig.Graph:
    """Build a bipartite graph connecting documents to their entities.

    Edge weights are determined by entity type:
    - statute edges: BIPARTITE_WEIGHTS["statute"]
    - judge edges: BIPARTITE_WEIGHTS["judge"]
    - domain edges: BIPARTITE_WEIGHTS["domain"]
    - outcome edges: BIPARTITE_WEIGHTS["outcome"]

    Returns:
        igraph Graph with vertex attribute 'type' (True=entity, False=doc)
        and 'name', plus edge attribute 'weight'.
    """
    if weights is None:
        weights = BIPARTITE_WEIGHTS

    # Collect all unique entities
    doc_ids = sorted(records.keys())
    entities: dict[str, str] = {}  # entity_key -> entity_type

    for rec in records.values():
        for s in rec.statutes:
            entities[f"statute:{s}"] = "statute"
        for j in rec.judges:
            entities[f"judge:{j}"] = "judge"
        entities[f"domain:{rec.domain}"] = "domain"
        if rec.outcome:
            entities[f"outcome:{rec.outcome}"] = "outcome"

    entity_keys = sorted(entities.keys())

    # Build vertex list: docs first, then entities
    n_docs = len(doc_ids)
    n_entities = len(entity_keys)
    n_vertices = n_docs + n_entities

    doc_idx = {did: i for i, did in enumerate(doc_ids)}
    ent_idx = {ek: n_docs + i for i, ek in enumerate(entity_keys)}

    # Build edges
    edges = []
    edge_weights = []

    for rec in records.values():
        di = doc_idx[rec.doc_id]
        for s in rec.statutes:
            ei = ent_idx[f"statute:{s}"]
            edges.append((di, ei))
            edge_weights.append(weights.get("statute", 0.5))
        for j in rec.judges:
            ei = ent_idx[f"judge:{j}"]
            edges.append((di, ei))
            edge_weights.append(weights.get("judge", 0.15))
        ei = ent_idx[f"domain:{rec.domain}"]
        edges.append((di, ei))
        edge_weights.append(weights.get("domain", 0.1))
        if rec.outcome:
            ei = ent_idx[f"outcome:{rec.outcome}"]
            edges.append((di, ei))
            edge_weights.append(weights.get("outcome", 0.05))

    g = ig.Graph(n=n_vertices, edges=edges, directed=False)
    g.vs["name"] = doc_ids + entity_keys
    g.vs["type"] = [False] * n_docs + [True] * n_entities
    g.es["weight"] = edge_weights

    logger.info(
        "Bipartite graph: %d docs, %d entities, %d edges",
        n_docs, n_entities, len(edges),
    )
    return g


def project_to_doc_graph(
    bipartite: ig.Graph,
    max_entity_degree: int = 500,
) -> ig.Graph:
    """Project bipartite graph onto document-document similarity graph.

    Two documents share an edge if they share at least one entity.
    Edge weight = sum of shared entity edge weights.

    Entities with more than max_entity_degree document connections are
    skipped — they are too common to be discriminative and would create
    O(n^2) edges that overwhelm community detection.
    """
    doc_vertices = [v.index for v in bipartite.vs if not v["type"]]
    n_docs = len(doc_vertices)

    # Build doc-doc adjacency via shared entities
    doc_neighbors: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))

    entity_vertices = [v.index for v in bipartite.vs if v["type"]]
    n_skipped = 0
    for ev in entity_vertices:
        # Find all doc neighbors of this entity
        neighbors = bipartite.neighbors(ev)
        neighbor_weights = {}
        for n in neighbors:
            if not bipartite.vs[n]["type"]:  # is a doc
                eid = bipartite.get_eid(n, ev)
                neighbor_weights[n] = bipartite.es[eid]["weight"]

        # Skip overly common entities (analogous to stop words)
        if len(neighbor_weights) > max_entity_degree:
            n_skipped += 1
            continue

        # Add pairwise doc-doc edges
        doc_list = list(neighbor_weights.keys())
        for i in range(len(doc_list)):
            for j in range(i + 1, len(doc_list)):
                d1, d2 = doc_list[i], doc_list[j]
                w = min(neighbor_weights[d1], neighbor_weights[d2])
                doc_neighbors[d1][d2] += w
                doc_neighbors[d2][d1] += w

    if n_skipped:
        logger.info(
            "Projection: skipped %d high-degree entities (>%d docs)",
            n_skipped, max_entity_degree,
        )

    # Build projected graph
    doc_names = [bipartite.vs[v]["name"] for v in doc_vertices]
    doc_idx_map = {v: i for i, v in enumerate(doc_vertices)}

    edges = []
    weights = []
    seen = set()
    for d1, neighbors in doc_neighbors.items():
        for d2, w in neighbors.items():
            pair = (min(d1, d2), max(d1, d2))
            if pair not in seen:
                seen.add(pair)
                edges.append((doc_idx_map[d1], doc_idx_map[d2]))
                weights.append(w)

    g = ig.Graph(n=n_docs, edges=edges, directed=False)
    g.vs["name"] = doc_names
    g.es["weight"] = weights

    logger.info(
        "Projected doc graph: %d docs, %d edges, avg degree %.1f",
        n_docs, len(edges),
        np.mean(g.degree()) if n_docs > 0 else 0,
    )
    return g


def run_leiden_multiresolution(
    doc_graph: ig.Graph,
    resolutions: list[float] | None = None,
) -> dict[float, list[int]]:
    """Run Leiden community detection at multiple resolutions.

    Returns:
        dict mapping resolution -> list of community assignments (one per vertex)
    """
    if resolutions is None:
        resolutions = LEIDEN_RESOLUTIONS

    results = {}
    for res in resolutions:
        partition = leidenalg.find_partition(
            doc_graph,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=res,
            seed=42,
        )
        results[res] = partition.membership
        n_communities = max(partition.membership) + 1 if partition.membership else 0
        modularity = partition.modularity
        logger.info(
            "Leiden res=%.1f: %d communities, modularity=%.4f",
            res, n_communities, modularity,
        )

    return results


@dataclass
class CommunityFeatures:
    """Community-based features for a document pair."""
    same_community: dict[float, bool] = field(default_factory=dict)
    community_jaccard: float = 0.0  # Jaccard over all multi-res memberships
    shared_statutes: int = 0
    shared_judges: int = 0
    same_domain: bool = False
    same_outcome: bool = False
    entity_overlap_score: float = 0.0


def compute_pair_features(
    query_id: str,
    candidate_id: str,
    records: dict[str, EntityRecord],
    communities: dict[float, list[int]],
    doc_name_to_idx: dict[str, int],
) -> CommunityFeatures:
    """Compute GraphRAG Lite features for a (query, candidate) pair."""
    feats = CommunityFeatures()

    q_rec = records.get(query_id)
    c_rec = records.get(candidate_id)

    if q_rec and c_rec:
        # Shared entities
        q_statutes = set(q_rec.statutes)
        c_statutes = set(c_rec.statutes)
        feats.shared_statutes = len(q_statutes & c_statutes)

        q_judges = set(q_rec.judges)
        c_judges = set(c_rec.judges)
        feats.shared_judges = len(q_judges & c_judges)

        feats.same_domain = q_rec.domain == c_rec.domain
        feats.same_outcome = (
            q_rec.outcome is not None
            and c_rec.outcome is not None
            and q_rec.outcome == c_rec.outcome
        )

        # Entity overlap score (weighted Jaccard)
        weights = BIPARTITE_WEIGHTS
        q_entities = (
            {f"s:{s}" for s in q_rec.statutes}
            | {f"j:{j}" for j in q_rec.judges}
            | {f"d:{q_rec.domain}"}
        )
        c_entities = (
            {f"s:{s}" for s in c_rec.statutes}
            | {f"j:{j}" for j in c_rec.judges}
            | {f"d:{c_rec.domain}"}
        )
        intersection = len(q_entities & c_entities)
        union = len(q_entities | c_entities)
        feats.entity_overlap_score = intersection / union if union > 0 else 0.0

    # Community features
    q_idx = doc_name_to_idx.get(query_id)
    c_idx = doc_name_to_idx.get(candidate_id)

    if q_idx is not None and c_idx is not None:
        same_count = 0
        for res, membership in communities.items():
            if q_idx < len(membership) and c_idx < len(membership):
                same = membership[q_idx] == membership[c_idx]
                feats.same_community[res] = same
                if same:
                    same_count += 1

        # Jaccard over community assignments
        n_resolutions = len(communities)
        feats.community_jaccard = same_count / n_resolutions if n_resolutions > 0 else 0.0

    return feats


def features_to_dict(feats: CommunityFeatures) -> dict[str, float]:
    """Flatten CommunityFeatures to a dict for the meta-learner."""
    d: dict[str, float] = {}
    for res, same in feats.same_community.items():
        d[f"same_community_{res:.1f}"] = float(same)
    d["community_jaccard"] = feats.community_jaccard
    d["shared_statutes"] = float(feats.shared_statutes)
    d["shared_judges"] = float(feats.shared_judges)
    d["same_domain"] = float(feats.same_domain)
    d["same_outcome"] = float(feats.same_outcome)
    d["entity_overlap_score"] = feats.entity_overlap_score
    return d


class GraphRAGLite:
    """Full GraphRAG Lite pipeline: extract → graph → communities → features."""

    def __init__(self):
        self.records: dict[str, EntityRecord] = {}
        self.bipartite: ig.Graph | None = None
        self.doc_graph: ig.Graph | None = None
        self.communities: dict[float, list[int]] = {}
        self.doc_name_to_idx: dict[str, int] = {}
        self._ppr_cache: dict[str, dict[str, float]] = {}  # doc_id -> {doc_id: ppr_score}

    def fit(self, corpus: dict[str, str], compute_ppr: bool = False) -> None:
        """Build the full GraphRAG Lite index from a corpus.

        Args:
            corpus: dict mapping doc_id -> preprocessed text
            compute_ppr: if True, precompute PPR vectors for all documents
        """
        logger.info("Building GraphRAG Lite index...")
        self.records = build_entity_records(corpus)
        self.bipartite = build_bipartite_graph(self.records)
        self.doc_graph = project_to_doc_graph(self.bipartite)
        self.communities = run_leiden_multiresolution(self.doc_graph)
        self.doc_name_to_idx = {
            name: i for i, name in enumerate(self.doc_graph.vs["name"])
        }

        if compute_ppr:
            self._precompute_ppr()

        logger.info("GraphRAG Lite index complete.")

    def _precompute_ppr(self, damping: float = 0.85) -> None:
        """Precompute Personalized PageRank from each document node.

        For each document, runs PPR with teleport set to that document's
        entity neighbors in the bipartite graph. The resulting scores
        capture how much "relevance mass" flows to other documents through
        shared legal entities (statutes, judges, domains).

        Uses the projected doc-doc graph (not bipartite) for efficiency.
        """
        logger.info("Precomputing Personalized PageRank (damping=%.2f)...", damping)
        n_docs = self.doc_graph.vcount()
        doc_names = self.doc_graph.vs["name"]

        for i, doc_id in enumerate(doc_names):
            # Personalized restart vector: uniform over this node
            reset = [0.0] * n_docs
            reset[i] = 1.0

            ppr_scores = self.doc_graph.personalized_pagerank(
                vertices=None,
                directed=False,
                damping=damping,
                reset=reset,
                weights="weight",
            )
            # Store only non-zero scores (sparse — most docs get ~0)
            self._ppr_cache[doc_id] = {
                doc_names[j]: score
                for j, score in enumerate(ppr_scores)
                if score > 1e-8 and j != i
            }

            if (i + 1) % 500 == 0:
                logger.info("  PPR: %d/%d documents", i + 1, n_docs)

        logger.info("PPR complete: %d documents, avg %.0f non-zero neighbors",
                     n_docs,
                     np.mean([len(v) for v in self._ppr_cache.values()]) if self._ppr_cache else 0)

    def get_ppr_features(
        self, query_id: str, candidate_id: str,
    ) -> dict[str, float]:
        """Get PPR features for a (query, candidate) pair.

        Returns:
            ppr_score: PPR score from query to candidate
            ppr_rank: rank of candidate in query's PPR scores (lower = closer)
        """
        if not self._ppr_cache:
            return {"ppr_score": 0.0, "ppr_rank": 999.0}

        q_scores = self._ppr_cache.get(query_id, {})
        ppr_score = q_scores.get(candidate_id, 0.0)

        # Compute rank (1-based, lower is better)
        if q_scores and ppr_score > 0:
            rank = sum(1 for s in q_scores.values() if s > ppr_score) + 1
            ppr_rank = float(rank)
        else:
            ppr_rank = 999.0

        return {"ppr_score": ppr_score, "ppr_rank": ppr_rank}

    def get_pair_features(
        self, query_id: str, candidate_id: str,
    ) -> dict[str, float]:
        """Get community features for a (query, candidate) pair."""
        feats = compute_pair_features(
            query_id, candidate_id,
            self.records, self.communities, self.doc_name_to_idx,
        )
        d = features_to_dict(feats)

        # Add PPR features if computed
        if self._ppr_cache:
            d.update(self.get_ppr_features(query_id, candidate_id))

        return d

    def get_batch_features(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[dict[str, float]]:
        """Get features for a batch of (query, candidate) pairs."""
        return [self.get_pair_features(q, c) for q, c in pairs]
