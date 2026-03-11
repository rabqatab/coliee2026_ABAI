"""Knowledge graph construction from extracted entities."""
import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


def load_extractions(extractions_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all extraction JSON files from a directory."""
    extractions = {}
    for path in sorted(extractions_dir.glob("*.json")):
        data = json.loads(path.read_text())
        extractions[data["doc_id"]] = data
    logger.info("Loaded %d extractions from %s", len(extractions), extractions_dir)
    return extractions


def build_knowledge_graph(
    extractions_dir_or_dict: Path | dict[str, dict[str, Any]],
) -> nx.Graph:
    """Build a knowledge graph from entity extractions.

    Node types: Case, Statute, LegalConcept, LegalTest, Judge, Outcome, Domain
    Edge types: APPLIES, INVOKES_CONCEPT, APPLIES_TEST, DECIDED_BY, HAS_OUTCOME, IN_DOMAIN
    """
    if isinstance(extractions_dir_or_dict, Path):
        extractions = load_extractions(extractions_dir_or_dict)
    else:
        extractions = extractions_dir_or_dict

    G = nx.Graph()

    for doc_id, data in extractions.items():
        case_id = f"case:{doc_id}"

        # Add Case node
        G.add_node(
            case_id,
            type="Case",
            doc_id=doc_id,
            case_type=data.get("case_type", "other"),
            word_count=data.get("word_count", 0),
        )

        # Statutes
        for statute in data.get("statutes", []):
            statute_id = f"statute:{statute}"
            G.add_node(statute_id, type="Statute", name=statute)
            G.add_edge(case_id, statute_id, relation="APPLIES")

        # Legal concepts
        for concept in data.get("concepts", []):
            concept_id = f"concept:{concept}"
            G.add_node(concept_id, type="LegalConcept", name=concept)
            G.add_edge(case_id, concept_id, relation="INVOKES_CONCEPT")

        # Legal tests
        for test in data.get("tests", []):
            test_id = f"test:{test}"
            G.add_node(test_id, type="LegalTest", name=test)
            G.add_edge(case_id, test_id, relation="APPLIES_TEST")

        # Judges
        for judge in data.get("judges", []):
            judge_id = f"judge:{judge}"
            G.add_node(judge_id, type="Judge", name=judge)
            G.add_edge(case_id, judge_id, relation="DECIDED_BY")

        # Outcome
        outcome = data.get("outcome")
        if outcome:
            outcome_id = f"outcome:{outcome}"
            G.add_node(outcome_id, type="Outcome", name=outcome)
            G.add_edge(case_id, outcome_id, relation="HAS_OUTCOME")

        # Domain
        domain = data.get("domain", "other")
        domain_id = f"domain:{domain}"
        G.add_node(domain_id, type="Domain", name=domain)
        G.add_edge(case_id, domain_id, relation="IN_DOMAIN")

    # Log graph stats
    node_types = {}
    for _, d in G.nodes(data=True):
        t = d.get("type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1

    logger.info(
        "Graph built: %d nodes (%s), %d edges",
        G.number_of_nodes(),
        ", ".join(f"{k}:{v}" for k, v in sorted(node_types.items())),
        G.number_of_edges(),
    )
    return G


def save_graph(G: nx.Graph, output_dir: Path) -> None:
    """Save graph as GraphML + JSON metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, output_dir / "knowledge_graph.graphml")

    # Save as adjacency JSON for easier loading
    data = nx.node_link_data(G)
    (output_dir / "knowledge_graph.json").write_text(
        json.dumps(data, indent=2, default=str)
    )
    logger.info("Graph saved to %s", output_dir)


def load_graph(graph_dir: Path) -> nx.Graph:
    """Load graph from JSON."""
    data = json.loads((graph_dir / "knowledge_graph.json").read_text())
    return nx.node_link_graph(data)
