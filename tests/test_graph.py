"""Tests for knowledge graph construction."""
import json
from pathlib import Path

import pytest
import networkx as nx

from graphrag.graph import build_knowledge_graph, save_graph, load_graph


@pytest.fixture
def sample_extractions(tmp_path):
    """Create sample extraction files."""
    docs = {
        "000001": {
            "doc_id": "000001",
            "statutes": ["IRPA", "IRPA s. 72(1)"],
            "judges": ["Mosley J."],
            "outcome": "application dismissed",
            "concepts": ["standard of review", "procedural fairness"],
            "tests": ["Dunsmuir test"],
            "holdings": ["Officer's decision was reasonable"],
            "case_type": "judicial_review",
            "domain": "immigration",
        },
        "000002": {
            "doc_id": "000002",
            "statutes": ["IRPA", "IRPA s. 96"],
            "judges": ["Russell J."],
            "outcome": "application allowed",
            "concepts": ["standard of review", "credibility"],
            "tests": ["Dunsmuir test"],
            "holdings": ["Board failed to consider evidence"],
            "case_type": "judicial_review",
            "domain": "immigration",
        },
    }
    ext_dir = tmp_path / "extractions"
    ext_dir.mkdir()
    for doc_id, data in docs.items():
        (ext_dir / f"{doc_id}.json").write_text(json.dumps(data))
    return ext_dir


class TestBuildGraph:
    def test_creates_case_nodes(self, sample_extractions):
        G = build_knowledge_graph(sample_extractions)
        case_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Case"]
        assert len(case_nodes) == 2

    def test_creates_statute_nodes(self, sample_extractions):
        G = build_knowledge_graph(sample_extractions)
        statute_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Statute"]
        assert len(statute_nodes) >= 2

    def test_creates_concept_nodes(self, sample_extractions):
        G = build_knowledge_graph(sample_extractions)
        concept_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "LegalConcept"]
        assert len(concept_nodes) >= 2

    def test_shared_entities_create_paths(self, sample_extractions):
        G = build_knowledge_graph(sample_extractions)
        # Both cases share IRPA and "standard of review"
        assert nx.has_path(G, "case:000001", "case:000002")


class TestSaveLoad:
    def test_roundtrip(self, sample_extractions, tmp_path):
        G = build_knowledge_graph(sample_extractions)
        graph_path = tmp_path / "graph"
        save_graph(G, graph_path)
        G2 = load_graph(graph_path)
        assert set(G.nodes()) == set(G2.nodes())
        assert len(G.edges()) == len(G2.edges())
