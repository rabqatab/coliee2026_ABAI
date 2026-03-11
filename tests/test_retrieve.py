"""Tests for multi-signal retrieval and fusion."""
import pytest
from graphrag.retrieve import reciprocal_rank_fusion, weighted_entity_score


class TestRRF:
    def test_basic_fusion(self):
        rankings = {
            "signal_1": [("a", 10), ("b", 8), ("c", 5)],
            "signal_2": [("b", 10), ("c", 8), ("a", 5)],
        }
        fused = reciprocal_rank_fusion(rankings, k=60)
        doc_ids = [d for d, _ in fused]
        assert set(doc_ids) == {"a", "b", "c"}
        # b: 1/(60+2) + 1/(60+1) = highest
        assert fused[0][0] == "b"

    def test_empty_rankings(self):
        result = reciprocal_rank_fusion({}, k=60)
        assert result == []


class TestWeightedEntityScore:
    def test_shared_entities(self):
        query_entities = {
            "statutes": {"IRPA", "Charter"},
            "concepts": {"standard of review"},
            "tests": {"Dunsmuir test"},
            "domain": "immigration",
            "judge": "Mosley J.",
        }
        candidate_entities = {
            "statutes": {"IRPA"},
            "concepts": {"standard of review", "procedural fairness"},
            "tests": {"Dunsmuir test"},
            "domain": "immigration",
            "judge": "Russell J.",
        }
        score = weighted_entity_score(query_entities, candidate_entities)
        assert 0 < score <= 1.0

    def test_no_overlap(self):
        query = {
            "statutes": {"IRPA"},
            "concepts": {"fairness"},
            "tests": set(),
            "domain": "immigration",
            "judge": "Mosley J.",
        }
        candidate = {
            "statutes": {"Patent Act"},
            "concepts": {"novelty"},
            "tests": set(),
            "domain": "IP",
            "judge": "Russell J.",
        }
        score = weighted_entity_score(query, candidate)
        assert score == 0.0
