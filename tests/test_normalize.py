"""Tests for entity normalization."""
import pytest
from coliee_task1.utils.normalize import normalize_statute, normalize_judge, merge_regex_llm


class TestNormalizeStatute:
    def test_irpa_full(self):
        assert normalize_statute("Immigration and Refugee Protection Act") == "IRPA"

    def test_irpa_with_citation(self):
        result = normalize_statute(
            "Immigration and Refugee Protection Act, S.C. 2001, c. 27"
        )
        assert result == "IRPA"

    def test_already_abbreviated(self):
        assert normalize_statute("IRPA") == "IRPA"

    def test_federal_courts_act(self):
        result = normalize_statute("Federal Courts Act, R.S.C. 1985, c. F-7")
        assert result == "Federal Courts Act"

    def test_section_preserved(self):
        result = normalize_statute("IRPA s. 72(1)")
        assert "IRPA" in result
        assert "72(1)" in result


class TestNormalizeJudge:
    def test_simple(self):
        assert normalize_judge("Mosley J.") == "Mosley J."

    def test_with_comma(self):
        assert normalize_judge("Mosley, J.") == "Mosley J."

    def test_extra_whitespace(self):
        assert normalize_judge("  Mosley  J. ") == "Mosley J."


class TestMergeRegexLlm:
    def test_merges_statutes(self):
        regex = {
            "statutes": ["IRPA", "Federal Courts Act, R.S.C. 1985, c. F-7"],
            "judges": ["Mosley J."],
            "outcome": "application dismissed",
        }
        llm = {
            "legal_concepts": ["standard of review"],
            "legal_tests": ["Dunsmuir test"],
            "statutes_applied": [
                {"name": "IRPA", "section": "s. 72(1)", "context": "judicial review"}
            ],
            "key_holdings": ["Officer's decision unreasonable"],
            "case_type": "judicial_review",
            "legal_domain": "immigration",
        }
        merged = merge_regex_llm(regex, llm)
        assert "IRPA" in merged["statutes"]
        assert "Mosley J." in merged["judges"]
        assert "standard of review" in merged["concepts"]
        assert merged["domain"] == "immigration"
