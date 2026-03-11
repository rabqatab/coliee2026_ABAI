"""Tests for regex-based entity extraction."""
import pytest
from graphrag.extract_regex import extract_statutes, extract_judges, extract_outcome


class TestExtractStatutes:
    def test_irpa_full_name(self):
        text = "Immigration and Refugee Protection Act, S.C. 2001, c. 27"
        result = extract_statutes(text)
        assert any("Immigration and Refugee Protection Act" in s for s in result)

    def test_section_reference(self):
        text = "pursuant to subsection 72(1) of the IRPA"
        result = extract_statutes(text)
        assert any("72(1)" in s for s in result)

    def test_multiple_statutes(self):
        text = """
        The Federal Courts Act, R.S.C. 1985, c. F-7
        and the Immigration and Refugee Protection Act, S.C. 2001, c. 27
        """
        result = extract_statutes(text)
        assert len(result) >= 2

    def test_no_statutes(self):
        text = "This is a simple text with no statute references."
        result = extract_statutes(text)
        assert len(result) == 0


class TestExtractJudges:
    def test_simple_judge(self):
        text = "Mosley J."
        result = extract_judges(text)
        assert "Mosley J." in result

    def test_with_comma(self):
        text = "Before: Mosley, J."
        result = extract_judges(text)
        assert len(result) >= 1

    def test_justice_full_title(self):
        text = "The Honourable Mr. Justice Mosley"
        result = extract_judges(text)
        assert len(result) >= 1


class TestExtractOutcome:
    def test_dismissed(self):
        text = "The application is dismissed."
        result = extract_outcome(text)
        assert result == "application dismissed"

    def test_allowed(self):
        text = "The appeal is allowed."
        result = extract_outcome(text)
        assert result == "appeal allowed"

    def test_no_outcome(self):
        text = "The parties submitted their arguments."
        result = extract_outcome(text)
        assert result is None
