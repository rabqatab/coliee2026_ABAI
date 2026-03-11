"""Entity normalization and deduplication."""
import re
from typing import Any

# Canonical statute aliases
STATUTE_ALIASES: dict[str, str] = {
    "immigration and refugee protection act": "IRPA",
    "irpa": "IRPA",
    "the act (irpa)": "IRPA",
    "citizenship act": "Citizenship Act",
    "canada evidence act": "Canada Evidence Act",
    "criminal code": "Criminal Code",
    "charter of rights and freedoms": "Charter",
    "canadian charter of rights and freedoms": "Charter",
    "charter": "Charter",
    "federal courts act": "Federal Courts Act",
    "income tax act": "Income Tax Act",
    "canada labour code": "Canada Labour Code",
    "customs act": "Customs Act",
    "access to information act": "Access to Information Act",
    "privacy act": "Privacy Act",
    "patent act": "Patent Act",
    "copyright act": "Copyright Act",
    "trade-marks act": "Trade-marks Act",
    "competition act": "Competition Act",
    "bankruptcy and insolvency act": "Bankruptcy and Insolvency Act",
    "indian act": "Indian Act",
    "national defence act": "National Defence Act",
    "canada elections act": "Canada Elections Act",
}


def normalize_statute(raw: str) -> str:
    """Normalize a statute name to its canonical form.

    Handles full names, abbreviations, and citation suffixes.
    """
    # Strip citation suffix (e.g., ", S.C. 2001, c. 27")
    name = re.sub(r",?\s*(?:S\.C\.|R\.S\.C\.)\s*\d{4}.*$", "", raw).strip()

    # Check for section reference
    section_match = re.search(r"\bs\.\s*(\d+(?:\(\d+\))?(?:\.\d+)?)", name)
    section = section_match.group(0) if section_match else None

    # Remove section from name for lookup
    lookup = re.sub(r"\bs\.\s*\d+.*$", "", name).strip()
    lookup_lower = lookup.lower().rstrip(",. ")

    canonical = STATUTE_ALIASES.get(lookup_lower, lookup)

    if section:
        return f"{canonical} {section}"
    return canonical


def normalize_judge(raw: str) -> str:
    """Normalize a judge name to 'Lastname J.' format."""
    name = raw.strip()
    # Remove comma before J.
    name = re.sub(r",\s*J\.", " J.", name)
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def merge_regex_llm(
    regex_result: dict[str, Any],
    llm_result: dict[str, Any],
) -> dict[str, Any]:
    """Merge regex and LLM extraction results into a unified entity record.

    Returns a flat dict with normalized entities.
    """
    # Normalize regex statutes
    statutes = set()
    for s in regex_result.get("statutes", []):
        statutes.add(normalize_statute(s))

    # Add LLM statutes
    for s in llm_result.get("statutes_applied", []):
        if isinstance(s, dict):
            name = s.get("name", "")
            section = s.get("section", "")
            full = f"{name} {section}".strip() if section else name
        else:
            full = str(s)
        statutes.add(normalize_statute(full))

    # Normalize judges
    judges = set()
    for j in regex_result.get("judges", []):
        judges.add(normalize_judge(j))

    # Concepts and tests from LLM
    concepts = list(set(
        c.lower().strip() for c in llm_result.get("legal_concepts", [])
    ))
    tests = list(set(
        t.strip() for t in llm_result.get("legal_tests", [])
    ))

    return {
        "statutes": sorted(statutes),
        "judges": sorted(judges),
        "outcome": regex_result.get("outcome"),
        "concepts": concepts,
        "tests": tests,
        "holdings": llm_result.get("key_holdings", []),
        "case_type": llm_result.get("case_type", "other"),
        "domain": llm_result.get("legal_domain", "other"),
    }
