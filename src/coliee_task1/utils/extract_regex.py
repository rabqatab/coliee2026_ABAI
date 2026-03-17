"""Regex-based entity extraction from legal case documents."""
import re


def extract_statutes(text: str) -> list[str]:
    """Extract statute references from case text.

    Captures full Act names with citations and section references.
    """
    statutes = []

    # Full statute citations: "Act Name, S.C./R.S.C. YYYY, c. X"
    full_pattern = r"([\w\s]+Act),?\s*(S\.C\.|R\.S\.C\.)\s*\d{4},?\s*c\.\s*[\w.\-]+"
    for match in re.finditer(full_pattern, text):
        statutes.append(match.group(0).strip())

    # Section references: "section/subsection/paragraph N(N) of the ACT"
    section_pattern = (
        r"(?:section|subsection|paragraph|s\.)\s*(\d+(?:\(\d+\))?(?:\.\d+)?)"
        r"\s+of\s+(?:the\s+)?([\w\s]*(?:Act|Regulations?|IRPA|PRRA))"
    )
    for match in re.finditer(section_pattern, text, re.IGNORECASE):
        statutes.append(f"{match.group(2).strip()} s. {match.group(1)}")

    # Abbreviated references: "IRPA", "PRRA", "CBSA"
    abbrev_pattern = r"\b(IRPA|PRRA|CBSA|FCTD|FCA|SCC)\b"
    for match in re.finditer(abbrev_pattern, text):
        statutes.append(match.group(1))

    return list(set(statutes))


def extract_judges(text: str) -> list[str]:
    """Extract judge names from case text."""
    judges = []

    # "Name J." or "Name, J." or "Name J.A."
    pattern1 = r"(\b[A-Z][a-z]+),?\s*J\.(?:A\.)?"
    for match in re.finditer(pattern1, text):
        judges.append(f"{match.group(1)} J.")

    # "The Honourable Mr./Madam Justice Name"
    pattern2 = r"(?:The\s+)?Honou?rable\s+(?:Mr\.|Madam)\s+Justice\s+(\w+)"
    for match in re.finditer(pattern2, text, re.IGNORECASE):
        judges.append(f"{match.group(1)} J.")

    # "Before: Name J."
    pattern3 = r"Before:\s*(\w+),?\s*J\.(?:A\.)?"
    for match in re.finditer(pattern3, text):
        judges.append(f"{match.group(1)} J.")

    return list(set(judges))


def extract_outcome(text: str) -> str | None:
    """Extract case outcome from text.

    Returns normalized string like 'application dismissed' or None.
    """
    pattern = r"(?:the\s+)?(application|appeal|motion)\s+(?:is\s+|be\s+)?(dismissed|allowed|granted)"
    # Search in the last 2000 characters (outcome is typically at the end)
    search_text = text[-2000:] if len(text) > 2000 else text
    match = re.search(pattern, search_text, re.IGNORECASE)
    if match:
        return f"{match.group(1).lower()} {match.group(2).lower()}"
    return None


def extract_paragraph_markers(text: str) -> list[int]:
    """Extract paragraph numbers from [N] markers."""
    return [int(m) for m in re.findall(r"\[(\d+)\]", text)]
