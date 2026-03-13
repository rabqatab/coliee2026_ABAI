"""Judge name extraction and matching.

Extracts judge names from case headers and computes IDF-weighted
overlap scores for (query, candidate) pairs.
"""
import re
import math
import logging
from collections import Counter

logger = logging.getLogger(__name__)


def extract_judges(text: str) -> list[str]:
    """Extract judge names from case text."""
    judges = []
    # Look for common patterns: "Justice X", "Judge X", "J.A.", "J."
    patterns = [
        r'(?:Justice|Judge|Madam Justice|Mr\. Justice)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'([A-Z][a-z]+)\s+(?:J\.A\.|J\.)',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text[:2000]):  # judges usually in header
            name = match.group(1).strip().lower()
            if len(name) > 2 and name not in ('the', 'this', 'that'):
                judges.append(name)
    return list(set(judges))


class JudgeMatcher:
    def __init__(self):
        self.judge_index: dict[str, set[str]] = {}  # doc_id -> set of judge names
        self.judge_idf: dict[str, float] = {}  # judge_name -> IDF score

    def build_index(self, corpus: dict[str, str]):
        """Extract judges for all docs and compute IDF."""
        for doc_id, text in corpus.items():
            self.judge_index[doc_id] = set(extract_judges(text))

        # Compute IDF for each judge
        n_docs = len(corpus)
        judge_counts: Counter = Counter()
        for judges in self.judge_index.values():
            for j in judges:
                judge_counts[j] += 1

        self.judge_idf = {
            j: math.log((n_docs + 1) / (count + 1))
            for j, count in judge_counts.items()
        }
        logger.info("Indexed %d unique judges across %d docs",
                     len(self.judge_idf), n_docs)

    def match(self, query_id: str, candidate_id: str) -> tuple[float, float]:
        """Return (binary_match, idf_weighted_score)."""
        q_judges = self.judge_index.get(query_id, set())
        d_judges = self.judge_index.get(candidate_id, set())

        shared = q_judges & d_judges
        binary = 1.0 if shared else 0.0
        idf_score = sum(self.judge_idf.get(j, 0.0) for j in shared)

        return binary, idf_score
