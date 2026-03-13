"""Build a case-case + case-statute bipartite graph for CaseLink."""
import re
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def extract_statutes_simple(text: str) -> list[str]:
    """Extract statute references from case text.

    Captures common Canadian legal citation patterns:
    - "section X of the Y Act" or "s. X"
    - "paragraph X(a)" style references
    - Named Acts (Immigration, Tax, Criminal, Patent, etc.)
    """
    statutes = set()
    patterns = [
        r'(?:section|s\.)\s+(\d+(?:\.\d+)?)',
        r'paragraph\s+(\d+\([a-z]\))',
        r'(?:Immigration|Tax|Criminal|Patent)\s+(?:and\s+\w+\s+)?(?:Protection\s+)?Act',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            statutes.add(match.group(0).lower().strip())
    return list(statutes)


class CaseGraph:
    """Heterogeneous graph with case nodes, statute nodes, and two edge types."""

    def __init__(self):
        self.case_ids: list[str] = []
        self.statute_ids: list[str] = []
        self.case_case_edges: list[tuple[int, int]] = []      # (src_idx, dst_idx)
        self.case_statute_edges: list[tuple[int, int]] = []    # (case_idx, statute_idx)
        self._case_to_idx: dict[str, int] = {}
        self._statute_to_idx: dict[str, int] = {}

    def build(
        self,
        corpus: dict[str, str],
        labels: dict[str, list[str]],
        train_queries: list[str],
    ) -> None:
        """Build heterogeneous graph from corpus and training labels.

        Case-case edges come from known citations (training labels).
        Case-statute edges come from regex-extracted statute references.
        """
        # Index all cases
        self.case_ids = sorted(corpus.keys())
        self._case_to_idx = {cid: i for i, cid in enumerate(self.case_ids)}

        # Case-Case edges from training labels (known citations)
        edge_set: set[tuple[int, int]] = set()
        for qid in train_queries:
            if qid not in self._case_to_idx:
                continue
            for did in labels.get(qid, []):
                if did in self._case_to_idx:
                    src = self._case_to_idx[qid]
                    dst = self._case_to_idx[did]
                    edge_set.add((src, dst))
                    edge_set.add((dst, src))  # undirected

        self.case_case_edges = sorted(edge_set)

        # Extract statutes for each case
        case_statutes: dict[str, list[str]] = {}
        for cid, text in corpus.items():
            case_statutes[cid] = extract_statutes_simple(text)

        # Build statute index (only statutes appearing in >= 2 cases)
        statute_counts: dict[str, int] = defaultdict(int)
        for stats in case_statutes.values():
            for s in set(stats):
                statute_counts[s] += 1

        freq_statutes = [s for s, c in statute_counts.items() if c >= 2]
        self.statute_ids = sorted(freq_statutes)
        self._statute_to_idx = {sid: i for i, sid in enumerate(self.statute_ids)}

        # Case-Statute edges
        cs_set: set[tuple[int, int]] = set()
        for cid, stats in case_statutes.items():
            if cid not in self._case_to_idx:
                continue
            for s in set(stats):
                if s in self._statute_to_idx:
                    cs_set.add((self._case_to_idx[cid], self._statute_to_idx[s]))
        self.case_statute_edges = sorted(cs_set)

        logger.info(
            "Graph: %d cases, %d statutes, %d case-case edges, %d case-statute edges",
            len(self.case_ids),
            len(self.statute_ids),
            len(self.case_case_edges),
            len(self.case_statute_edges),
        )

    def get_case_neighbors(self, case_idx: int) -> list[int]:
        """Get all case neighbors (via case-case edges)."""
        return [dst for src, dst in self.case_case_edges if src == case_idx]

    def get_statute_neighbors(self, case_idx: int) -> list[int]:
        """Get statute indices connected to a case."""
        return [sid for cid, sid in self.case_statute_edges if cid == case_idx]

    def get_cases_for_statute(self, statute_idx: int) -> list[int]:
        """Get case indices connected to a statute."""
        return [cid for cid, sid in self.case_statute_edges if sid == statute_idx]
