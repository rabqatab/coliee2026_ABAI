"""Tests for CaseLink 2025 baseline."""


def test_caselink_interface():
    from baselines.caselink.model import CaseLinkBaseline

    model = CaseLinkBaseline()
    assert model.name() == "CaseLink 2025 (GNN)"


def test_graph_construction(mini_corpus, mini_labels):
    from baselines.caselink.graph import CaseGraph

    graph = CaseGraph()
    graph.build(mini_corpus, mini_labels, ["q1.txt", "q2.txt"])
    assert len(graph.case_ids) == len(mini_corpus)
    assert len(graph.case_case_edges) > 0


def test_statute_extraction():
    from baselines.caselink.graph import extract_statutes_simple

    text = "Under section 72 of the Immigration and Refugee Protection Act"
    statutes = extract_statutes_simple(text)
    assert len(statutes) >= 1


def test_statute_extraction_multiple():
    from baselines.caselink.graph import extract_statutes_simple

    text = (
        "Pursuant to section 18.1 and section 72 of the Federal Courts Act, "
        "and paragraph 3(a) of the Criminal Code"
    )
    statutes = extract_statutes_simple(text)
    # Should find at least the two section references
    assert len(statutes) >= 2


def test_graph_edges_undirected(mini_corpus, mini_labels):
    from baselines.caselink.graph import CaseGraph

    graph = CaseGraph()
    graph.build(mini_corpus, mini_labels, ["q1.txt", "q2.txt"])

    # If (A, B) exists as an edge, (B, A) should also exist
    edge_set = set(graph.case_case_edges)
    for src, dst in list(edge_set):
        assert (dst, src) in edge_set, f"Edge ({src},{dst}) exists but ({dst},{src}) missing"


def test_graph_case_neighbors(mini_corpus, mini_labels):
    from baselines.caselink.graph import CaseGraph

    graph = CaseGraph()
    graph.build(mini_corpus, mini_labels, ["q1.txt", "q2.txt"])

    # q1.txt -> d1.txt, d3.txt in labels, so d1 and d3 indices should be neighbors of q1
    q1_idx = graph._case_to_idx["q1.txt"]
    neighbors = graph.get_case_neighbors(q1_idx)
    assert graph._case_to_idx["d1.txt"] in neighbors
    assert graph._case_to_idx["d3.txt"] in neighbors
