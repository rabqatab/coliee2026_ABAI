"""Tests for vanilla BM25 baseline."""
from baselines.bm25.model import VanillaBM25


def test_vanilla_bm25_interface():
    model = VanillaBM25()
    assert model.name() == "BM25 (vanilla)"


def test_vanilla_bm25_ranking(mini_corpus, mini_labels):
    model = VanillaBM25()
    model.train(mini_corpus, ["q1.txt"], mini_labels)
    results = model.predict("q1.txt", mini_corpus)
    doc_ids = [did for did, _ in results]
    assert "q1.txt" not in doc_ids
    assert len(results) > 0
    # Immigration docs should rank higher than patent/tax docs
    assert results[0][0] in ("d1.txt", "d3.txt")
