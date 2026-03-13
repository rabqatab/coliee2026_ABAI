"""Tests for UMNLP 2024 baseline."""


def test_umnlp_interface():
    from baselines.umnlp.model import UMNLPBaseline
    model = UMNLPBaseline()
    assert model.name() == "UMNLP 2024 (Propositions+NN)"


def test_proposition_extraction():
    from baselines.umnlp.propositions import PropositionExtractor
    extractor = PropositionExtractor(window_size=10)
    text = "This is before <FRAGMENT_SUPPRESSED> and this is after the marker"
    windows = extractor.extract_windows(text)
    assert len(windows) >= 1
    # Window should contain words from both sides of the marker
    assert "before" in windows[0]
    assert "after" in windows[0]


def test_proposition_extraction_no_markers():
    from baselines.umnlp.propositions import PropositionExtractor
    extractor = PropositionExtractor(window_size=10)
    text = "A plain document with no markers at all."
    windows = extractor.extract_windows(text)
    # Should fall back to first 500 chars
    assert len(windows) == 1
    assert windows[0] == text


def test_proposition_extraction_multiple_markers():
    from baselines.umnlp.propositions import PropositionExtractor
    extractor = PropositionExtractor(window_size=5)
    text = "aaa bbb ccc <FRAGMENT_SUPPRESSED> ddd eee fff ggg hhh <FRAGMENT_SUPPRESSED> iii jjj"
    windows = extractor.extract_windows(text)
    assert len(windows) == 2


def test_judge_extraction():
    from baselines.umnlp.judge_match import extract_judges
    text = "Justice Smith presided. Mr. Justice Brown concurred."
    judges = extract_judges(text)
    assert len(judges) >= 1
    assert "smith" in judges or "brown" in judges


def test_judge_matcher():
    from baselines.umnlp.judge_match import JudgeMatcher
    corpus = {
        "q1.txt": "Justice Smith decided the matter.",
        "d1.txt": "Justice Smith and Justice Brown reviewed.",
        "d2.txt": "Justice Jones presided over this case.",
    }
    matcher = JudgeMatcher()
    matcher.build_index(corpus)

    binary, idf = matcher.match("q1.txt", "d1.txt")
    assert binary == 1.0
    assert idf > 0.0

    binary2, idf2 = matcher.match("q1.txt", "d2.txt")
    assert binary2 == 0.0
    assert idf2 == 0.0


def test_quotation_overlap():
    from baselines.umnlp.features import quotation_overlap
    text1 = "the quick brown fox jumps over the lazy dog today"
    text2 = "the quick brown fox jumps over something else entirely now"
    score = quotation_overlap(text1, text2, n=5)
    assert score > 0.0

    # No overlap
    text3 = "completely different words that have nothing in common here"
    score2 = quotation_overlap(text1, text3, n=5)
    assert score2 == 0.0


def test_feature_extraction(mini_corpus):
    import numpy as np
    from baselines.umnlp.features import extract_umnlp_features
    from baselines.umnlp.judge_match import JudgeMatcher

    judge_matcher = JudgeMatcher()
    judge_matcher.build_index(mini_corpus)

    # Fake proposition embeddings
    rng = np.random.RandomState(42)
    prop_embeddings = {}
    for did in mini_corpus:
        emb = rng.randn(384).astype(np.float32)
        emb /= np.linalg.norm(emb)
        prop_embeddings[did] = emb

    bm25_candidates = {
        "q1.txt": [("d1.txt", 10.0), ("d3.txt", 8.0), ("d2.txt", 3.0)],
    }

    feats = extract_umnlp_features(
        "q1.txt", "d1.txt", mini_corpus, bm25_candidates,
        prop_embeddings, judge_matcher,
    )
    expected_keys = {
        "proposition_sim", "judge_match", "judge_rarity_score",
        "quotation_overlap", "para_max_sim", "para_mean_sim",
        "bm25_score", "doc_len_ratio",
    }
    assert set(feats.keys()) == expected_keys
    assert feats["bm25_score"] == 10.0
    assert feats["doc_len_ratio"] > 0.0
