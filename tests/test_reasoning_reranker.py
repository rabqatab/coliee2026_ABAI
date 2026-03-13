"""Tests for reasoning reranker."""


def test_build_reasoning_prompt():
    """Verify prompt construction includes query and candidate text."""
    from graphrag.reasoning_reranker import build_reasoning_prompt

    prompt = build_reasoning_prompt(
        query_text="The applicant seeks judicial review...",
        candidate_text="In Smith v. Canada, the court held...",
    )
    assert "applicant seeks judicial review" in prompt
    assert "Smith v. Canada" in prompt
    assert "relevant" in prompt.lower() or "cite" in prompt.lower()


def test_parse_relevance_score():
    """Verify score extraction from model output."""
    from graphrag.reasoning_reranker import parse_relevance_score

    output1 = "<think>Both cases deal with immigration...</think>\nRelevant: Yes\nScore: 0.85"
    score1 = parse_relevance_score(output1)
    assert 0.8 <= score1 <= 0.9

    output2 = "<think>Different legal areas...</think>\nRelevant: No\nScore: 0.15"
    score2 = parse_relevance_score(output2)
    assert 0.1 <= score2 <= 0.2

    # Fallback for unparseable output
    score3 = parse_relevance_score("gibberish")
    assert score3 == 0.5  # neutral fallback


def test_parse_relevance_score_yes_no_fallback():
    """When no explicit score but Yes/No present, use fallback scores."""
    from graphrag.reasoning_reranker import parse_relevance_score

    assert parse_relevance_score("Relevant: Yes\nNo explicit score") == 0.75
    assert parse_relevance_score("Relevant: No\nNo explicit score") == 0.25
