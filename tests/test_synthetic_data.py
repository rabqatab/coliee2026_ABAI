"""Tests for synthetic data generation."""


def test_extract_key_facts_prompt():
    """Verify prompt construction for key fact extraction."""
    from coliee_task1.utils.synthetic_data import build_extraction_prompt

    prompt = build_extraction_prompt(
        "The applicant is a citizen of Iran who sought refugee status..."
    )
    assert "key" in prompt.lower()
    assert "Iran" in prompt


def test_build_synthetic_pair():
    """Verify synthetic pair construction."""
    from coliee_task1.utils.synthetic_data import build_synthetic_pair

    pair = build_synthetic_pair(
        query_summary="Immigration case about refugee status from Iran",
        candidate_text="In this case, the applicant from Iran...",
        candidate_id="012345.txt",
    )
    assert pair["query"] is not None
    assert pair["candidate_id"] == "012345.txt"
    assert pair["label"] == 1
    assert pair["synthetic"] is True
