"""Shared test fixtures for baseline tests."""
import pytest


@pytest.fixture
def mini_corpus():
    return {
        "q1.txt": "immigration refugee protection act judicial review",
        "q2.txt": "patent infringement claim intellectual property",
        "d1.txt": "immigration law refugee status determination IRPA",
        "d2.txt": "patent claim prior art obviousness test",
        "d3.txt": "immigration appeal refugee claim board decision",
        "d4.txt": "income tax assessment deduction revenue",
    }


@pytest.fixture
def mini_labels():
    return {
        "q1.txt": ["d1.txt", "d3.txt"],
        "q2.txt": ["d2.txt"],
    }
