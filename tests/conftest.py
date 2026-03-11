"""Shared test fixtures."""
import json
import pytest
from pathlib import Path


@pytest.fixture
def sample_corpus(tmp_path):
    """Create a small sample corpus for testing."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    docs = {
        "000001.txt": "[1] This is a judicial review of an immigration decision. "
        "The Immigration and Refugee Protection Act, S.C. 2001, c. 27 applies. "
        "Mosley J. The standard of review is reasonableness per Dunsmuir. "
        "The application is dismissed.",
        "000002.txt": "[1] Another immigration case reviewing officer credibility finding. "
        "IRPA s. 96 applies. Russell J. Procedural fairness was breached. "
        "The application is allowed.",
        "000003.txt": "[1] Patent infringement case under the Patent Act. "
        "The test for claim construction follows Free World Trust. "
        "Hughes J. The motion is dismissed.",
    }
    for name, text in docs.items():
        (docs_dir / name).write_text(text)

    labels = {"000001.txt": ["000002.txt"]}
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels))

    return docs_dir, labels_path
