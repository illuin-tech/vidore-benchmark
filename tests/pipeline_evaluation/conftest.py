"""
Shared fixtures for pipeline evaluation tests.
"""

import pytest
from PIL import Image


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing."""
    return Image.new("RGB", (100, 100), color="red")


@pytest.fixture
def sample_corpus_images():
    """Create a list of sample corpus images."""
    return [
        Image.new("RGB", (100, 100), color="red"),
        Image.new("RGB", (100, 100), color="blue"),
        Image.new("RGB", (100, 100), color="green"),
    ]


@pytest.fixture
def sample_corpus_texts():
    """Create a list of sample corpus texts (markdown)."""
    return [
        "# Document 1\n\nThis is the first document.",
        "# Document 2\n\n## Section A\n\nContent here.",
        "# Document 3\n\n| Col1 | Col2 |\n|------|------|\n| A    | B    |",
    ]


@pytest.fixture
def sample_query_ids():
    """Create sample query IDs."""
    return ["q1", "q2", "q3"]


@pytest.fixture
def sample_queries():
    """Create sample query texts."""
    return ["What is the revenue?", "Show me the chart", "Find the table"]


@pytest.fixture
def sample_corpus_ids():
    """Create sample corpus IDs."""
    return ["doc1", "doc2", "doc3"]


@pytest.fixture
def sample_qrels():
    """Create sample qrels in pytrec_eval format."""
    return {
        "q1": {"doc1": 1},
        "q2": {"doc2": 1},
        "q3": {"doc3": 1},
    }


@pytest.fixture
def sample_query_languages():
    """Create sample query language mapping."""
    return {
        "q1": "english",
        "q2": "english",
        "q3": "french",
    }
