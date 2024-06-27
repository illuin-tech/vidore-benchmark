from typing import List, Tuple

import pytest
from PIL import Image

DUMMY_TEXT = """\
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore \
et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut \
aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse \
cillum dolore eu fugiat nulla pariatur.
"""


@pytest.fixture
def query_document_pairs_fixture() -> List[Tuple[str, str]]:
    return [
        (
            "What are some common outcome areas targeted by TAII for different age groups?",
            "./tests/retrievers/data/ai_sample.jpeg",
        ),
        (
            "What types of accounts or products allow investors to defer paying taxes?",
            "./tests/retrievers/data/energy_sample.jpeg",
        ),
        (
            "What is the chemical formula for the ferroelectric material Lead Zirconium Titanate (PZT)?",
            "./tests/retrievers/data/gov_sample.jpeg",
        ),
        (
            "What are some mandates for the EPA under the Pollution Prevention Act?",
            "./tests/retrievers/data/healthcare_sample.jpeg",
        ),
        (
            "Quelle partie de la production pétrolière du Kazakhstan provient de champs en mer ?",
            "./tests/retrievers/data/shift_sample.jpeg",
        ),
    ]


@pytest.fixture
def queries_fixtures(query_document_pairs_fixture) -> List[str]:
    return [query for query, _ in query_document_pairs_fixture]


@pytest.fixture
def document_filepaths_fixture(query_document_pairs_fixture) -> List[Image.Image]:
    return [doc for _, doc in query_document_pairs_fixture]


@pytest.fixture
def document_images_fixture(document_filepaths_fixture) -> List[Image.Image]:
    return [Image.open(doc) for doc in document_filepaths_fixture]


@pytest.fixture
def document_ocr_text_fixture(document_filepaths_fixture) -> List[str]:
    return [DUMMY_TEXT for _ in document_filepaths_fixture]
