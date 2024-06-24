from typing import List, Tuple

import pytest
from PIL import Image


@pytest.fixture
def query_document_pairs_fixture() -> List[Tuple[str, Image.Image]]:
    return [
        (
            "What are some common outcome areas targeted by TAII for different age groups?",
            Image.open("./tests/retrievers/data/ai_sample.jpeg"),
        ),
        (
            "What types of accounts or products allow investors to defer paying taxes?",
            Image.open("./tests/retrievers/data/energy_sample.jpeg"),
        ),
        (
            "What is the chemical formula for the ferroelectric material Lead Zirconium Titanate (PZT)?",
            Image.open("./tests/retrievers/data/gov_sample.jpeg"),
        ),
        (
            "What are some mandates for the EPA under the Pollution Prevention Act?",
            Image.open("./tests/retrievers/data/healthcare_sample.jpeg"),
        ),
        (
            "Quelle partie de la production pétrolière du Kazakhstan provient de champs en mer ?",
            Image.open("./tests/retrievers/data/shift_sample.jpeg"),
        ),
    ]


@pytest.fixture
def queries_fixtures(query_document_pairs_fixture) -> List[str]:
    return [query for query, _ in query_document_pairs_fixture]


@pytest.fixture
def documents_fixture(query_document_pairs_fixture) -> List[Image.Image]:
    return [doc for _, doc in query_document_pairs_fixture]
