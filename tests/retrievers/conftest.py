from typing import List

import pytest
from PIL import Image

DUMMY_TEXT = "Lorem ipsum"


@pytest.fixture
def queries_fixture() -> List[str]:
    queries = [
        "What is the organizational structure for our R&D department?",
        "Can you provide a breakdown of last year’s financial performance?",
    ]
    return queries


@pytest.fixture
def image_passage_fixture(queries_fixture) -> List[Image.Image]:
    images = [Image.new("RGB", (16, 16), color="black") for _ in queries_fixture]
    return images


@pytest.fixture
def text_passage_fixture(queries_fixture) -> List[str]:
    return [DUMMY_TEXT for _ in queries_fixture]
