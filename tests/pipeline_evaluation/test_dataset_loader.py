"""
Tests for pipeline evaluation dataset loader.
"""

from unittest.mock import patch

import pytest
from PIL import Image

from vidore_benchmark.pipeline_evaluation.dataset_loader import (
    get_available_datasets,
    load_vidore_dataset,
    print_dataset_info,
)


class TestGetAvailableDatasets:
    def test_returns_list(self):
        datasets = get_available_datasets()
        assert isinstance(datasets, list)

    def test_contains_expected_datasets(self):
        datasets = get_available_datasets()
        expected = [
            "vidore/vidore_v3_hr",
            "vidore/vidore_v3_finance_en",
            "vidore/vidore_v3_industrial",
            "vidore/vidore_v3_pharmaceuticals",
            "vidore/vidore_v3_computer_science",
            "vidore/vidore_v3_energy",
            "vidore/vidore_v3_physics",
            "vidore/vidore_v3_finance_fr",
        ]
        assert datasets == expected


class MockDataset:
    """Mock HuggingFace Dataset."""

    def __init__(self, data):
        self.data = data
        if data:
            self.column_names = list(data[0].keys())
        else:
            self.column_names = []

    def filter(self, func):
        return MockDataset([item for item in self.data if func(item)])

    def __getitem__(self, key):
        if isinstance(key, str):
            return [item[key] for item in self.data]
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)


class TestLoadVidoreDataset:
    @pytest.fixture
    def mock_queries_dataset(self):
        """Create mock queries dataset."""
        data = [
            {"query_id": 1, "query": "What is the revenue?", "language": "english"},
            {"query_id": 2, "query": "Find the chart", "language": "english"},
            {"query_id": 3, "query": "Quel est le chiffre?", "language": "french"},
        ]
        return MockDataset(data)

    @pytest.fixture
    def mock_corpus_dataset(self):
        """Create mock corpus dataset with images and markdown."""
        img1 = Image.new("RGB", (100, 100), color="red")
        img2 = Image.new("RGB", (100, 100), color="blue")
        img3 = Image.new("RGB", (100, 100), color="green")

        data = [
            {"corpus_id": "doc1", "image": img1, "markdown": "# Document 1\n\nRevenue: $100M"},
            {"corpus_id": "doc2", "image": img2, "markdown": "# Document 2\n\n## Chart Data"},
            {"corpus_id": "doc3", "image": img3, "markdown": "# Document 3\n\nTable content"},
        ]
        return MockDataset(data)

    @pytest.fixture
    def mock_qrels_dataset(self):
        """Create mock qrels dataset."""
        data = [
            {"query_id": 1, "corpus_id": "doc1", "score": 1},
            {"query_id": 2, "corpus_id": "doc2", "score": 1},
            {"query_id": 3, "corpus_id": "doc3", "score": 1},
        ]
        return MockDataset(data)

    def test_load_dataset_returns_correct_structure(
        self, mock_queries_dataset, mock_corpus_dataset, mock_qrels_dataset
    ):
        """Test that load_vidore_dataset returns correctly structured data."""
        with patch("vidore_benchmark.pipeline_evaluation.dataset_loader.load_dataset") as mock_load:

            def load_dataset_side_effect(name, data_dir, split):
                if data_dir == "queries":
                    return mock_queries_dataset
                elif data_dir == "corpus":
                    return mock_corpus_dataset
                elif data_dir == "qrels":
                    return mock_qrels_dataset

            mock_load.side_effect = load_dataset_side_effect

            result = load_vidore_dataset("vidore/vidore_v3_industrial")

            # Unpack results (7 items: corpus_images and corpus_texts are separate)
            query_ids, queries, corpus_ids, corpus_images, corpus_texts, qrels, query_languages = result

            # Verify query_ids
            assert query_ids == ["1", "2", "3"]

            # Verify queries
            assert queries == ["What is the revenue?", "Find the chart", "Quel est le chiffre?"]

            # Verify corpus_ids
            assert corpus_ids == ["doc1", "doc2", "doc3"]

            # Verify corpus_images
            assert len(corpus_images) == 3
            assert all(isinstance(img, Image.Image) for img in corpus_images)
            assert corpus_texts == [
                "# Document 1\n\nRevenue: $100M",
                "# Document 2\n\n## Chart Data",
                "# Document 3\n\nTable content",
            ]

            # Verify qrels format
            assert qrels == {
                "1": {"doc1": 1},
                "2": {"doc2": 1},
                "3": {"doc3": 1},
            }

            # Verify query_languages
            assert query_languages == {
                "1": "english",
                "2": "english",
                "3": "french",
            }

    def test_load_dataset_with_language_filter(self, mock_queries_dataset, mock_corpus_dataset, mock_qrels_dataset):
        """Test language filtering correctly filters queries and qrels."""
        with patch("vidore_benchmark.pipeline_evaluation.dataset_loader.load_dataset") as mock_load:

            def load_dataset_side_effect(name, data_dir, split):
                if data_dir == "queries":
                    return mock_queries_dataset
                elif data_dir == "corpus":
                    return mock_corpus_dataset
                elif data_dir == "qrels":
                    return mock_qrels_dataset

            mock_load.side_effect = load_dataset_side_effect

            result = load_vidore_dataset("vidore/vidore_v3_hr", language="english")

            query_ids, queries, corpus_ids, corpus_images, corpus_texts, qrels, query_languages = result

            # Only English queries should be included
            assert query_ids == ["1", "2"]
            assert queries == ["What is the revenue?", "Find the chart"]

            # Qrels should only include English queries
            assert "1" in qrels
            assert "2" in qrels
            assert "3" not in qrels

            # Query languages should only have English queries
            assert all(lang == "english" for lang in query_languages.values())

    def test_load_dataset_with_french_filter(self, mock_queries_dataset, mock_corpus_dataset, mock_qrels_dataset):
        """Test filtering for French queries only."""
        with patch("vidore_benchmark.pipeline_evaluation.dataset_loader.load_dataset") as mock_load:

            def load_dataset_side_effect(name, data_dir, split):
                if data_dir == "queries":
                    return mock_queries_dataset
                elif data_dir == "corpus":
                    return mock_corpus_dataset
                elif data_dir == "qrels":
                    return mock_qrels_dataset

            mock_load.side_effect = load_dataset_side_effect

            result = load_vidore_dataset("vidore/vidore_v3_hr", language="french")

            query_ids, queries, corpus_ids, corpus_images, corpus_texts, qrels, query_languages = result

            assert query_ids == ["3"]
            assert queries == ["Quel est le chiffre?"]
            assert query_languages == {"3": "french"}

    def test_invalid_dataset_name_raises_error(self):
        """Test that invalid dataset name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            load_vidore_dataset("invalid/dataset_name")

        assert "Unknown dataset" in str(exc_info.value)
        assert "invalid/dataset_name" in str(exc_info.value)

    def test_empty_queries_after_filter_raises_error(
        self, mock_queries_dataset, mock_corpus_dataset, mock_qrels_dataset
    ):
        """Test that filtering to no queries raises ValueError."""
        with patch("vidore_benchmark.pipeline_evaluation.dataset_loader.load_dataset") as mock_load:

            def load_dataset_side_effect(name, data_dir, split):
                if data_dir == "queries":
                    return mock_queries_dataset
                elif data_dir == "corpus":
                    return mock_corpus_dataset
                elif data_dir == "qrels":
                    return mock_qrels_dataset

            mock_load.side_effect = load_dataset_side_effect

            with pytest.raises(ValueError) as exc_info:
                load_vidore_dataset("vidore/vidore_v3_hr", language="german")

            assert "No queries found" in str(exc_info.value)
            assert "german" in str(exc_info.value)

    def test_hf_loading_error_raises_runtime_error(self):
        """Test that HuggingFace loading errors are wrapped in RuntimeError."""
        with patch("vidore_benchmark.pipeline_evaluation.dataset_loader.load_dataset") as mock_load:
            mock_load.side_effect = Exception("Network error")

            with pytest.raises(RuntimeError) as exc_info:
                load_vidore_dataset("vidore/vidore_v3_hr")

            assert "Failed to load dataset" in str(exc_info.value)
            assert "Network error" in str(exc_info.value)


class TestPrintDatasetInfo:
    def test_print_dataset_info_runs_without_error(self, capsys):
        """Test that print_dataset_info executes without errors."""
        img1 = Image.new("RGB", (100, 100), color="red")
        img2 = Image.new("RGB", (100, 100), color="blue")

        query_ids = ["q1", "q2"]
        queries = ["Query 1", "Query 2"]
        corpus_ids = ["doc1", "doc2"]
        corpus_images = [img1, img2]
        corpus_texts = ["Markdown 1", "Markdown 2"]
        qrels = {"q1": {"doc1": 1}, "q2": {"doc2": 1}}

        print_dataset_info(
            "test_dataset",
            query_ids,
            queries,
            corpus_ids,
            corpus_images,
            corpus_texts,
            qrels,
        )

        captured = capsys.readouterr()
        assert "test_dataset" in captured.out
        assert "Queries:" in captured.out
        assert "Corpus images:" in captured.out
        assert "Corpus texts" in captured.out
        assert "Sample query" in captured.out

    def test_print_dataset_info_shows_correct_counts(self, capsys):
        """Test that print_dataset_info shows correct statistics."""
        img = Image.new("RGB", (100, 100), color="red")

        query_ids = ["q1", "q2", "q3"]
        queries = ["Query 1", "Query 2", "Query 3"]
        corpus_ids = ["doc1", "doc2", "doc3", "doc4"]
        corpus_images = [img] * 4
        corpus_texts = ["Text 1", "Text 2", "Text 3", "Text 4"]
        qrels = {
            "q1": {"doc1": 1, "doc2": 1},
            "q2": {"doc3": 1},
            "q3": {"doc4": 1},
        }

        print_dataset_info(
            "test_dataset",
            query_ids,
            queries,
            corpus_ids,
            corpus_images,
            corpus_texts,
            qrels,
        )

        captured = capsys.readouterr()
        assert "3" in captured.out  # 3 queries
        assert "4" in captured.out  # 4 corpus items

    def test_print_dataset_info_mismatched_lengths_raises_error(self):
        """Test that mismatched corpus lengths raise AssertionError."""
        img = Image.new("RGB", (100, 100), color="red")

        query_ids = ["q1"]
        queries = ["Query 1"]
        corpus_ids = ["doc1", "doc2"]
        corpus_images = [img]  # Only 1 image but 2 IDs
        corpus_texts = ["Text 1", "Text 2"]
        qrels = {"q1": {"doc1": 1}}

        with pytest.raises(AssertionError):
            print_dataset_info(
                "test_dataset",
                query_ids,
                queries,
                corpus_ids,
                corpus_images,
                corpus_texts,
                qrels,
            )
