"""
Tests for example pipeline implementations.
"""

import json

import pytest

from vidore_benchmark.pipeline_evaluation.base_pipeline import BasePipeline
from vidore_benchmark.pipeline_evaluation.pipelines.file_based_pipeline import FileBasedPipeline
from vidore_benchmark.pipeline_evaluation.pipelines.random_pipeline import RandomPipeline


class TestRandomPipeline:
    """Tests for RandomPipeline implementation."""

    def test_random_pipeline_is_base_pipeline_subclass(self):
        """Test that RandomPipeline inherits from BasePipeline."""
        pipeline = RandomPipeline()
        assert isinstance(pipeline, BasePipeline)

    def test_default_initialization(self):
        """Test default initialization parameters."""
        pipeline = RandomPipeline()
        assert pipeline.seed == 42
        assert pipeline.top_k == 10

    def test_custom_initialization(self):
        """Test custom initialization parameters."""
        pipeline = RandomPipeline(seed=123, top_k=5)
        assert pipeline.seed == 123
        assert pipeline.top_k == 5

    def test_retrieve_returns_dict(self):
        """Test that retrieve returns a dictionary."""
        pipeline = RandomPipeline()

        query_ids = ["q1", "q2"]
        queries = ["Query 1", "Query 2"]
        corpus_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        corpus_images = [None] * 5  # Mock images
        corpus_texts = ["Text"] * 5

        results = pipeline.retrieve(query_ids, queries, corpus_ids, corpus_images, corpus_texts)

        assert isinstance(results, dict)
        assert set(results.keys()) == {"q1", "q2"}

    def test_retrieve_respects_top_k(self):
        """Test that retrieve returns at most top_k results per query."""
        pipeline = RandomPipeline(top_k=3)

        query_ids = ["q1"]
        queries = ["Query 1"]
        corpus_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        corpus_images = [None] * 5
        corpus_texts = ["Text"] * 5

        results = pipeline.retrieve(query_ids, queries, corpus_ids, corpus_images, corpus_texts)

        assert len(results["q1"]) == 3

    def test_retrieve_handles_small_corpus(self):
        """Test that retrieve handles corpus smaller than top_k."""
        pipeline = RandomPipeline(top_k=10)

        query_ids = ["q1"]
        queries = ["Query 1"]
        corpus_ids = ["doc1", "doc2"]  # Only 2 documents
        corpus_images = [None] * 2
        corpus_texts = ["Text"] * 2

        results = pipeline.retrieve(query_ids, queries, corpus_ids, corpus_images, corpus_texts)

        assert len(results["q1"]) == 2  # Should return all 2, not 10

    def test_retrieve_scores_are_floats(self):
        """Test that all retrieval scores are floats."""
        pipeline = RandomPipeline()

        query_ids = ["q1"]
        queries = ["Query 1"]
        corpus_ids = ["doc1", "doc2", "doc3"]
        corpus_images = [None] * 3
        corpus_texts = ["Text"] * 3

        results = pipeline.retrieve(query_ids, queries, corpus_ids, corpus_images, corpus_texts)

        for corpus_id, score in results["q1"].items():
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        query_ids = ["q1", "q2"]
        queries = ["Query 1", "Query 2"]
        corpus_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        corpus_images = [None] * 5
        corpus_texts = ["Text"] * 5

        pipeline1 = RandomPipeline(seed=42)
        results1 = pipeline1.retrieve(query_ids, queries, corpus_ids, corpus_images, corpus_texts)

        pipeline2 = RandomPipeline(seed=42)
        results2 = pipeline2.retrieve(query_ids, queries, corpus_ids, corpus_images, corpus_texts)

        assert results1 == results2

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        query_ids = ["q1"]
        queries = ["Query 1"]
        corpus_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        corpus_images = [None] * 5
        corpus_texts = ["Text"] * 5

        pipeline1 = RandomPipeline(seed=42)
        results1 = pipeline1.retrieve(query_ids, queries, corpus_ids, corpus_images, corpus_texts)

        pipeline2 = RandomPipeline(seed=123)
        results2 = pipeline2.retrieve(query_ids, queries, corpus_ids, corpus_images, corpus_texts)

        # Results should be different (with very high probability)
        assert results1 != results2

    def test_retrieve_with_empty_corpus(self):
        """Test retrieve with empty corpus."""
        pipeline = RandomPipeline()

        query_ids = ["q1"]
        queries = ["Query 1"]
        corpus_ids = []
        corpus_images = []
        corpus_texts = []

        results = pipeline.retrieve(query_ids, queries, corpus_ids, corpus_images, corpus_texts)

        assert results["q1"] == {}

    def test_retrieve_with_empty_queries(self):
        """Test retrieve with empty queries."""
        pipeline = RandomPipeline()

        query_ids = []
        queries = []
        corpus_ids = ["doc1", "doc2"]
        corpus_images = [None] * 2
        corpus_texts = ["Text"] * 2

        results = pipeline.retrieve(query_ids, queries, corpus_ids, corpus_images, corpus_texts)

        assert results == {}


class TestFileBasedPipeline:
    """Tests for FileBasedPipeline implementation."""

    @pytest.fixture
    def valid_run_file(self, tmp_path):
        """Create a valid run file for testing."""
        run_data = {
            "q1": {"doc1": 0.95, "doc2": 0.87, "doc3": 0.72},
            "q2": {"doc2": 0.91, "doc1": 0.83, "doc4": 0.65},
            "q3": {"doc3": 0.88, "doc5": 0.76},
        }
        file_path = tmp_path / "valid_run.json"
        with open(file_path, "w") as f:
            json.dump(run_data, f)
        return file_path

    @pytest.fixture
    def invalid_structure_file(self, tmp_path):
        """Create a file with invalid structure (not nested dict)."""
        run_data = {"q1": "not_a_dict", "q2": {"doc1": 0.5}}
        file_path = tmp_path / "invalid_structure.json"
        with open(file_path, "w") as f:
            json.dump(run_data, f)
        return file_path

    @pytest.fixture
    def invalid_json_file(self, tmp_path):
        """Create a file with invalid JSON."""
        file_path = tmp_path / "invalid.json"
        with open(file_path, "w") as f:
            f.write("not valid json {{{")
        return file_path

    def test_file_based_pipeline_is_base_pipeline_subclass(self, valid_run_file):
        """Test that FileBasedPipeline inherits from BasePipeline."""
        pipeline = FileBasedPipeline(str(valid_run_file))
        assert isinstance(pipeline, BasePipeline)

    def test_load_valid_run_file(self, valid_run_file):
        """Test loading a valid run file."""
        pipeline = FileBasedPipeline(str(valid_run_file))

        assert pipeline.run_file_path == str(valid_run_file)
        assert "q1" in pipeline.run_data
        assert "q2" in pipeline.run_data
        assert "q3" in pipeline.run_data

    def test_retrieve_returns_loaded_data(self, valid_run_file):
        """Test that retrieve returns the loaded run data."""
        pipeline = FileBasedPipeline(str(valid_run_file))

        # These arguments are ignored by FileBasedPipeline
        query_ids = ["q1", "q2", "q3"]
        queries = ["Query 1", "Query 2", "Query 3"]
        corpus_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        corpus_images = [None] * 5
        corpus_texts = ["Text"] * 5

        results = pipeline.retrieve(query_ids, queries, corpus_ids, corpus_images, corpus_texts)

        assert results["q1"]["doc1"] == 0.95
        assert results["q2"]["doc2"] == 0.91
        assert results["q3"]["doc3"] == 0.88

    def test_file_not_found_raises_error(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            FileBasedPipeline("/nonexistent/path/to/file.json")

        assert "not found" in str(exc_info.value).lower()

    def test_invalid_json_raises_error(self, invalid_json_file):
        """Test that invalid JSON raises JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            FileBasedPipeline(str(invalid_json_file))

    def test_invalid_structure_raises_error(self, invalid_structure_file):
        """Test that invalid structure raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            FileBasedPipeline(str(invalid_structure_file))

        assert "dict" in str(exc_info.value).lower()

    def test_non_dict_root_raises_error(self, tmp_path):
        """Test that a non-dict root raises ValueError."""
        file_path = tmp_path / "array_root.json"
        with open(file_path, "w") as f:
            json.dump(["not", "a", "dict"], f)

        with pytest.raises(ValueError) as exc_info:
            FileBasedPipeline(str(file_path))

        assert "dict" in str(exc_info.value).lower()

    def test_retrieve_ignores_input_arguments(self, valid_run_file):
        """Test that retrieve ignores input arguments and uses loaded data."""
        pipeline = FileBasedPipeline(str(valid_run_file))

        # Pass completely different query/corpus IDs
        results1 = pipeline.retrieve(["x1"], ["X query"], ["y1"], [None], ["Text"])
        results2 = pipeline.retrieve(["z1", "z2"], ["Z query 1", "Z query 2"], ["w1"], [None], ["Text"])

        # Both should return the same loaded data
        assert results1 == results2
        assert results1 == pipeline.run_data

    def test_empty_run_file(self, tmp_path):
        """Test loading an empty run file."""
        file_path = tmp_path / "empty.json"
        with open(file_path, "w") as f:
            json.dump({}, f)

        pipeline = FileBasedPipeline(str(file_path))
        assert pipeline.run_data == {}

        results = pipeline.retrieve(["q1"], ["Query"], ["doc1"], [None], ["Text"])
        assert results == {}

    def test_scores_preserved_exactly(self, tmp_path):
        """Test that score values are preserved exactly."""
        run_data = {"q1": {"doc1": 0.123456789, "doc2": 1e-10, "doc3": 0.999999999}}
        file_path = tmp_path / "precise_scores.json"
        with open(file_path, "w") as f:
            json.dump(run_data, f)

        pipeline = FileBasedPipeline(str(file_path))
        results = pipeline.retrieve(["q1"], ["Query"], ["doc1"], [None], ["Text"])

        assert results["q1"]["doc1"] == 0.123456789
        assert results["q1"]["doc2"] == 1e-10
        assert results["q1"]["doc3"] == 0.999999999
