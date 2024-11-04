from datetime import datetime

import pytest

from vidore_benchmark.evaluation.interfaces import MetadataModel, MetricsModel, ViDoReBenchmarkResults


class TestViDoReBenchmarkCreation:
    def test_metrics_model_creation(self):
        """Test basic creation of MetricsModel with valid data"""
        metrics = MetricsModel(
            ndcg_at_1=0.5,
            ndcg_at_3=0.7,
            map_at_1=0.4,
        )

        assert metrics.ndcg_at_1 == 0.5
        assert metrics.ndcg_at_3 == 0.7
        assert metrics.map_at_1 == 0.4

    def test_metadata_creation(self):
        """Test creation of Metadata with required fields"""
        current_time = datetime.now()
        metadata = MetadataModel(timestamp=current_time, vidore_benchmark_hash="1234567890abcdef")

        assert metadata.timestamp == current_time
        assert metadata.vidore_benchmark_hash == "1234567890abcdef"

    def test_metadata_extra_fields(self):
        """Test that Metadata allows extra fields"""
        metadata = MetadataModel(
            timestamp=datetime.now(),
            vidore_benchmark_hash="1234567890abcdef",
            extra_field="extra_value",
        )

        assert metadata.extra_field == "extra_value"

    def test_vidore_benchmark_results_creation(self):
        """Test creation of complete ViDoReBenchmarkResults"""
        current_time = datetime.now()
        results = ViDoReBenchmarkResults(
            metadata=MetadataModel(
                timestamp=current_time,
                vidore_benchmark_hash="1234567890abcdef",
            ),
            metrics={
                "test_dataset": MetricsModel(ndcg_at_1=0.5, ndcg_at_3=0.7),
            },
        )

        assert results.metadata.timestamp == current_time
        assert results.metadata.vidore_benchmark_hash == "1234567890abcdef"
        assert results.metrics["test_dataset"].ndcg_at_1 == 0.5
        assert results.metrics["test_dataset"].ndcg_at_3 == 0.7


class TestViDoReBenchmarkMerging:
    def test_merge_results(self):
        """Test basic merging of two results with different benchmarks."""
        result1 = ViDoReBenchmarkResults(
            metadata=MetadataModel(timestamp=datetime(2024, 1, 1), vidore_benchmark_hash="hash1"),
            metrics={"benchmark1": MetricsModel(ndcg_at_1=0.5)},
        )

        result2 = ViDoReBenchmarkResults(
            metadata=MetadataModel(timestamp=datetime(2024, 2, 1), vidore_benchmark_hash="hash2"),
            metrics={"benchmark2": MetricsModel(ndcg_at_1=0.7)},
        )

        merged = ViDoReBenchmarkResults.merge([result1, result2])

        # Check that metadata comes from the latest result
        assert merged.metadata.timestamp == datetime(2024, 2, 1)
        assert merged.metadata.vidore_benchmark_hash == "hash2"

        # Check that metrics from both results are present
        assert len(merged.metrics) == 2
        assert merged.metrics["benchmark1"].ndcg_at_1 == 0.5
        assert merged.metrics["benchmark2"].ndcg_at_1 == 0.7

    def test_merge_duplicate_keys_error(self):
        """Test that merging results with duplicate benchmark keys raises ValueError."""
        result1 = ViDoReBenchmarkResults(
            metadata=MetadataModel(timestamp=datetime(2024, 1, 1), vidore_benchmark_hash="hash1"),
            metrics={
                "benchmark1": MetricsModel(ndcg_at_1=0.5),
                "benchmark2": MetricsModel(ndcg_at_1=0.6),
            },
        )

        result2 = ViDoReBenchmarkResults(
            metadata=MetadataModel(timestamp=datetime(2024, 2, 1), vidore_benchmark_hash="hash2"),
            metrics={
                "benchmark2": MetricsModel(ndcg_at_1=0.7),  # Duplicate key with result1
                "benchmark3": MetricsModel(ndcg_at_1=0.8),
            },
        )

        # Should raise ValueError due to duplicate 'benchmark2' key
        with pytest.raises(ValueError, match="Duplicate dataset keys found in the input results"):
            ViDoReBenchmarkResults.merge([result1, result2])
