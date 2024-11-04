from datetime import datetime

from vidore_benchmark.evaluation.interfaces import Metadata, MetricsModel, ViDoReBenchmarkResults


def test_metrics_model_creation():
    """Test basic creation of MetricsModel with valid data"""
    metrics = MetricsModel(
        ndcg_at_1=0.5,
        ndcg_at_3=0.7,
        map_at_1=0.4,
    )

    assert metrics.ndcg_at_1 == 0.5
    assert metrics.ndcg_at_3 == 0.7
    assert metrics.map_at_1 == 0.4


def test_metadata_creation():
    """Test creation of Metadata with required fields"""
    current_time = datetime.now()
    metadata = Metadata(timestamp=current_time, vidore_benchmark_hash="1234567890abcdef")

    assert metadata.timestamp == current_time
    assert metadata.vidore_benchmark_hash == "1234567890abcdef"


def test_metadata_extra_fields():
    """Test that Metadata allows extra fields"""
    metadata = Metadata(
        timestamp=datetime.now(),
        vidore_benchmark_hash="1234567890abcdef",
        extra_field="extra_value",
    )

    assert metadata.extra_field == "extra_value"


def test_vidore_benchmark_results_creation():
    """Test creation of complete ViDoReBenchmarkResults"""
    current_time = datetime.now()

    results = ViDoReBenchmarkResults(
        metadata=Metadata(
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
