import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vidore_benchmark.evaluation.interfaces import ViDoReBenchmarkResults
from vidore_benchmark.main import app


@pytest.fixture
def output_dir():
    """Fixture to create and clean up the output directory."""
    output_path = Path("outputs")
    output_path.mkdir(exist_ok=True)
    yield output_path

    # Clean up output files after test
    if output_path.exists():
        for file in output_path.glob("*"):
            file.unlink()
        output_path.rmdir()


@pytest.fixture
def cli_runner():
    """Fixture for typer CLI runner."""
    return CliRunner()


def test_evaluate_retriever_e2e(cli_runner, output_dir):
    """
    End-to-end test for evaluate_retriever command using a dummy dataset and model.
    """
    # Arrange
    dataset_name = "vidore/syntheticDocQA_dummy"
    model_class = "dummy_retriever"

    # Act
    result = cli_runner.invoke(
        app,
        [
            "evaluate-retriever",
            "--model-class",
            model_class,
            "--dataset-name",
            dataset_name,
            "--split",
            "test",
            "--batch-query",
            "2",
            "--batch-passage",
            "2",
            "--batch-score",
            "2",
        ],
    )

    # Assert
    assert result.exit_code == 0, f"CLI command failed with error: {result.stdout}"

    # Check if result file was created
    vidore_results_file = output_dir / f"{model_class}_metrics.json"
    assert vidore_results_file.exists(), "Metrics file was not created"

    # Load JSON
    try:
        with open(vidore_results_file, "r", encoding="utf-8") as f:
            vidore_results = json.load(f)
    except Exception as e:
        pytest.fail(f"Failed to load JSON file: {e}")

    # Load results using the ViDoReBenchmarkResults format
    try:
        vidore_results = ViDoReBenchmarkResults(**vidore_results)
    except Exception as e:
        pytest.fail(f"Failed to load results using the `ViDoReBenchmarkResults` format: {e}")

    metrics = vidore_results.metrics

    # Check if metrics contain the expected dataset
    assert dataset_name in metrics, f"Metrics for dataset {dataset_name} not found"

    # Check metrics are not empty
    dataset_metrics = metrics[dataset_name]

    # Check if metrics are within valid range (0 to 1)
    for metric_name, metric_value in dataset_metrics.items():
        if not np.isnan(metric_value):
            assert 0 <= metric_value <= 1, f"Metric {metric_name} outside valid range: {metric_value}"

    # Check for specific NDCG@5 output in stdout
    assert f"NDCG@5 for {model_class} on {dataset_name}:" in result.stdout
