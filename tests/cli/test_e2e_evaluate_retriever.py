import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from vidore_benchmark.main import app


@pytest.fixture
def cli_runner():
    """Fixture for typer CLI runner."""
    return CliRunner()


@pytest.mark.parametrize(
    "dataset_name,dataset_format",
    [
        ("vidore/syntheticDocQA_dummy", "qa"),
        ("vidore/syntheticDocQA_beir_dummy", "beir"),
    ],
)
def test_evaluate_retriever(cli_runner, dataset_name, dataset_format):
    """
    End-to-end test for the `evaluate_retriever` command.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define the model class
        model_class = "dummy_vision_retriever"

        # Run the CLI command
        result = cli_runner.invoke(
            app,
            [
                "evaluate-retriever",
                "--model-class",
                model_class,
                "--dataset-name",
                dataset_name,
                "--dataset-format",
                dataset_format,
                "--split",
                "test",
                "--batch-query",
                "2",
                "--batch-passage",
                "2",
                "--batch-score",
                "2",
                "--output-dir",
                temp_dir,
            ],
        )

        # Assert
        assert result.exit_code == 0, f"CLI command failed with error: {result.stdout}"

        # Check if metrics file was created
        metrics_file = Path(temp_dir) / f"{model_class}_metrics.json"
        assert metrics_file.exists(), "Metrics file was not created"

        # Load and validate metrics
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        # Check if metrics contain the expected dataset
        assert dataset_name in metrics, f"Metrics for dataset {dataset_name} not found"

        # Check metrics are not empty
        dataset_metrics = metrics[dataset_name]

        # Check if metrics are within valid range (0 to 1)
        for metric_name, metric_value in dataset_metrics.items():
            if not np.isnan(metric_value) and "naucs" not in metric_name:
                assert 0 <= metric_value <= 1, f"Metric {metric_name} outside valid range: {metric_value}"
