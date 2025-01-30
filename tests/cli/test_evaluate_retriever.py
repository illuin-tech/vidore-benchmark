import json
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vidore_benchmark.cli.main import app
from vidore_benchmark.evaluation.interfaces import ViDoReBenchmarkResults


@pytest.fixture
def cli_runner():
    """Fixture for typer CLI runner."""
    return CliRunner()


@pytest.mark.parametrize(
    "dataset_name,dataset_format",
    [
        ("vidore/vidore_benchmark_qa_dummy", "qa"),
        ("vidore/syntheticDocQA_beir_dummy", "beir"),
    ],
)
def test_evaluate_retriever(
    cli_runner: CliRunner,
    dataset_name: str,
    dataset_format: str,
):
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

        # Check if result file was created
        vidore_results_file = Path(temp_dir) / f"{model_class}_metrics.json"
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

        # Check for specific nDCG@5 output in stdout
        assert f"nDCG@5 for {model_class} on {dataset_name}:" in result.stdout
