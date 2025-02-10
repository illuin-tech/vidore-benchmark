import json
import logging
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vidore_benchmark.cli.main import _sanitize_model_id, app
from vidore_benchmark.evaluation.interfaces import ViDoReBenchmarkResults

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def cli_runner():
    """Fixture for typer CLI runner."""
    return CliRunner()


@pytest.mark.parametrize(
    "model_class,dataset_name,dataset_format",
    [
        ("dummy_vision_retriever", "vidore/vidore_benchmark_qa_dummy", "qa"),
        ("bm25", "vidore/vidore_benchmark_ocr_qa_dummy", "qa"),
        ("dummy_vision_retriever", "vidore/vidore_benchmark_beir_dummy", "beir"),
    ],
)
def test_run_evaluate_retriever(
    cli_runner: CliRunner,
    model_class: str,
    dataset_name: str,
    dataset_format: str,
):
    """
    End-to-end test for the `evaluate_retriever` command.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
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

        logger.info(f"CLI output: {result.stdout}")

        # Check if the CLI command ran successfully
        assert result.exit_code == 0, f"CLI command failed with error: {result.stdout}"

        # Assert that the metrics file was properly created
        model_id = _sanitize_model_id(model_class)
        vidore_results_file = Path(temp_dir) / f"{model_id}_metrics.json"
        assert vidore_results_file.exists(), "Metrics file was not created"
        try:
            with open(vidore_results_file, "r", encoding="utf-8") as f:
                vidore_results = json.load(f)
        except Exception as e:
            pytest.fail(f"Failed to load JSON file: {e}")

        # Assert that the results have the correct format
        try:
            vidore_results = ViDoReBenchmarkResults(**vidore_results)
        except Exception as e:
            pytest.fail(f"Failed to load results using the `ViDoReBenchmarkResults` format: {e}")

        # Assert that the dataset name is present in the metrics
        metrics = vidore_results.metrics
        assert dataset_name in metrics, f"Metrics for dataset {dataset_name} not found"
