import json
import tempfile
from pathlib import Path
from typing import Optional

import pytest
from typer.testing import CliRunner

from vidore_benchmark.cli.main import _sanitize_model_id, app
from vidore_benchmark.evaluation.interfaces import ViDoReBenchmarkResults


def _are_vidore_results_close(
    result_1: ViDoReBenchmarkResults,
    result_2: ViDoReBenchmarkResults,
    tolerance: float = 1e-3,
) -> bool:
    """
    Check if two `ViDoReBenchmarkResults` objects are close within a tolerance.

    Args:
        result1: First ViDoReBenchmarkResults object.
        result2: Second ViDoReBenchmarkResults object.
        tolerance: Tolerance for comparison.

    Returns:
        True if the results are close within the tolerance, False otherwise.
    """
    if result_1.metrics.keys() != result_2.metrics.keys():
        return False

    for dataset in result_1.metrics:
        metrics1 = result_1.metrics[dataset]
        metrics2 = result_2.metrics[dataset]

        if metrics1.keys() != metrics2.keys():
            return False

        for metric, value1 in metrics1.items():
            value2 = metrics2[metric]
            if value1 is None and value2 is None:
                continue
            if value1 is None or value2 is None:
                return False
            if abs(value1 - value2) > tolerance:
                return False

    return True


@pytest.fixture
def cli_runner():
    """Fixture for typer CLI runner."""
    return CliRunner()


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_class,model_name,dataset_name,dataset_format",
    [
        ("siglip", "google/siglip-so400m-patch14-384", "vidore/tabfquad_test_subsampled", "qa"),
        ("bm25", None, "vidore/tabfquad_test_subsampled_ocr_chunk", "qa"),
        ("siglip", "google/siglip-so400m-patch14-384", "vidore/tabfquad_test_subsampled_beir", "beir"),
    ],
)
def test_e2e_evaluate_retriever(
    cli_runner: CliRunner,
    model_class: str,
    model_name: Optional[str],
    dataset_name: str,
    dataset_format: str,
):
    """
    End-to-end test for the `evaluate_retriever` command.
    """
    # Load expected results for comparison
    expected_results_path = Path("tests/data/e2e_vidore_results/google_siglip-so400m-patch14-384_metrics.json")
    with open(expected_results_path, "r", encoding="utf-8") as f:
        expected_results = ViDoReBenchmarkResults(**json.load(f))

    with tempfile.TemporaryDirectory() as temp_dir:
        # Define the model used for retrieval
        model_class = "siglip"
        model_name = "google/siglip-so400m-patch14-384"

        # Run the CLI command
        result = cli_runner.invoke(
            app,
            [
                "evaluate-retriever",
                "--model-class",
                model_class,
                "--model-name",
                model_name,
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
        model_id = _sanitize_model_id(model_class, model_name)
        vidore_results_file = Path(temp_dir) / f"{model_id}_metrics.json"
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

        # Verify results match expected with some tolerance
        if not _are_vidore_results_close(vidore_results, expected_results):
            # Copy the results file to outputs directory for debugging
            outputs_dir = Path("outputs")
            outputs_dir.mkdir(exist_ok=True, parents=True)
            vidore_results_file_copy = outputs_dir / vidore_results_file.name
            vidore_results_file.rename(vidore_results_file_copy)

            pytest.fail(
                f"Results do not match expected. "
                f"Check {vidore_results_file_copy} and {expected_results_path} for details."
            )

        metrics = vidore_results.metrics

        # Check if metrics contain the expected dataset
        assert dataset_name in metrics, f"Metrics for dataset {dataset_name} not found"

        # Check for specific nDCG@5 output in stdout
        assert f"nDCG@5 for {model_class} on {dataset_name}:" in result.stdout
