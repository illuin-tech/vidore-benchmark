import json
import tempfile
from pathlib import Path
from typing import Optional

import pytest
from typer.testing import CliRunner

from vidore_benchmark.cli.main import _sanitize_model_id, app
from vidore_benchmark.evaluation.interfaces import ViDoReBenchmarkResults


def _are_vidore_ndcg_results_close(
    result_1: ViDoReBenchmarkResults,
    result_2: ViDoReBenchmarkResults,
    tolerance: float = 3e-2,
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
            if metric.startswith("ndcg_"):
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
    "model_class,model_name,dataset_name,dataset_format,expected_results_filepath",
    [
        (
            "colidefics3",
            "vidore/colSmol-256M",
            "vidore/tabfquad_test_subsampled",
            "qa",
            "tests/data/e2e_vidore_results/colsmol_256M_tabfquad.json",
        ),
        (
            "bm25",
            None,
            "vidore/tabfquad_test_subsampled_ocr_chunk",
            "qa",
            "tests/data/e2e_vidore_results/bm25_tabfquad.json",
        ),
    ],
)
def test_e2e_evaluate_retriever_on_one_dataset(
    cli_runner: CliRunner,
    model_class: str,
    model_name: Optional[str],
    dataset_name: str,
    dataset_format: str,
    expected_results_filepath: str,
):
    """
    End-to-end test for the `evaluate_retriever` command.
    """
    expected_results_path = Path(expected_results_filepath)
    with open(expected_results_path, "r", encoding="utf-8") as f:
        expected_results = ViDoReBenchmarkResults(**json.load(f))

    with tempfile.TemporaryDirectory() as temp_dir:
        result = cli_runner.invoke(
            app,
            [
                "evaluate-retriever",
                "--model-class",
                model_class,
                "--model-name",
                model_name if model_name else "",
                "--dataset-name",
                dataset_name,
                "--dataset-format",
                dataset_format,
                "--split",
                "test",
                "--batch-query",
                "4",
                "--batch-passage",
                "4",
                "--batch-score",
                "4",
                "--output-dir",
                temp_dir,
            ],
        )

        assert result.exit_code == 0, f"CLI command failed with error: {result.stdout}"

        model_id = _sanitize_model_id(model_class, model_name)
        vidore_results_file = Path(temp_dir) / f"{model_id}_metrics.json"
        print(f"Metrics file path: {vidore_results_file}")
        assert vidore_results_file.exists(), "Metrics file was not created"

        try:
            with open(vidore_results_file, "r", encoding="utf-8") as f:
                vidore_results = json.load(f)
        except Exception as e:
            pytest.fail(f"Failed to load JSON file: {e}")

        try:
            vidore_results = ViDoReBenchmarkResults(**vidore_results)
        except Exception as e:
            pytest.fail(f"Failed to load results using the `ViDoReBenchmarkResults` format: {e}")

        if not _are_vidore_ndcg_results_close(vidore_results, expected_results):
            # Copy the results file to outputs directory for debugging
            outputs_dir = Path("outputs")
            outputs_dir.mkdir(exist_ok=True, parents=True)
            vidore_results_file_copy = outputs_dir / vidore_results_file.name
            vidore_results_file.rename(vidore_results_file_copy)

            pytest.fail(
                f"Results do not match expected. Check `{vidore_results_file_copy}` (output) and "
                "{expected_results_path}` (expected) for more details."
            )

        metrics = vidore_results.metrics
        assert dataset_name in metrics, f"Metrics for dataset {dataset_name} not found"
