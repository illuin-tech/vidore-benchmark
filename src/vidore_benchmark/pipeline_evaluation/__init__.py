"""
vidore-eval: Python CLI tool for evaluating pipelines on vidore v3 datasets.
"""

__version__ = "0.1.0"

from vidore_benchmark.pipeline_evaluation.base_pipeline import BasePipeline
from vidore_benchmark.pipeline_evaluation.dataset_loader import (
    get_available_datasets,
    load_vidore_dataset,
    print_dataset_info,
)
from vidore_benchmark.pipeline_evaluation.evaluator import aggregate_results, evaluate_retrieval
from vidore_benchmark.pipeline_evaluation.pipelines.file_based_pipeline import FileBasedPipeline
from vidore_benchmark.pipeline_evaluation.pipelines.random_pipeline import RandomPipeline

__all__ = [
    "BasePipeline",
    "evaluate_retrieval",
    "aggregate_results",
    "load_vidore_dataset",
    "get_available_datasets",
    "print_dataset_info",
    "FileBasedPipeline",
    "RandomPipeline",
]
