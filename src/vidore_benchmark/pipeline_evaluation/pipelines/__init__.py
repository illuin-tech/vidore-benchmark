"""
Built-in pipeline implementations.
"""

from vidore_benchmark.pipeline_evaluation.pipelines.file_based_pipeline import FileBasedPipeline
from vidore_benchmark.pipeline_evaluation.pipelines.random_pipeline import RandomPipeline

__all__ = ["RandomPipeline", "FileBasedPipeline"]
