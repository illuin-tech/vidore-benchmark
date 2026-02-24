"""
FileBasedPipeline: Load pre-computed retrieval results from a JSON file.

This pipeline is useful for evaluating retrieval results that were
computed offline or by external systems. The run file should be in
pytrec_eval format: {query_id: {corpus_id: score}}.
"""

import json
from typing import Dict, List

from vidore_benchmark.pipeline_evaluation.base_pipeline import BasePipeline


class FileBasedPipeline(BasePipeline):
    """
    A pipeline that loads pre-computed retrieval results from a JSON file.

    The JSON file should contain retrieval results in the format:
    {
        "query_id_1": {"corpus_id_1": score, "corpus_id_2": score, ...},
        "query_id_2": {"corpus_id_1": score, "corpus_id_3": score, ...},
        ...
    }

    This format matches the pytrec_eval "run" format, where each query
    maps to a dictionary of corpus IDs and their retrieval scores.

    Example:
        >>> pipeline = FileBasedPipeline("my_results.json")
        >>> results = evaluate_retrieval(
        ...     pipeline, query_ids, queries, corpus_ids, corpus, qrels
        ... )
    """

    def __init__(self, run_file_path: str):
        """
        Initialize the FileBasedPipeline.

        Args:
            run_file_path: Path to JSON file containing pre-computed results

        Raises:
            FileNotFoundError: If the run file doesn't exist
            json.JSONDecodeError: If the file is not valid JSON
            ValueError: If the JSON structure is invalid
        """
        self.run_file_path = run_file_path

        # Load the run file
        try:
            with open(run_file_path, "r") as f:
                self.run_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Run file not found: {run_file_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in run file {run_file_path}: {e.msg}", e.doc, e.pos)

        # Validate structure
        if not isinstance(self.run_data, dict):
            raise ValueError(f"Run file must contain a JSON object (dict), got {type(self.run_data)}")

        # Validate that values are dicts (query_id -> {corpus_id: score})
        for query_id, corpus_scores in self.run_data.items():
            if not isinstance(corpus_scores, dict):
                raise ValueError(
                    f"Each query must map to a dict of corpus scores, "
                    f"but query '{query_id}' maps to {type(corpus_scores)}"
                )

    def search(self, query_ids: List[str], queries: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Return pre-computed retrieval results from the loaded file.

        Note: The corpus and queries parameters are not used by this pipeline,
        as results were pre-computed. However, they are part of the BasePipeline
        interface signature.

        Args:
            query_ids: List of query identifiers
            queries: List of query texts (not used)
            corpus_ids: List of corpus item identifiers (not used)
            corpus_images: List of corpus images (not used)
            corpus_texts: List of corpus texts (not used)

        Returns:
            Dictionary mapping query_id to {corpus_id: score} pairs,
            loaded from the run file
        """
        # Note: We could add validation here to check that run_data
        # contains results for all query_ids, but we'll let pytrec_eval
        # handle missing queries gracefully (as it does in evaluator.py)

        return self.run_data
