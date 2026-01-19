"""
Core evaluation orchestration using pytrec_eval.
"""

import time
from typing import Any, Dict, List

import pytrec_eval

from vidore_benchmark.pipeline_evaluation.base_pipeline import BasePipeline


def evaluate_retrieval(
    pipeline: BasePipeline,
    query_ids: List[str],
    queries: List[str],
    corpus_ids: List[str],
    corpus: List[Any],
    qrels: Dict[str, Dict[str, int]],
    metrics: List[str] = None,
    track_time: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a pipeline using pytrec_eval.

    Args:
        pipeline: Instance of BasePipeline with user's pipeline logic
        query_ids: List of query identifiers
        queries: List of query texts
        corpus_ids: List of corpus item identifiers
        corpus: List of corpus items (e.g., PIL.Image objects)
        qrels: Ground truth relevance judgments in pytrec_eval format
               {query_id: {doc_id: relevance_score}}
        metrics: List of metrics to calculate (default: ['ndcg_cut_10'])
        track_time: Whether to track retrieval time (default: True)

    Returns:
        Dictionary of evaluation results per query:
        {
            'q1': {'ndcg_cut_10': 0.85, ...},
            'q2': {'ndcg_cut_10': 0.72, ...},
            ...
        }
        If track_time=True, also includes timing information in a special '_timing' key.
    """
    if metrics is None:
        metrics = ["ndcg_cut_10"]

    # Call user's pipeline implementation with time tracking
    if track_time:
        start_time = time.time()

    run = pipeline.retrieve(query_ids, queries, corpus_ids, corpus)

    if track_time:
        end_time = time.time()
        total_time = (end_time - start_time) * 1000  # in milliseconds

    # Validate run format
    if not isinstance(run, dict):
        raise ValueError(f"Pipeline must return a dict, got {type(run)}")

    for query_id in query_ids:
        if query_id not in run:
            # If pipeline didn't return results for a query, add empty results
            run[query_id] = {}

    # Create pytrec_eval evaluator
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, set(metrics))

    # Evaluate
    results = evaluator.evaluate(run)

    # Add timing information if tracking
    if track_time:
        num_queries = len(query_ids)
        results["_timing"] = {
            "total_retrieval_time_milliseconds": total_time,
            "average_time_per_query_milliseconds": total_time / num_queries if num_queries > 0 else 0.0,
            "num_queries": num_queries,
            "queries_per_second": num_queries / (total_time / 1000) if total_time > 0 else 0.0,
        }

    return results


def aggregate_results(results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate aggregate statistics across all queries.

    Args:
        results: Per-query evaluation results from evaluate_retrieval()

    Returns:
        Dictionary of aggregated metrics (mean across queries).
        If timing information is present, it is included directly.
    """
    if not results:
        return {}

    # Extract timing information if present
    timing_info = results.pop("_timing", None)

    if not results:
        # Only timing info was present
        return timing_info if timing_info else {}

    # Get all metric names from first query
    metric_names = list(next(iter(results.values())).keys())

    aggregated = {}
    for metric in metric_names:
        scores = [results[qid][metric] for qid in results]
        aggregated[metric] = sum(scores) / len(scores)

    # Add timing information back if it was present
    if timing_info:
        aggregated.update(timing_info)

    return aggregated
