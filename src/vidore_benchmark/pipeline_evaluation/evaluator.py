"""
Core evaluation orchestration using pytrec_eval.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional

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
) -> Dict[str, Any]:
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

    # Call user's pipeline implementation.
    #
    # Contract:
    # {
    #   query_id: {
    #     "results": {corpus_id: score, ...},
    #     "runtime_milliseconds": float,
    #   },
    #   ...
    # }
    run_with_timing = pipeline.retrieve(query_ids, queries, corpus_ids, corpus)

    # Validate run format
    if not isinstance(run_with_timing, dict):
        raise ValueError(f"Pipeline must return a dict, got {type(run_with_timing)}")

    run: Dict[str, Dict[str, float]] = {}
    per_query_runtime_milliseconds: Dict[str, float] = {}
    for query_id in query_ids:
        if query_id not in run_with_timing:
            # If pipeline didn't return results for a query, add empty results.
            run[query_id] = {}
            per_query_runtime_milliseconds[query_id] = 0.0
            continue

        query_payload = run_with_timing[query_id]
        if not isinstance(query_payload, dict):
            raise ValueError(
                f"Pipeline must return dict payload per query_id, but query '{query_id}' "
                f"maps to {type(query_payload)}"
            )

        if "results" not in query_payload:
            raise ValueError(f"Pipeline payload for query '{query_id}' missing required key 'results'")
        if "runtime_milliseconds" not in query_payload:
            raise ValueError(
                f"Pipeline payload for query '{query_id}' missing required key 'runtime_milliseconds'"
            )

        query_results = query_payload["results"]
        if not isinstance(query_results, dict):
            raise ValueError(
                f"Pipeline payload 'results' for query '{query_id}' must be a dict, got {type(query_results)}"
            )

        runtime_ms = query_payload["runtime_milliseconds"]
        if not isinstance(runtime_ms, (int, float)):
            raise ValueError(
                f"Pipeline payload 'runtime_milliseconds' for query '{query_id}' must be a number, "
                f"got {type(runtime_ms)}"
            )

        # pytrec_eval "run" format expects {query_id: {doc_id: score}}
        run[query_id] = query_results
        per_query_runtime_milliseconds[query_id] = float(runtime_ms)

    # Create pytrec_eval evaluator
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, set(metrics))

    # Evaluate
    results = evaluator.evaluate(run)

    # Timing summary is derived solely from per-query runtimes (no wall-clock timing here).
    # `track_time` is kept only for the public signature; per-query runtime is required regardless.
    if track_time:
        num_queries = len(query_ids)
        total_time_milliseconds = sum(per_query_runtime_milliseconds.values())
        results["_timing"] = {
            "total_retrieval_time_milliseconds": total_time_milliseconds,
            "average_time_per_query_milliseconds": total_time_milliseconds / num_queries if num_queries > 0 else 0.0,
            "num_queries": num_queries,
            "queries_per_second": num_queries / (total_time_milliseconds / 1000)
            if total_time_milliseconds > 0
            else 0.0,
            "per_query_retrieval_time_milliseconds": per_query_runtime_milliseconds,
        }

    return results


def aggregate_results(
    results: Dict[str, Any], query_languages: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Calculate aggregate statistics across all queries.

    If query_languages is provided, also computes per-language aggregates.

    Args:
        results: Per-query evaluation results from evaluate_retrieval()
        query_languages: Optional mapping of query_id to language

    Returns:
        Dictionary of aggregated metrics. If query_languages is provided:
        {
            'overall': {'ndcg_cut_10': 0.85, ...},
            'by_language': {
                'english': {'ndcg_cut_10': 0.87, ...},
                'french': {'ndcg_cut_10': 0.82, ...},
            },
            'timing': {...}  # if timing info present
        }
        Otherwise, just returns flat aggregated metrics.
    """
    if not results:
        return {}

    # Extract timing information if present
    timing_info = results.pop("_timing", None)

    if not results:
        # Only timing info was present
        return {"timing": timing_info} if timing_info else {}

    # Get all metric names from first query
    metric_names = list(next(iter(results.values())).keys())

    # If no language splitting requested, return simple aggregation
    if query_languages is None:
        aggregated = {}
        for metric in metric_names:
            scores = [results[qid][metric] for qid in results]
            aggregated[metric] = sum(scores) / len(scores)

        # Add timing information back if it was present
        if timing_info:
            aggregated.update(timing_info)

        return aggregated

    # Split results by language
    results_by_language = defaultdict(dict)
    for query_id, query_results in results.items():
        lang = query_languages.get(query_id, "unknown")
        results_by_language[lang][query_id] = query_results

    # Compute overall aggregates
    overall_aggregated = {}
    for metric in metric_names:
        scores = [results[qid][metric] for qid in results]
        overall_aggregated[metric] = sum(scores) / len(scores)

    # Compute per-language aggregates
    by_language_aggregated = {}
    for lang, lang_results in results_by_language.items():
        lang_aggregated = {}
        for metric in metric_names:
            scores = [lang_results[qid][metric] for qid in lang_results]
            lang_aggregated[metric] = sum(scores) / len(scores)
        lang_aggregated["num_queries"] = len(lang_results)
        by_language_aggregated[lang] = lang_aggregated

    # Build final result structure
    final_result = {
        "overall": overall_aggregated,
        "by_language": by_language_aggregated,
    }

    # Add timing information
    if timing_info:
        final_result["timing"] = timing_info

    return final_result
