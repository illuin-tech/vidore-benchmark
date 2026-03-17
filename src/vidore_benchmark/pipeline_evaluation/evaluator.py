"""
Core evaluation orchestration using pytrec_eval.
"""

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import pytrec_eval

from vidore_benchmark.pipeline_evaluation.base_pipeline import BasePipeline


def evaluate_retrieval(
    pipeline: BasePipeline,
    query_ids: List[str],
    queries: List[str],
    corpus_ids: List[str],
    corpus_images: List[Any],
    corpus_texts: List[str],
    qrels: Dict[str, Dict[str, int]],
    dataset_name: Optional[str] = None,
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
        corpus_images: List of corpus images (PIL.Image objects)
        corpus_texts: List of corpus texts (markdown strings)
        qrels: Ground truth relevance judgments in pytrec_eval format
               {query_id: {doc_id: relevance_score}}
        dataset_name: Dataset name,
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

    # Call the pipeline's method to get retrieval results
    # Indexing step
    start_time_indexing = time.time()
    pipeline.index(
        corpus_ids=corpus_ids, corpus_images=corpus_images, corpus_texts=corpus_texts, dataset_name=dataset_name
    )
    indexing_time = time.time() - start_time_indexing

    # Avoid tracking indexing time if no other thing is done than storing the corpus
    if indexing_time < 1e-5:
        indexing_time = 0.0

    # Search step
    start_time_search = time.time()
    result = pipeline.search(query_ids=query_ids, queries=queries)
    search_time = time.time() - start_time_search

    total_time = indexing_time + search_time

    if isinstance(result, tuple):
        run, infos = result
    else:
        run, infos = result, None

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
        num_corpus = len(corpus_ids)
        results["_timing"] = {
            "total_retrieval_time_milliseconds": total_time * 1000,
            "indexing_time_milliseconds": indexing_time * 1000,
            "search_time_milliseconds": search_time * 1000,
            "num_queries": num_queries,
            "num_corpus": num_corpus,
            "indexing_throughput_ms_per_doc": (indexing_time * 1000) / num_corpus if num_corpus > 0 else None,
            "search_throughput_ms_per_query": (search_time * 1000) / num_queries if num_queries > 0 else None,
        }
    # Add additional pipeline infos if provided
    if infos is not None:
        results["_infos"] = infos

    return results


def aggregate_results(
    results: Dict[str, Dict[str, float]], query_languages: Optional[Dict[str, str]] = None
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
            'infos': {...}  # if pipeline infos present
        }
        Otherwise, just returns flat aggregated metrics.
    """
    if not results:
        return {}

    # Extract timing information if present
    timing_info = results.pop("_timing", None)

    additional_infos = results.pop("_infos", None)

    if not results:
        # Only timing info was present
        final_result = {}
        if timing_info:
            final_result["timing"] = timing_info
        if additional_infos:
            final_result["infos"] = additional_infos
        return final_result

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
        if additional_infos:
            aggregated.update(additional_infos)

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
    if additional_infos:
        final_result["infos"] = additional_infos
    return final_result
