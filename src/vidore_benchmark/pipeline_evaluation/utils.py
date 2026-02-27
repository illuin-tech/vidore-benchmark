#!/usr/bin/env python3
"""
Multi-dataset evaluation script.

Evaluates a pipeline on all available vidore v3 datasets and displays
a summary table of results. Useful for benchmarking pipelines across
the entire vidore v3 benchmark suite.

By default, uses RandomPipeline for testing the multi-dataset functionality.
"""

import traceback
from typing import Dict, List, Tuple

from vidore_benchmark.pipeline_evaluation import (
    aggregate_results,
    evaluate_retrieval,
    load_vidore_dataset,
)


def evaluate_single_dataset(dataset_name: str, pipeline, language: str = None) -> Tuple[float, int, str]:
    """
    Evaluate pipeline on a single dataset.

    Args:
        dataset_name: Name of the vidore v3 dataset
        pipeline: Pipeline instance to evaluate
        language: Optional language filter

    Returns:
        Tuple of (ndcg_score, num_queries, error_message)
        If evaluation fails, ndcg_score will be None and error_message will be set
    """
    try:
        # Load dataset
        query_ids, queries, corpus_ids, corpus_images, corpus_texts, qrels, query_languages = load_vidore_dataset(
            dataset_name, language=language
        )

        # Run evaluation
        results = evaluate_retrieval(
            pipeline=pipeline,
            query_ids=query_ids,
            queries=queries,
            corpus_ids=corpus_ids,
            corpus_images=corpus_images,
            corpus_texts=corpus_texts,
            qrels=qrels,
            metrics=["ndcg_cut_10"],
        )

        # Aggregate results
        aggregated = aggregate_results(results, query_languages=query_languages)
        ndcg_score = aggregated["ndcg_cut_10"]
        num_queries = len(results)

        return ndcg_score, num_queries, None

    except Exception as e:
        # Get full error details including traceback
        error_details = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        return None, 0, error_details


def print_results_table(results: List[Dict]):
    """
    Print evaluation results in a formatted table.

    Args:
        results: List of dicts with keys: dataset_name, ndcg, num_queries, error
    """
    print("\n" + "=" * 80)
    print("MULTI-DATASET EVALUATION RESULTS")
    print("=" * 80)
    print(f"{'Dataset':<40} {'NDCG@10':>10} {'Queries':>10} {'Status':>15}")
    print("-" * 80)

    successful_results = []
    failed_results = []

    for result in results:
        dataset_name = result["dataset_name"]
        # Shorten dataset name for display (remove 'vidore/' prefix)
        display_name = dataset_name.replace("vidore/", "")

        if result["error"]:
            print(f"{display_name:<40} {'N/A':>10} {'N/A':>10} {'FAILED':>15}")
            failed_results.append(result)
        else:
            ndcg = result["ndcg"]
            num_queries = result["num_queries"]
            print(f"{display_name:<40} {ndcg:>10.4f} {num_queries:>10} {'SUCCESS':>15}")
            successful_results.append(result)

    print("-" * 80)

    # Summary statistics
    if successful_results:
        avg_ndcg = sum(r["ndcg"] for r in successful_results) / len(successful_results)
        total_queries = sum(r["num_queries"] for r in successful_results)
        print(f"{'AVERAGE':<40} {avg_ndcg:>10.4f} {total_queries:>10} ({len(successful_results)}/{len(results)})")
    else:
        print(f"{'AVERAGE':<40} {'N/A':>10} {'N/A':>10} (0/{len(results)})")

    print("=" * 80)

    # Print error details if any
    if failed_results:
        print("\nERROR DETAILS:")
        print("-" * 80)
        for result in failed_results:
            dataset_name = result["dataset_name"].replace("vidore/", "")
            error = result["error"]
            print(f"\n{dataset_name}:")
            print(error)
        print("-" * 80)
