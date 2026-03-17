"""
CLI commands for evaluating pipelines on ViDoRe v3 datasets.

This CLI allows you to evaluate custom retrieval pipelines implemented as Python classes
that inherit from BasePipeline. You can evaluate built-in pipelines (Random, FileBased)
or pipelines you implement yourself.
"""

import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

from vidore_benchmark.pipeline_evaluation import (
    BasePipeline,
    aggregate_results,
    evaluate_retrieval,
    get_available_datasets,
    load_vidore_dataset,
    print_dataset_info,
)
from vidore_benchmark.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="""
    CLI for evaluating abstract pipelines on ViDoRe v3 datasets.

    Evaluate custom retrieval pipelines that inherit from BasePipeline.
    Supports built-in pipelines (random, file-based) and custom Python implementations.
    """,
    no_args_is_help=True,
)


@app.callback()
def main(log_level: Annotated[str, typer.Option("--log", help="Logging level")] = "warning"):
    """Initialize logging configuration."""
    setup_logging(log_level)
    logger.info("Logging level set to `%s`", log_level)


def _load_pipeline_from_module(module_path: str, class_name: str, **kwargs) -> BasePipeline:
    """
    Dynamically load a pipeline class from a Python file.

    Args:
        module_path: Path to the Python file containing the pipeline class
        class_name: Name of the pipeline class to instantiate
        **kwargs: Arguments to pass to the pipeline constructor

    Returns:
        Instantiated pipeline object
    """
    module_path = Path(module_path).resolve()

    if not module_path.exists():
        raise FileNotFoundError(f"Module file not found: {module_path}")

    # Load the module
    spec = importlib.util.spec_from_file_location("custom_pipeline", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["custom_pipeline"] = module
    spec.loader.exec_module(module)

    # Get the class
    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in {module_path}")

    pipeline_class = getattr(module, class_name)

    # Verify it's a BasePipeline subclass
    if not issubclass(pipeline_class, BasePipeline):
        raise TypeError(f"Class '{class_name}' must inherit from BasePipeline")

    # Instantiate the pipeline
    return pipeline_class(**kwargs)


@app.command()
def list_datasets():
    """
    List all available ViDoRe v3 datasets.

    Example:
        vidore-benchmark pipeline list-datasets
    """
    datasets = get_available_datasets()

    print("\n" + "=" * 70)
    print("Available ViDoRe v3 Datasets")
    print("=" * 70)
    for i, dataset_name in enumerate(datasets, 1):
        print(f"{i:2d}. {dataset_name}")
    print("=" * 70)
    print(f"\nTotal: {len(datasets)} datasets\n")


@app.command()
def evaluate(
    dataset_name: Annotated[str, typer.Option(help="Name of ViDoRe v3 dataset (e.g., 'vidore/vidore_v3_hr')")],
    pipeline_type: Annotated[
        Optional[str], typer.Option(help="Built-in pipeline type: 'random' or 'file-based'")
    ] = None,
    module_path: Annotated[Optional[str], typer.Option(help="Path to Python file with custom pipeline class")] = None,
    class_name: Annotated[
        Optional[str], typer.Option(help="Name of pipeline class (required with --module-path)")
    ] = None,
    pipeline_args: Annotated[
        Optional[str], typer.Option(help="JSON string of arguments to pass to pipeline constructor")
    ] = None,
    output_file: Annotated[
        Optional[str], typer.Option(help="Path to save evaluation results (default: auto-generated)")
    ] = None,
    language: Annotated[
        Optional[str], typer.Option(help="Language filter for queries (e.g., 'english', 'french')")
    ] = None,
    split: Annotated[str, typer.Option(help="Dataset split to evaluate on")] = "test",
    show_dataset_info: Annotated[bool, typer.Option(help="Show dataset information before evaluation")] = True,
):
    """
    Evaluate a pipeline on a ViDoRe v3 dataset.

    You can either use a built-in pipeline (--pipeline-type) or a custom pipeline (--module-path + --class-name).

    Examples:
        # Evaluate random baseline
        vidore-benchmark pipeline evaluate \\
            --dataset-name vidore/vidore_v3_hr \\
            --pipeline-type random \\
            --pipeline-args '{"seed": 42, "top_k": 10}'

        # Evaluate custom pipeline
        vidore-benchmark pipeline evaluate \\
            --dataset-name vidore/vidore_v3_hr \\
            --module-path my_pipeline.py \\
            --class-name MyCustomPipeline \\
            --pipeline-args '{"model_name": "my-model"}'

        # Evaluate file-based pipeline
        vidore-benchmark pipeline evaluate \\
            --dataset-name vidore/vidore_v3_hr \\
            --pipeline-type file-based \\
            --pipeline-args '{"run_file_path": "results.json"}'
    """
    # Validate input
    if pipeline_type is None and module_path is None:
        print("\n❌ Error: Must specify either --pipeline-type or --module-path\n")
        raise typer.Exit(code=1)

    if pipeline_type is not None and module_path is not None:
        print("\n❌ Error: Cannot specify both --pipeline-type and --module-path\n")
        raise typer.Exit(code=1)

    if module_path is not None and class_name is None:
        print("\n❌ Error: --class-name is required when using --module-path\n")
        raise typer.Exit(code=1)

    # Parse pipeline arguments
    kwargs = {}
    if pipeline_args:
        try:
            kwargs = json.loads(pipeline_args)
        except json.JSONDecodeError as e:
            print(f"\n❌ Error parsing pipeline arguments: {e}\n")
            raise typer.Exit(code=1)

    # Load the dataset
    logger.info(f"Loading dataset: {dataset_name}")
    try:
        query_ids, queries, corpus_ids, corpus_images, corpus_texts, qrels, query_languages = load_vidore_dataset(
            dataset_name=dataset_name, split=split, language=language
        )
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}\n")
        raise typer.Exit(code=1)

    # Show dataset info if requested
    if show_dataset_info:
        print_dataset_info(dataset_name, query_ids, queries, corpus_ids, corpus_images, corpus_texts, qrels)

    # Load the pipeline
    try:
        if pipeline_type == "random":
            module_path = "pipeline_implementations/random_pipeline.py"
            pipeline = _load_pipeline_from_module(module_path, "RandomPipeline", **kwargs)
        elif pipeline_type == "file-based":
            module_path = "pipeline_implementations/file_based_pipeline.py"
            pipeline = _load_pipeline_from_module(module_path, "FileBasedPipeline", **kwargs)
        elif module_path:
            logger.info(f"Loading custom pipeline from {module_path}")
            pipeline = _load_pipeline_from_module(module_path, class_name, **kwargs)
        else:
            print(f"\n❌ Error: Unknown pipeline type: {pipeline_type}\n")
            raise typer.Exit(code=1)
    except Exception as e:
        print(f"\n❌ Error loading pipeline: {e}\n")
        raise typer.Exit(code=1)

    # Run evaluation
    print("\nRunning evaluation...")
    try:
        results = evaluate_retrieval(
            pipeline=pipeline,
            query_ids=query_ids,
            queries=queries,
            corpus_ids=corpus_ids,
            corpus_images=corpus_images,
            corpus_texts=corpus_texts,
            qrels=qrels,
            dataset_name=dataset_name,
            metrics=[
                "ndcg_cut_1",
                "ndcg_cut_5",
                "ndcg_cut_10",
                "ndcg_cut_20",
                "ndcg_cut_100",
                "recall_1",
                "recall_5",
                "recall_10",
                "recall_20",
                "recall_50",
                "recall_100",
                "P_1",
                "P_5",
                "P_10",
                "P_20",
                "map",
                "map_cut_1",
                "map_cut_10",
                "map_cut_100",
                "recip_rank",
            ],
        )
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}\n")
        raise typer.Exit(code=1)

    # Calculate aggregates (with language splitting if applicable)
    aggregated = aggregate_results(results, query_languages)

    # Display results
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)

    # Check if we have language-split results
    if "overall" in aggregated and "by_language" in aggregated:
        # Display overall metrics
        overall_metrics = aggregated["overall"]
        timing_info = aggregated.get("timing", {})

        print("\n--- Overall Results ---")
        key_metrics = ["ndcg_cut_10", "ndcg_cut_5", "recall_10", "recall_5", "map", "recip_rank"]
        for metric in key_metrics:
            if metric in overall_metrics:
                print(f"  {metric:25s}: {overall_metrics[metric]:.4f}")

        # Display per-language breakdown
        print("\n--- Results by Language ---")
        for lang, lang_metrics in aggregated["by_language"].items():
            num_queries = lang_metrics.get("num_queries", 0)
            print(f"\n{lang.capitalize()} ({num_queries} queries):")
            for metric in key_metrics:
                if metric in lang_metrics:
                    print(f"  {metric:25s}: {lang_metrics[metric]:.4f}")

        # Calculate how many additional metrics were saved
        all_metrics_count = len(overall_metrics)
        displayed_count = len([m for m in key_metrics if m in overall_metrics])
        other_count = all_metrics_count - displayed_count
        if other_count > 0:
            print(f"\n({other_count} additional metrics saved to file)")

        # Display timing metrics
        if timing_info:
            print("\n--- Timing Metrics ---")
            for metric, value in timing_info.items():
                if "milliseconds" in metric:
                    print(f"  {metric:40s}: {value:.2f}ms")
                else:
                    print(f"  {metric:40s}: {value:.2f}")
    else:
        # Original flat display (backward compatibility)
        timing_metrics = {
            k: v for k, v in aggregated.items() if k.startswith(("total_", "average_", "queries_", "num_"))
        }
        retrieval_metrics = {k: v for k, v in aggregated.items() if k not in timing_metrics}

        # Display key retrieval metrics only
        key_metrics = ["ndcg_cut_10", "ndcg_cut_5", "recall_10", "recall_5", "map", "recip_rank"]
        if retrieval_metrics:
            print("\nKey Retrieval Metrics:")
            for metric in key_metrics:
                if metric in retrieval_metrics:
                    print(f"  {metric:25s}: {retrieval_metrics[metric]:.4f}")

            other_count = len(retrieval_metrics) - len([m for m in key_metrics if m in retrieval_metrics])
            if other_count > 0:
                print(f"\n  ({other_count} additional metrics saved to file)")

        # Display timing metrics
        if timing_metrics:
            print("\nTiming Metrics:")
            for metric, value in timing_metrics.items():
                if "milliseconds" in metric:
                    print(f"  {metric:40s}: {value:.2f}ms")
                else:
                    print(f"  {metric:40s}: {value:.2f}")

    print("=" * 70 + "\n")

    # Save results
    if output_file is None:
        # Auto-generate output filename
        if pipeline_type:
            pipeline_name = pipeline_type
        else:
            pipeline_name = class_name
        dataset_short = dataset_name.split("/")[-1]
        if not os.path.exists("results"):
            os.makedirs("results")
        if not os.path.exists("results/metrics"):
            os.makedirs("results/metrics")
        if not os.path.exists(f"results/metrics/{pipeline_name}"):
            os.makedirs(f"results/metrics/{pipeline_name}")
        output_file = f"results/metrics/{pipeline_name}/{dataset_short}.json"

    output_path = Path(output_file)
    output_data = {
        "dataset": dataset_name,
        "split": split,
        "language": language,
        "pipeline_type": pipeline_type,
        "module_path": str(module_path) if module_path else None,
        "class_name": class_name,
        "pipeline_args": kwargs,
        "aggregated_metrics": aggregated,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Results saved to: {output_path}\n")
    print(
        "To submit your pipeline to the leaderboard, please also write a description file following the template in"
        " 'results/pipeline_descriptions/' and submit a pull request to the GitHub repository."
    )


@app.command()
def evaluate_all(
    pipeline_type: Annotated[
        Optional[str], typer.Option(help="Built-in pipeline type: 'random' or 'file-based'")
    ] = None,
    module_path: Annotated[Optional[str], typer.Option(help="Path to Python file with custom pipeline class")] = None,
    class_name: Annotated[
        Optional[str], typer.Option(help="Name of pipeline class (required with --module-path)")
    ] = None,
    pipeline_args: Annotated[
        Optional[str], typer.Option(help="JSON string of arguments to pass to pipeline constructor")
    ] = None,
    output_dir: Annotated[str, typer.Option(help="Directory to save evaluation results")] = "results/",
    language: Annotated[
        Optional[str], typer.Option(help="Language filter for queries (e.g., 'english', 'french')")
    ] = None,
    split: Annotated[str, typer.Option(help="Dataset split to evaluate on")] = "test",
):
    """
    Evaluate a pipeline on all ViDoRe v3 datasets.

    Examples:
        # Evaluate random baseline on all datasets
        vidore-benchmark pipeline evaluate-all \\
            --pipeline-type random \\
            --pipeline-args '{"seed": 42}'

        # Evaluate custom pipeline on all datasets
        vidore-benchmark pipeline evaluate-all \\
            --module-path my_pipeline.py \\
            --class-name MyPipeline \\
            --output-dir my_results/
    """
    # Validate input
    if pipeline_type is None and module_path is None:
        print("\n❌ Error: Must specify either --pipeline-type or --module-path\n")
        raise typer.Exit(code=1)

    if pipeline_type is not None and module_path is not None:
        print("\n❌ Error: Cannot specify both --pipeline-type and --module-path\n")
        raise typer.Exit(code=1)

    if module_path is not None and class_name is None:
        print("\n❌ Error: --class-name is required when using --module-path\n")
        raise typer.Exit(code=1)

    # Parse pipeline arguments
    kwargs = {}
    if pipeline_args:
        try:
            kwargs = json.loads(pipeline_args)
        except json.JSONDecodeError as e:
            print(f"\n❌ Error parsing pipeline arguments: {e}\n")
            raise typer.Exit(code=1)

    datasets = get_available_datasets()

    print(f"\n{'=' * 70}")
    print(f"Evaluating on {len(datasets)} datasets")
    print(f"{'=' * 70}\n")

    all_results = {}

    for i, dataset_name in enumerate(datasets, 1):
        print(f"\n[{i}/{len(datasets)}] Evaluating: {dataset_name}")
        print("-" * 70)

        try:
            # Load dataset
            query_ids, queries, corpus_ids, corpus_images, corpus_texts, qrels, query_languages = load_vidore_dataset(
                dataset_name=dataset_name, split=split, language=language
            )

            # Load pipeline (create fresh instance for each dataset)
            if pipeline_type == "random":
                module_path = "pipeline_implementations/random_pipeline.py"
                pipeline = _load_pipeline_from_module(module_path, "RandomPipeline", **kwargs)
            elif pipeline_type == "file-based":
                module_path = "pipeline_implementations/file_based_pipeline.py"
                pipeline = _load_pipeline_from_module(module_path, "FileBasedPipeline", **kwargs)
            elif module_path:
                pipeline = _load_pipeline_from_module(module_path, class_name, **kwargs)
            else:
                print(f"\n❌ Error: Unknown pipeline type: {pipeline_type}\n")
                raise typer.Exit(code=1)

            # Evaluate
            results = evaluate_retrieval(
                pipeline=pipeline,
                query_ids=query_ids,
                queries=queries,
                corpus_ids=corpus_ids,
                corpus_images=corpus_images,
                corpus_texts=corpus_texts,
                qrels=qrels,
                dataset_name=dataset_name,
                metrics=[
                    "ndcg_cut_1",
                    "ndcg_cut_5",
                    "ndcg_cut_10",
                    "ndcg_cut_20",
                    "ndcg_cut_100",
                    "recall_1",
                    "recall_5",
                    "recall_10",
                    "recall_20",
                    "recall_50",
                    "recall_100",
                    "P_1",
                    "P_5",
                    "P_10",
                    "P_20",
                    "map",
                    "map_cut_1",
                    "map_cut_10",
                    "map_cut_100",
                    "recip_rank",
                ],
            )

            # Calculate aggregates (with language splitting)
            aggregated = aggregate_results(results, query_languages)

            # Store results
            all_results[dataset_name] = {
                "aggregated_metrics": aggregated,
                "per_query_metrics": results,
            }

            # Save individual dataset results
            if pipeline_type:
                pipeline_name = pipeline_type
            else:
                pipeline_name = class_name

            dataset_short = dataset_name.split("/")[-1]
            result_dir = Path(output_dir) / pipeline_name
            result_dir.mkdir(parents=True, exist_ok=True)

            result_file = result_dir / f"{dataset_short}.json"

            output_data = {
                "dataset": dataset_name,
                "split": split,
                "language": language,
                "pipeline_type": pipeline_type,
                "module_path": str(module_path) if module_path else None,
                "class_name": class_name,
                "pipeline_args": kwargs,
                "aggregated_metrics": aggregated,
            }

            with open(result_file, "w") as f:
                json.dump(output_data, f, indent=2)

            # Display summary with timing
            # Handle both language-split and flat result formats
            if "overall" in aggregated:
                ndcg_score = aggregated["overall"].get("ndcg_cut_10", 0.0)
                timing = aggregated.get("timing", {})
                avg_time = timing.get("average_time_per_query_milliseconds", 0.0)
                print(f"NDCG@10: {ndcg_score:.4f} | Avg time/query: {avg_time:.2f}ms")

                # Show language breakdown if available
                if "by_language" in aggregated and len(aggregated["by_language"]) > 1:
                    for lang, lang_metrics in aggregated["by_language"].items():
                        lang_ndcg = lang_metrics.get("ndcg_cut_10", 0.0)
                        num_queries = lang_metrics.get("num_queries", 0)
                        print(f"  - {lang}: {lang_ndcg:.4f} ({num_queries} queries)")
            else:
                ndcg_score = aggregated.get("ndcg_cut_10", 0.0)
                avg_time = aggregated.get("average_time_per_query_milliseconds", 0.0)
                print(f"NDCG@10: {ndcg_score:.4f} | Avg time/query: {avg_time:.2f}ms")

            print(f"  Saved to: {result_file}")

        except Exception as e:
            print(f"❌ Error evaluating {dataset_name}: {e}")
            all_results[dataset_name] = {"error": str(e)}

    # Print summary
    print(f"\n{'=' * 70}")
    print("✓ Evaluation complete!")

    if pipeline_type:
        pipeline_name = pipeline_type
    else:
        pipeline_name = class_name

    result_dir = Path(output_dir) / pipeline_name
    print(f"✓ Results saved to: {result_dir}/")

    # Display aggregate summary
    successful_datasets = [k for k, v in all_results.items() if "error" not in v]
    if successful_datasets:
        print(f"\nSuccessfully evaluated {len(successful_datasets)}/{len(datasets)} datasets")
        print("\nAverage NDCG@10 across datasets:")

        # Extract NDCG scores handling both flat and nested formats
        ndcg_scores = []
        for ds in successful_datasets:
            aggregated = all_results[ds]["aggregated_metrics"]
            if "overall" in aggregated:
                ndcg_scores.append(aggregated["overall"].get("ndcg_cut_10", 0.0))
            else:
                ndcg_scores.append(aggregated.get("ndcg_cut_10", 0.0))

        avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
        print(f"  {avg_ndcg:.4f}")

    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    app()
