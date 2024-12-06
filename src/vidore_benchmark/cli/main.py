import logging
import os
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Dict, List, Optional, cast

import huggingface_hub
import typer
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from transformers import set_seed

from vidore_benchmark.compression.token_pooling import BaseEmbeddingPooler, HierarchicalEmbeddingPooler
from vidore_benchmark.evaluation.interfaces import MetadataModel, ViDoReBenchmarkResults
from vidore_benchmark.evaluation.vidore_evaluators import ViDoReEvaluatorBEIR, ViDoReEvaluatorQA
from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
from vidore_benchmark.retrievers.registry_utils import load_vision_retriever_from_registry
from vidore_benchmark.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

load_dotenv(override=True)
set_seed(42)

app = typer.Typer(
    help="""
    CLI for evaluating vision retrievers.
    Can be used to evaluate on the ViDoRe benchmark and to generate metrics for the ViDoRe leaderboard.
    """,
    no_args_is_help=True,
)


def _sanitize_model_id(model_class: str, pretrained_model_name_or_path: Optional[str] = None) -> str:
    """
    Return sanitized model ID for properly saving metrics as files.
    """
    model_id = pretrained_model_name_or_path if pretrained_model_name_or_path is not None else model_class
    model_id = model_id.replace("/", "_")
    return model_id


def _get_metrics_from_vidore_evaluator(
    vision_retriever: BaseVisionRetriever,
    embedding_pooler: Optional[BaseEmbeddingPooler],
    dataset_name: str,
    dataset_format: str,
    split: str,
    batch_query: int,
    batch_passage: int,
    batch_score: Optional[int],
    dataloader_prebatch_size: Optional[int],
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Rooter function to get metrics from the ViDoRe evaluator depending on the dataset format.

    Args:
        vision_retriever (BaseVisionRetriever)
        embedding_pooler (Optional[BaseEmbeddingPooler])
        dataset_name (str): Dataset name
        dataset_format (str): Dataset format
        split (str): Dataset split
        batch_query (int): Batch size for query embedding inference
        batch_passage (int): Batch size for passages embedding inference
        batch_score (Optional[int]): Batch size for score computation
        dataloader_prebatch_size (Optional[int]): Prebatch size for the dataloader

    Returns:
        Dict[str, Dict[str, Optional[float]]]: Metrics per dataset.
            Example: {"dataset_name": {"ndcg_at_5": 0.5, "ndcg_at_10": 0.6}}
    """
    if dataset_format == "qa":
        vidore_evaluator = ViDoReEvaluatorQA(
            vision_retriever=vision_retriever,
            embedding_pooler=embedding_pooler,
        )
        ds = load_dataset(dataset_name, split=split)
        metrics = {
            dataset_name: vidore_evaluator.evaluate_dataset(
                ds=ds,
                ds_format=dataset_format,
                batch_query=batch_query,
                batch_passage=batch_passage,
                batch_score=batch_score,
                dataloader_prebatch_size=dataloader_prebatch_size,
            )
        }

    elif dataset_format == "beir":
        vidore_evaluator = ViDoReEvaluatorBEIR(
            vision_retriever=vision_retriever,
            embedding_pooler=embedding_pooler,
        )
        ds = {
            "corpus": cast(Dataset, load_dataset(dataset_name, name="corpus", split=split)),
            "queries": cast(Dataset, load_dataset(dataset_name, name="queries", split=split)),
            "qrels": cast(Dataset, load_dataset(dataset_name, name="qrels", split=split)),
        }
        metrics = {
            dataset_name: vidore_evaluator.evaluate_dataset(
                ds=ds,
                ds_format=dataset_format,
                batch_query=batch_query,
                batch_passage=batch_passage,
                batch_score=batch_score,
                dataloader_prebatch_size=dataloader_prebatch_size,
            )
        }
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}")

    return metrics


@app.callback()
def main(log_level: Annotated[str, typer.Option("--log", help="Logging level")] = "warning"):
    setup_logging(log_level)
    logger.info("Logging level set to `%s`", log_level)


@app.command()
def evaluate_retriever(
    model_class: Annotated[str, typer.Option(help="Model class")],
    pretrained_model_name_or_path: Annotated[
        Optional[str],
        typer.Option("--model-name", help="If Hf model, passed to the `model.from_pretrained` method."),
    ] = None,
    dataset_name: Annotated[Optional[str], typer.Option(help="Hf Hub dataset name.")] = None,
    collection_name: Annotated[
        Optional[str],
        typer.Option(help="Dataset collection to use for evaluation. Can be a Hf collection id or a local dirpath."),
    ] = None,
    dataset_format: Annotated[str, typer.Option(help='Dataset format ("qa" or "beir") to use for evaluation')] = "qa",
    split: Annotated[str, typer.Option(help="Dataset split")] = "test",
    batch_query: Annotated[int, typer.Option(help="Batch size for query embedding inference")] = 8,
    batch_passage: Annotated[int, typer.Option(help="Batch size for passages embedding inference")] = 8,
    batch_score: Annotated[Optional[int], typer.Option(help="Batch size for score computation")] = 16,
    dataloader_prebatch_size: Annotated[Optional[int], typer.Option(help="Prebatch size for the dataloader")] = None,
    use_token_pooling: Annotated[
        bool, typer.Option(help="Whether to use token pooling for passage embeddings")
    ] = False,
    pool_factor: Annotated[int, typer.Option(help="Pooling factor for hierarchical token pooling")] = 3,
    output_dir: Annotated[str, typer.Option(help="Directory where to save the metrics")] = "outputs",
):
    """
    Evaluate the vision retriever on the given dataset or dataset collection.
    The metrics are saved to a JSON file and follow the `ViDoReBenchmarkResults` schema.
    """

    logging.info(f"Evaluating retriever `{model_class}`")

    # Sanity check
    if dataset_name is None and collection_name is None:
        raise ValueError("Please provide a dataset name or collection name")
    elif dataset_name is not None and collection_name is not None:
        raise ValueError("Please provide only one of dataset name or collection name")

    # Create the vision retriever
    retriever = load_vision_retriever_from_registry(
        model_class,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
    )

    # Sanitize the model ID to use as a filename
    model_id = _sanitize_model_id(model_class, pretrained_model_name_or_path)

    # Get the pooling strategy
    embedding_pooler = HierarchicalEmbeddingPooler(pool_factor) if use_token_pooling else None

    # Create the output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load the dataset(s) and evaluate
    if dataset_name is None and collection_name is None:
        raise ValueError("Please provide a dataset name or collection name.")

    elif dataset_name is not None:
        metrics = _get_metrics_from_vidore_evaluator(
            vision_retriever=retriever,
            embedding_pooler=embedding_pooler,
            dataset_name=dataset_name,
            dataset_format=dataset_format,
            split=split,
            batch_query=batch_query,
            batch_passage=batch_passage,
            batch_score=batch_score,
            dataloader_prebatch_size=dataloader_prebatch_size,
        )

        if use_token_pooling:
            savepath = output_path / f"{model_id}_metrics_pool_factor_{pool_factor}.json"
        else:
            savepath = output_path / f"{model_id}_metrics.json"

        print(f"nDCG@5 for {model_id} on {dataset_name}: {metrics[dataset_name]['ndcg_at_5']}")

        results = ViDoReBenchmarkResults(
            metadata=MetadataModel(
                timestamp=datetime.now(),
                vidore_benchmark_version=version("vidore_benchmark"),
            ),
            metrics={dataset_name: metrics[dataset_name]},
        )

        with open(str(savepath), "w", encoding="utf-8") as f:
            f.write(results.model_dump_json(indent=4))

        print(f"Benchmark results saved to `{savepath}`")

    elif collection_name is not None:
        if os.path.isdir(collection_name):
            print(f"Loading datasets from local directory: `{collection_name}`")
            dataset_names = os.listdir(collection_name)
            dataset_names = [os.path.join(collection_name, dataset) for dataset in dataset_names]
        else:
            print(f"Loading datasets from the Hf Hub collection: {collection_name}")
            collection = huggingface_hub.get_collection(collection_name)
            dataset_names = [dataset_item.item_id for dataset_item in collection.items]

        # Placeholder for all metrics
        metrics_all: Dict[str, Dict[str, Optional[float]]] = {}
        results_all: List[ViDoReBenchmarkResults] = []

        savedir = output_path / model_id.replace("/", "_")
        savedir.mkdir(parents=True, exist_ok=True)

        for dataset_name in dataset_names:
            print(f"\n---------------------------\nEvaluating {dataset_name}")

            metrics = _get_metrics_from_vidore_evaluator(
                vision_retriever=retriever,
                embedding_pooler=embedding_pooler,
                dataset_name=dataset_name,
                dataset_format=dataset_format,
                split=split,
                batch_query=batch_query,
                batch_passage=batch_passage,
                batch_score=batch_score,
                dataloader_prebatch_size=dataloader_prebatch_size,
            )

            metrics_all.update(metrics)

            # Sanitize the dataset item to use as a filename
            dataset_item_id = dataset_name.replace("/", "_")

            if use_token_pooling:
                savepath = savedir / f"{dataset_item_id}_metrics_pool_factor_{pool_factor}.json"
            else:
                savepath = savedir / f"{dataset_item_id}_metrics.json"

            print(f"nDCG@5 for {model_id} on {dataset_name}: {metrics[dataset_name]['ndcg_at_5']}")

            results = ViDoReBenchmarkResults(
                metadata=MetadataModel(
                    timestamp=datetime.now(),
                    vidore_benchmark_version=version("vidore_benchmark"),
                ),
                metrics={dataset_name: metrics[dataset_name]},
            )
            results_all.append(results)

            with open(str(savepath), "w", encoding="utf-8") as f:
                f.write(results.model_dump_json(indent=4))

            print(f"Benchmark results saved to `{savepath}`")

        if use_token_pooling:
            savepath_all = output_path / f"{model_id}_all_metrics_pool_factor_{pool_factor}.json"
        else:
            savepath_all = output_path / f"{model_id}_all_metrics.json"

        results_merged = ViDoReBenchmarkResults.merge(results_all)

        with open(str(savepath_all), "w", encoding="utf-8") as f:
            f.write(results_merged.model_dump_json(indent=4))

        print(f"Concatenated metrics saved to `{savepath_all}`")

    print("Done.")


if __name__ == "__main__":
    app()
