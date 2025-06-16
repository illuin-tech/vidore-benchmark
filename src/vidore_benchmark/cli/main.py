import logging
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Dict, List, Optional, cast

import typer
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import set_seed

from vidore_benchmark.evaluation.interfaces import MetadataModel, ViDoReBenchmarkResults
from vidore_benchmark.evaluation.vidore_evaluators import ViDoReEvaluatorQA
from vidore_benchmark.evaluation.vidore_evaluators.vidore_evaluator_beir import ViDoReEvaluatorBEIR
from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
from vidore_benchmark.retrievers.registry_utils import load_vision_retriever_from_registry
from vidore_benchmark.utils.data_utils import get_datasets_from_collection, get_configs_to_evaluate

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


def _sanitize_model_id(
    model_class: str,
    model_name: Optional[str] = None,
) -> str:
    """
    Return sanitized model ID for properly saving metrics as files.
    """
    model_id = model_class
    if model_name:
        model_id += f"_{model_name}"
    model_id = model_id.replace("/", "_")
    return model_id


def _get_metrics_from_vidore_evaluator(
    vision_retriever: BaseVisionRetriever,
    dataset_name: str,
    dataset_format: str,
    config: Optional[str],
    split: str,
    batch_query: int,
    batch_passage: int,
    batch_score: Optional[int] = None,
    dataloader_prebatch_query: Optional[int] = None,
    dataloader_prebatch_passage: Optional[int] = None,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Rooter function to get metrics from the ViDoRe evaluator depending on the dataset format.
    """
    if dataset_format.lower() == "qa":
        ds = (
            cast(Dataset, load_dataset(dataset_name, config, split=split))
            if config
            else cast(Dataset, load_dataset(dataset_name, split=split))
        )
        vidore_evaluator = ViDoReEvaluatorQA(vision_retriever)
    elif dataset_format.lower() == "beir":
        ds = {
            "corpus": cast(Dataset, load_dataset(dataset_name, name="corpus", split=split)),
            "queries": cast(Dataset, load_dataset(dataset_name, name="queries", split=split)),
            "qrels": cast(Dataset, load_dataset(dataset_name, name="qrels", split=split)),
        }
        vidore_evaluator = ViDoReEvaluatorBEIR(vision_retriever)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}")

    metrics = {
        dataset_name: vidore_evaluator.evaluate_dataset(
            ds=ds,
            ds_format=dataset_format,
            batch_query=batch_query,
            batch_passage=batch_passage,
            batch_score=batch_score,
            dataloader_prebatch_query=dataloader_prebatch_query,
            dataloader_prebatch_passage=dataloader_prebatch_passage,
        )
    }
    return metrics


@app.callback()
def main(log_level: Annotated[str, typer.Option("--log", help="Logging level")] = "warning"):
    setup_logging(log_level)
    logger.info("Logging level set to `%s`", log_level)


@app.command()
def evaluate_retriever(
    model_class: Annotated[str, typer.Option(help="Model class")],
    dataset_format: Annotated[
        str,
        typer.Option(
            help='Dataset format to use for evaluation ("qa" or "beir"). Use "qa" (uses query deduplication) for '
            'ViDoRe Benchmark v1 and "beir" (without query dedup) for ViDoRe Benchmark v2 (not released yet).'
        ),
    ],
    model_name: Annotated[
        Optional[str],
        typer.Option(
            "--model-name",
            help="For Hf transformers-based models, this value is passed to the `model.from_pretrained` method.",
        ),
    ] = None,
    dataset_name: Annotated[Optional[str], typer.Option(help="Hf Hub dataset name.")] = None,
    collection_name: Annotated[
        Optional[str],
        typer.Option(help="Dataset collection to use for evaluation. Can be a Hf collection id or a local dirpath."),
    ] = None,
    languages: Annotated[Optional[str], typer.Option(help="Comma-separated list of languages.")] = None,
    split: Annotated[str, typer.Option(help="Dataset split")] = "test",
    batch_query: Annotated[int, typer.Option(help="Batch size for query embedding inference")] = 4,
    batch_passage: Annotated[int, typer.Option(help="Batch size for passages embedding inference")] = 4,
    batch_score: Annotated[Optional[int], typer.Option(help="Batch size for retrieval score computation")] = 4,
    dataloader_prebatch_query: Annotated[
        Optional[int], typer.Option(help="Dataloader prebatch size for queries")
    ] = None,
    dataloader_prebatch_passage: Annotated[
        Optional[int], typer.Option(help="Dataloader prebatch size for passages")
    ] = None,
    num_workers: Annotated[
        int, typer.Option(help="Number of workers for dataloader in retrievers, when supported")
    ] = 0,
    output_dir: Annotated[str, typer.Option(help="Directory where to save the metrics")] = "outputs",
):
    """
    Evaluate a retriever on a given dataset or dataset collection.
    The MTEB retrieval metrics are saved to a JSON file.
    """

    if dataset_name is None and collection_name is None:
        raise ValueError("Please provide a dataset name or collection name")
    elif dataset_name is not None and collection_name is not None:
        raise ValueError("Please provide only one of dataset name or collection name")

    retriever = load_vision_retriever_from_registry(
        model_class,
        pretrained_model_name_or_path=model_name,
        num_workers=num_workers,
    )
    model_id = _sanitize_model_id(model_class, model_name=model_name)

    dataset_names: List[str] = []
    if dataset_name is not None:
        dataset_names = [dataset_name]
    elif collection_name is not None:
        dataset_names = get_datasets_from_collection(collection_name)

    metrics_all: Dict[str, Dict[str, Optional[float]]] = {}
    results_all: List[ViDoReBenchmarkResults] = []  # same as metrics_all but structured + with metadata

    savedir_root = Path(output_dir)
    savedir_datasets = savedir_root / model_id.replace("/", "_")
    savedir_datasets.mkdir(parents=True, exist_ok=True)

    language_configs = languages.split(",") if languages else []

    for dataset_name in tqdm(dataset_names, desc="Evaluating dataset(s)"):
        print(f"\n---------------------------\n{dataset_name}")
        langs_to_evaluate = get_configs_to_evaluate(
            dataset_name, language_configs
        ) if dataset_format.lower() == "qa" else [None]
        for config in sorted(langs_to_evaluate):
            config_suffix = config or "default"
            prediction_output_path = (
                    savedir_datasets
                    / f"{model_id}_{dataset_name}_{config_suffix}_predictions.json"
            )
            prediction_output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Predictions saved to `{prediction_output_path}`")

            metrics = _get_metrics_from_vidore_evaluator(
                vision_retriever=retriever,
                dataset_name=dataset_name,
                dataset_format=dataset_format,
                config=config_suffix,
                split=split,
                batch_query=batch_query,
                batch_passage=batch_passage,
                batch_score=batch_score,
                dataloader_prebatch_query=dataloader_prebatch_query,
                dataloader_prebatch_passage=dataloader_prebatch_passage,
            )
            metrics_all.update(metrics)

            print(f"nDCG@5 on {dataset_name}: {metrics[dataset_name]['ndcg_at_5']}")

            results = ViDoReBenchmarkResults(
                metadata=MetadataModel(
                    timestamp=datetime.now(),
                    vidore_benchmark_version=version("vidore_benchmark"),
                ),
                metrics={f"{dataset_name} ({config_suffix})": metrics[dataset_name]},
            )
            results_all.append(results)

        sanitized_dataset_name = dataset_name.replace("/", "_")
        savepath_results = savedir_datasets / f"{sanitized_dataset_name}_metrics.json"

        with open(str(savepath_results), "w", encoding="utf-8") as f:
            f.write(results.model_dump_json(indent=4))

        logger.info(f'ViDoRe Benchmark results for "{dataset_name}" saved to `{savepath_results}`')

    results_merged = ViDoReBenchmarkResults.merge(results_all)
    savepath_results_merged = savedir_root / f"{model_id}_metrics.json"

    with open(str(savepath_results_merged), "w", encoding="utf-8") as f:
        f.write(results_merged.model_dump_json(indent=4))

    print(f"ViDoRe Benchmark results saved to `{savepath_results_merged}`")


if __name__ == "__main__":
    app()
