import json
from typing import Annotated, cast

import typer
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from vidore_benchmark.evaluation.evaluate import evaluate_dataset
from vidore_benchmark.retrievers.utils.initialize_retrievers import load_vision_retriever_from_registry
from vidore_benchmark.utils.constants import OUTPUT_DIR
from vidore_benchmark.evaluation.evaluate import evaluate_dataset
from vidore_benchmark.utils.log_utils import log_metrics
import huggingface_hub

load_dotenv(override=True)


def main(
    model_name: Annotated[str, typer.Option(help="Model name to use for evaluation")],
    split: Annotated[str, typer.Option(help="Split to use for evaluation")],
    dataset_name: Annotated[str, typer.Option(help="Dataset on Hugging Face to evaluate")] = "",
    batch_query: Annotated[int, typer.Option(help="Batch size to use for evaluation")] = 1,
    batch_doc: Annotated[int, typer.Option(help="Batch size to use for evaluation")] = 4,
    collection_name: Annotated[str, typer.Option(help="Collection name to use for evaluation")] = "",
):
    """
    This script is used to evaluate a model on a given dataset using the MTEB metrics.
    """

    # Create the vision retriever
    retriever = load_vision_retriever_from_registry(model_name)

    # Load the dataset
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    savepath = OUTPUT_DIR / f"{model_name.replace('/', '_')}_metrics.json"
    metrics = {}

    if dataset_name != "":
        dataset = cast(Dataset, load_dataset(dataset_name, split=split))
        metrics = evaluate_dataset(retriever, dataset, batch_query=batch_query, batch_doc=batch_doc)
        log_metrics(metrics, dataset_name, log_file=str(savepath))
        print(f"NDCG@5 for {model_name} on {dataset_name}: {metrics['ndcg_at_5']}")

    elif collection_name != "":
        collection = huggingface_hub.get_collection(collection_name)
        datasets = collection.items
        for dataset_item in datasets:
            print(f"\n---------------------------\nEvaluating {dataset_item.item_id}")
            dataset = cast(Dataset, load_dataset(dataset_item.item_id, split=split))
            metrics = evaluate_dataset(retriever, dataset, batch_query=batch_query, batch_doc=batch_doc)
            log_metrics(metrics, dataset_item.item_id, log_file=str(savepath))
            print(f"Metrics saved to {savepath}")

            print(f"NDCG@5 for {model_name} on {dataset_item.item_id}: {metrics['ndcg_at_5']}")
    else:
        raise ValueError("Please provide a dataset name or collection name")


if __name__ == "__main__":
    typer.run(main)
