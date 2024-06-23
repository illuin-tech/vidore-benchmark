import json
from typing import Annotated, cast

import typer
from datasets import Dataset, load_dataset
from vidore_benchmark.evaluation.evaluate import evaluate_dataset
from vidore_benchmark.retrievers.utils.initialize_retrievers import load_vision_retriever_from_registry
from vidore_benchmark.utils.constants import OUTPUT_DIR


def main(
    model_name: Annotated[str, typer.Option(help="Model name to use for evaluation")],
    dataset_name: Annotated[str, typer.Option(help="Dataset on Hugging Face to evaluate")],
    split: Annotated[str, typer.Option(help="Split to use for evaluation")],
    batch_query: Annotated[int, typer.Option(help="Batch size to use for evaluation")] = 1,
    batch_doc: Annotated[int, typer.Option(help="Batch size to use for evaluation")] = 4,
):
    """
    This script is used to evaluate a model on a given dataset using the MTEB metrics.
    """

    # Create the vision retriever
    retriever = load_vision_retriever_from_registry(model_name)

    # Load the dataset
    dataset = cast(Dataset, load_dataset(dataset_name, split=split))

    # Evaluate the model
    metrics = evaluate_dataset(retriever, dataset, batch_query=batch_query, batch_doc=batch_doc)

    # Save the metrics as a JSON file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    savepath = OUTPUT_DIR / f"{model_name.replace('/', '_')}_metrics.json"
    with open(savepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f)
    print(f"Metrics saved to `{savepath}`")


if __name__ == "__main__":
    typer.run(main)
