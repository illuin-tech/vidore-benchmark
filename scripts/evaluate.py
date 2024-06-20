import json
from typing import Annotated, cast

import typer
from datasets import Dataset, load_dataset
from vidore_benchmark.models.utils.initialize_retrievers import create_vision_retriever
from vidore_benchmark.utils.constants import OUTPUT_DIR


def main(
    model_name: Annotated[str, typer.Option(help="Model name to use for evaluation")],
    dataset_name: Annotated[str, typer.Option(help="Dataset on Hugging Face to evaluate")],
    split: Annotated[str, typer.Option(help="Split to use for evaluation")],
    batch_size: Annotated[int, typer.Option(help="Batch size to use for evaluation")] = 4,
    collection_name: Annotated[str, typer.Option(help="Collection name to use for evaluation")] = "",
    text_only: Annotated[bool, typer.Option(help="If True, only text chunks will be used for evaluation")] = False,
):
    """
    This script is used to evaluate a model on a given dataset using the MTEB metrics.
    """

    # Create the vision retriever
    retriever = create_vision_retriever(model_name)

    # Load the dataset
    dataset = cast(Dataset, load_dataset(dataset_name, split=split))

    # Evaluate the model
    metrics = retriever.evaluate_dataset(ds=dataset, batch_size=batch_size)

    # Save the metrics as a JSON file
    savepath = OUTPUT_DIR / f"{model_name}_metrics.json"
    with open(savepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f)
    print(f"Metrics saved to `{savepath}`")


if __name__ == "__main__":
    typer.run(main)
