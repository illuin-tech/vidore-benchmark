import typer
from typing import Annotated
from vidore_benchmark.utils.intialize_retrievers import create_vision_retriever
from datasets import load_dataset

"""
This script is used to evaluate a model on a given dataset using the MTEB metrics.
"""


def main(
    model_name: Annotated[str, typer.Option(help="Model name to use for evaluation")],
    dataset_name: Annotated[str, typer.Option(help="Dataset on Hugging Face to evaluate")] = "",
    batch_size: Annotated[int, typer.Option(help="Batch size to use for evaluation")] = 4,
    is_multi_vector: Annotated[bool, typer.Option(help="If True, multi-vector evaluation will be used")] = False,

    collection_name: Annotated[str, typer.Option(help="Collection name to use for evaluation")] = "",
    text_only: Annotated[bool, typer.Option(help="If True, only text chunks will be used for evaluation")] = False,
):

    # Create the vision retriever
    retriever = create_vision_retriever(model_name)

    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Evaluate the model
    metrics = retriever.evaluate_dataset(dataset=dataset, batch_size=batch_size, is_multi_vector=is_multi_vector)

    # TODO: Print or save the metrics
    print(metrics)


if __name__ == "__main__":
    typer.run(main)
