from typing import Annotated, cast

import typer
from datasets import Dataset, load_dataset
from vidore_benchmark.models.utils.initialize_retrievers import create_vision_retriever


def main(
    model_name: Annotated[str, typer.Option(help="Model name to use for evaluation")],
    dataset_name: Annotated[str, typer.Option(help="Dataset on Hugging Face to evaluate")],
    split: Annotated[str, typer.Option(help="Split of the dataset to use for evaluation")],
    batch_size: Annotated[int, typer.Option(help="Batch size to use for evaluation")],
    query: Annotated[str, typer.Option(help="Query to use for retrieval")],
    k: Annotated[int, typer.Option(help="Number of documents to retrieve")],
):
    """
    This script is used to ask a query and retrieve the top-k documents from a given HuggingFace Dataset.
    """

    # Create the vision retriever
    retriever = create_vision_retriever(model_name)

    # Load the dataset
    dataset = cast(Dataset, load_dataset(dataset_name, split=split))

    # Get the top-k documents
    top_k = retriever.get_top_k(
        queries=[query],
        ds=dataset,
        batch_size=batch_size,
        k=k,
    )

    print(f"Top-{k} documents for the query '{query}':")

    for document, score in top_k:
        print(f"Document: {document}, Score: {score}")


if __name__ == "__main__":
    typer.run(main)
