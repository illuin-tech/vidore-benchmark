from pathlib import Path
from typing import Annotated

import typer
from vidore_benchmark.models.utils.initialize_retrievers import create_vision_retriever


def main(
    model_name: Annotated[str, typer.Option(help="Model name to use for evaluation")],
    data_dirpath: Annotated[
        str, typer.Option(help="Path to the folder containing the PDFs to use as the retrieval corpus")
    ],
    batch_size: Annotated[int, typer.Option(help="Batch size to use for evaluation")],
    query: Annotated[str, typer.Option(help="Query to use for retrieval")],
    k: Annotated[int, typer.Option(help="Number of documents to retrieve")],
):
    """
    This script is used to ask a query and retrieve the top-k documents from a given folder containing PDFs.
    The PDFs will be converted to a dataset of image pages and then used for retrieval.
    """

    assert Path(data_dirpath).is_dir(), f"Invalid data directory: `{data_dirpath}`"

    # Create the vision retriever
    retriever = create_vision_retriever(model_name)

    # Convert the PDFs in data_dirpath to a dataset
    # dataset = ...  # TODO

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
