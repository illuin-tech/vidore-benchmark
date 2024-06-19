import typer
from typing import Annotated
from datasets import load_dataset

from vidore_benchmark.utils.intialize_retrievers import create_vision_retriever

"""
This script is used to ask a query and retrieve the top-k documents from a given dataset.
"""


def main(
    model_name: Annotated[str, typer.Option(help="Model name to use for evaluation")],
    dataset_name: Annotated[str, typer.Option(help="Dataset on Hugging Face to evaluate")], 
    pdf_folder: Annotated[str, typer.Option(help="Path to the folder containing PDFs")],
    k: Annotated[int, typer.Option(help="Number of documents to retrieve")],
):

    # Create the vision retriever
    retriever = create_vision_retriever(model_name)
    query = [input("Enter the query: ")]

    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Get the top-k documents
    top_k = retriever.get_top_k(query, dataset, k)

    print(f"Top-{k} documents for the query '{query}':")

    for document, score in top_k:
        print(f"Document: {document}, Score: {score}")


if __name__ == "__main__":
    typer.run(main)
