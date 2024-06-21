from typing import Annotated, cast

import typer
from datasets import Dataset, load_dataset
from vidore_benchmark.retrievers.utils.initialize_retrievers import create_vision_retriever
from vidore_benchmark.evaluation.evaluation import get_top_k

def main(
    model_name: Annotated[str, typer.Option(help="Model name to use for evaluation")],
    dataset_name: Annotated[str, typer.Option(help="Dataset on Hugging Face to evaluate")],
    split: Annotated[str, typer.Option(help="Split of the dataset to use for evaluation")],
    batch_doc : Annotated[int, typer.Option(help="Batch size for documents")],
    query: Annotated[str, typer.Option(help="Query to use for retrieval")],
    k: Annotated[int, typer.Option(help="Number of documents to retrieve")],
):
    """
    This script is used to ask a query and retrieve the top-k documents from a given HuggingFace Dataset.
    >>> python scripts/retrieve_on_dataset.py --model-name BAAI/bge-m3 --dataset-name coldoc/shiftproject_test --split test --batch-query 1 --batch-doc 4 --k 5 --query 'Where is Eiffel Tower?'
    """

    # Create the vision retriever
    retriever = create_vision_retriever(model_name)

    # Load the dataset
    dataset = cast(Dataset, load_dataset(dataset_name, split=split))

    # Get the top-k documents
    top_k = get_top_k(
        retriever,
        [query],
        list(dataset["image"]) if retriever.visual_embedding else list(dataset["text_description"]),
        list(dataset["image_filename"]),
        batch_query=1,
        batch_doc=batch_doc,
        k=k,
    )
    print(f"Top-{k} documents for the query '{query}':")

    for document, score in top_k[query].items(): # type: ignore
        print(f"Document: {document}, Score: {score}")


if __name__ == "__main__":
    typer.run(main)
