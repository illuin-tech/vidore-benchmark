from typing import Annotated, cast

import typer
from datasets import Dataset, load_dataset
from vidore_benchmark.evaluation.evaluate import get_top_k
from vidore_benchmark.retrievers.utils.initialize_retrievers import load_vision_retriever_from_registry


def main(
    model_name: Annotated[str, typer.Option(help="Model name to use for evaluation")],
    dataset_name: Annotated[str, typer.Option(help="Dataset on Hugging Face to evaluate")],
    split: Annotated[str, typer.Option(help="Split of the dataset to use for evaluation")],
    batch_doc: Annotated[int, typer.Option(help="Batch size for documents")],
    query: Annotated[str, typer.Option(help="Query to use for retrieval")],
    k: Annotated[int, typer.Option(help="Number of documents to retrieve")],
):
    """
    This script is used to ask a query and retrieve the top-k documents from a given HuggingFace Dataset.
    
    >>> python scripts/retrieve_on_dataset.py \
        --model-name BAAI/bge-m3 \
        --dataset-name coldoc/shiftproject_test \
        --split test \
        --batch-query 1 \
        --batch-doc 4 \
        --k 5 \
        --query 'Where is Eiffel Tower?'
    """

    # Create the vision retriever
    retriever = load_vision_retriever_from_registry(model_name)

    # Load the dataset
    ds = cast(Dataset, load_dataset(dataset_name, split=split))

    # Get the top-k documents
    top_k = get_top_k(
        retriever,
        queries=[query],
        documents=list(ds["image"]) if retriever.use_visual_embedding else list(ds["text_description"]),
        file_names=list(ds["image_filename"]),
        batch_query=1,
        batch_doc=batch_doc,
        k=k,
    )
    print(f"Top-{k} documents for the query '{query}':")

    for document, score in top_k[query].items():  # type: ignore
        print(f"Document: {document}, Score: {score}")


if __name__ == "__main__":
    typer.run(main)
