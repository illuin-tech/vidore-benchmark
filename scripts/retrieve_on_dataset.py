from typing import Annotated, cast

import typer
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from vidore_benchmark.evaluation.evaluate import get_top_k
from vidore_benchmark.retrievers.utils.load_retriever import load_vision_retriever_from_registry

load_dotenv(override=True)


def main(
    model_name: Annotated[str, typer.Option(help="Model name alias (tagged with `@register_vision_retriever`)")],
    query: Annotated[str, typer.Option(help="Query to use for retrieval")],
    k: Annotated[int, typer.Option(help="Number of documents to retrieve")],
    dataset_name: Annotated[str, typer.Option(help="HuggingFace Hub dataset name")],
    split: Annotated[str, typer.Option(help="Dataset split")] = "test",
    batch_size: Annotated[int, typer.Option(help="Batch size for document embedding inference")] = 4,
):
    """
    Retrieve the top-k documents according to the given query.
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
        batch_doc=batch_size,
        k=k,
    )

    print(f"Top-{k} documents for the query '{query}':")
    for document, score in top_k[query].items():
        print(f"- Document `{document}` (score = {score})")


if __name__ == "__main__":
    typer.run(main)
