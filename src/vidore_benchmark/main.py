from pathlib import Path
from typing import Annotated, Optional, cast

import huggingface_hub
import typer
from datasets import Dataset, load_dataset
from dotenv import load_dotenv

from vidore_benchmark.evaluation.evaluate import evaluate_dataset, get_top_k
from vidore_benchmark.retrievers.utils.load_retriever import load_vision_retriever_from_registry
from vidore_benchmark.utils.constants import OUTPUT_DIR
from vidore_benchmark.utils.image_utils import generate_dataset_from_img_folder
from vidore_benchmark.utils.log_utils import log_metrics
from vidore_benchmark.utils.pdf_utils import convert_all_pdfs_to_images

load_dotenv(override=True)

app = typer.Typer(
    help="CLI for evaluating retrievers on the ViDoRe benchmark.",
    no_args_is_help=True,
)


@app.command()
def evaluate_retriever(
    model_name: Annotated[str, typer.Option(help="Model name alias (tagged with `@register_vision_retriever`)")],
    dataset_name: Annotated[Optional[str], typer.Option(help="HuggingFace Hub dataset name")] = None,
    split: Annotated[str, typer.Option(help="Dataset split")] = "test",
    batch_query: Annotated[int, typer.Option(help="Batch size for query embedding inference")] = 4,
    batch_doc: Annotated[int, typer.Option(help="Batch size for document embedding inference")] = 4,
    collection_name: Annotated[Optional[str], typer.Option(help="Collection name to use for evaluation")] = None,
):
    """
    Evaluate the retriever on the given dataset or collection.
    The metrics are saved to a JSON file.
    """

    # Sanity check
    if dataset_name is None and collection_name is None:
        raise ValueError("Please provide a dataset name or collection name")
    elif dataset_name is not None and collection_name is not None:
        raise ValueError("Please provide only one of dataset name or collection name")

    # Create the vision retriever
    retriever = load_vision_retriever_from_registry(model_name)

    # Load the dataset
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    savepath = OUTPUT_DIR / f"{model_name.replace('/', '_')}_metrics.json"
    metrics = {}

    if dataset_name is not None:
        dataset = cast(Dataset, load_dataset(dataset_name, split=split))
        metrics = evaluate_dataset(retriever, dataset, batch_query=batch_query, batch_doc=batch_doc)
        log_metrics(metrics, dataset_name, log_file=str(savepath))
        print(f"NDCG@5 for {model_name} on {dataset_name}: {metrics['ndcg_at_5']}")
    elif collection_name is not None:
        collection = huggingface_hub.get_collection(collection_name)
        datasets = collection.items
        for dataset_item in datasets:
            print(f"\n---------------------------\nEvaluating {dataset_item.item_id}")
            dataset = cast(Dataset, load_dataset(dataset_item.item_id, split=split))
            metrics = evaluate_dataset(retriever, dataset, batch_query=batch_query, batch_doc=batch_doc)
            log_metrics(metrics, dataset_item.item_id, log_file=str(savepath))
            print(f"Metrics saved to `{savepath}`")

            print(f"NDCG@5 for {model_name} on {dataset_item.item_id}: {metrics['ndcg_at_5']}")


@app.command()
def retrieve_on_dataset(
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


@app.command()
def retrieve_on_pdfs(
    model_name: Annotated[str, typer.Option(help="Model name to use for evaluation")],
    query: Annotated[str, typer.Option(help="Query to use for retrieval")],
    k: Annotated[int, typer.Option(help="Number of documents to retrieve")],
    data_dirpath: Annotated[
        str, typer.Option(help="Path to the folder containing the PDFs to use as the retrieval corpus")
    ],
    batch_size: Annotated[int, typer.Option(help="Batch size for document embedding inference")] = 4,
):
    """
    This script is used to ask a query and retrieve the top-k documents from a given folder containing PDFs.
    The PDFs will be converted to a dataset of image pages and then used for retrieval.
    """

    assert Path(data_dirpath).is_dir(), f"Invalid data directory: `{data_dirpath}`"

    # Create the vision retriever
    retriever = load_vision_retriever_from_registry(model_name)

    # Convert the PDFs to a collection of images
    convert_all_pdfs_to_images(data_dirpath)
    image_files = list(Path(data_dirpath).rglob("*.jpg"))
    print(f"Found {len(image_files)} images in the directory `{data_dirpath}`")

    # Generate a dataset using the images
    dataset = generate_dataset_from_img_folder(data_dirpath)

    # Get the top-k documents
    top_k = get_top_k(
        retriever,
        queries=[query],
        documents=list(dataset["image"]),
        file_names=list(dataset["image_filename"]),
        batch_query=1,
        batch_doc=batch_size,
        k=k,
    )

    print(f"Top-{k} documents for the query '{query}':")

    for document, score in top_k[query].items():  # type: ignore
        print(f"Document: {document}, Score: {score}")


if __name__ == "__main__":
    app()
