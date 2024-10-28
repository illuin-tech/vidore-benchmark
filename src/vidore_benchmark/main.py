import json
from pathlib import Path
from typing import Annotated, Optional, cast

import huggingface_hub
import numpy as np
import typer
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from loguru import logger

from vidore_benchmark.compression.quantization import EmbeddingBinarizer, EmbeddingInt8Quantizer
from vidore_benchmark.compression.token_pooling import HierarchicalEmbeddingPooler
from vidore_benchmark.evaluation.evaluate import evaluate_dataset, get_top_k
from vidore_benchmark.retrievers.colpali_retriever import ColPaliRetriever
from vidore_benchmark.retrievers.utils.load_retriever import load_vision_retriever_from_registry
from vidore_benchmark.utils.constants import OUTPUT_DIR
from vidore_benchmark.utils.image_utils import generate_dataset_from_img_folder
from vidore_benchmark.utils.logging_utils import setup_logging
from vidore_benchmark.utils.pdf_utils import convert_all_pdfs_to_images

load_dotenv(override=True)

app = typer.Typer(
    help="CLI for evaluating retrievers on the ViDoRe benchmark.",
    no_args_is_help=True,
)


@app.callback()
def main(log_level: Annotated[str, typer.Option("--log", help="Logging level")] = "warning"):
    logger.enable("vidore_benchmark")
    setup_logging(log_level)


@app.command()
def evaluate_retriever(
    model_class: Annotated[str, typer.Option(help="Model name alias (tagged with `@register_vision_retriever`)")],
    model_name: Annotated[str, typer.Option(help="Model name or path")] = None,
    dataset_name: Annotated[Optional[str], typer.Option(help="HuggingFace Hub dataset name")] = None,
    split: Annotated[str, typer.Option(help="Dataset split")] = "test",
    batch_query: Annotated[int, typer.Option(help="Batch size for query embedding inference")] = 4,
    batch_doc: Annotated[int, typer.Option(help="Batch size for document embedding inference")] = 4,
    batch_score: Annotated[Optional[int], typer.Option(help="Batch size for score computation")] = 4,
    collection_name: Annotated[Optional[str], typer.Option(help="Collection name to use for evaluation")] = None,
    quantization: Annotated[Optional[str], typer.Option(help="Quantization method to use for embeddings")] = None,
    use_token_pooling: Annotated[bool, typer.Option(help="Whether to use token pooling for text embeddings")] = False,
    pool_factor: Annotated[int, typer.Option(help="Pooling factor for hierarchical token pooling")] = 3,
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
    if model_name:
        retriever = load_vision_retriever_from_registry(model_class)(model_name=model_name)
        model_class = model_name
    else:
        retriever = load_vision_retriever_from_registry(model_class)()

    # Get the quantization strategy
    if quantization == "binarize":
        embedding_quantizer = EmbeddingBinarizer()
    elif quantization == "int8":
        if isinstance(retriever, ColPaliRetriever):
            # ColPali's embeddings are L2-normalized so they are in the range [-1, 1]
            naive_ranges = np.array([[-1.0, 1.0]] * retriever.emb_dim_doc).T
        else:
            raise NotImplementedError("Int8 quantization is only supported for ColPali retriever.")
        embedding_quantizer = EmbeddingInt8Quantizer(ranges=naive_ranges)
    elif quantization is None:
        embedding_quantizer = None
    else:
        raise ValueError(f"Invalid quantization method: {quantization}")

    # Get the pooling strategy
    embedding_pooler = HierarchicalEmbeddingPooler(pool_factor) if use_token_pooling else None

    # Create the output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the dataset
    if dataset_name is not None:
        dataset = cast(Dataset, load_dataset(dataset_name, split=split))
        metrics = {
            dataset_name: evaluate_dataset(
                retriever,
                dataset,
                batch_query=batch_query,
                batch_doc=batch_doc,
                batch_score=batch_score,
                embedding_quantizer=embedding_quantizer,
                embedding_pooler=embedding_pooler,
            )
        }

        if use_token_pooling:
            savepath = OUTPUT_DIR / f"{model_class.replace('/', '_')}_metrics_pool_factor_{pool_factor}.json"
        else:
            savepath = OUTPUT_DIR / f"{model_class.replace('/', '_')}_metrics.json"

        with open(str(savepath), "w", encoding="utf-8") as f:
            json.dump(metrics, f)

        print(f"Metrics saved to `{savepath}`")
        print(f"NDCG@5 for {model_class} on {dataset_name}: {metrics[dataset_name]['ndcg_at_5']}")

    elif collection_name is not None:
        # if it's a local directory, load the datasets from there
        import os
        if os.path.isdir(collection_name):
            print(f"Loading datasets from local directory: {collection_name}")
            datasets = os.listdir(collection_name)
            datasets = [os.path.join(collection_name, dataset) for dataset in datasets]
        else:
            print(f"Loading datasets from hub collection: {collection_name}")
            collection = huggingface_hub.get_collection(collection_name)
            datasets = collection.items
            datasets = [dataset_item.item_id for dataset_item in datasets]

        metrics_all = {}
        savedir = OUTPUT_DIR / model_class.replace("/", "_")
        savedir.mkdir(parents=True, exist_ok=True)

        for dataset_item in datasets:
            print(f"\n---------------------------\nEvaluating {dataset_item}")
            dataset = cast(Dataset, load_dataset(dataset_item, split=split))
            dataset_item = dataset_item.replace(collection_name + "/", "")
            metrics = {
                dataset_item: evaluate_dataset(
                    retriever,
                    dataset,
                    batch_query=batch_query,
                    batch_doc=batch_doc,
                    batch_score=batch_score,
                    embedding_quantizer=embedding_quantizer,
                    embedding_pooler=embedding_pooler,
                )
            }
            metrics_all.update(metrics)

            if use_token_pooling:
                savepath = savedir / f"{dataset_item.replace('/', '_')}_metrics_pool_factor_{pool_factor}.json"
            else:
                savepath = savedir / f"{dataset_item.replace('/', '_')}_metrics.json"

            with open(str(savepath), "w", encoding="utf-8") as f:
                json.dump(metrics, f)

            print(f"Metrics saved to `{savepath}`")
            print(f"NDCG@5 for {model_class} on {dataset_item}: {metrics[dataset_item]['ndcg_at_5']}")

        if use_token_pooling:
            savepath_all = OUTPUT_DIR / f"{model_class.replace('/', '_')}_all_metrics_pool_factor_{pool_factor}.json"
        else:
            savepath_all = OUTPUT_DIR / f"{model_class.replace('/', '_')}_all_metrics.json"

        with open(str(savepath_all), "w", encoding="utf-8") as f:
            json.dump(metrics_all, f)

        print(f"Concatenated metrics saved to `{savepath_all}`")

    else:
        raise ValueError("Please provide a dataset name or collection name.")

    print("Done.")


@app.command()
def retrieve_on_dataset(
    model_class: Annotated[str, typer.Option(help="Model name alias (tagged with `@register_vision_retriever`)")],
    model_name: Annotated[str, typer.Option(help="Model name or path")],
    query: Annotated[str, typer.Option(help="Query to use for retrieval")],
    k: Annotated[int, typer.Option(help="Number of documents to retrieve")],
    dataset_name: Annotated[str, typer.Option(help="HuggingFace Hub dataset name")],
    split: Annotated[str, typer.Option(help="Dataset split")] = "test",
    batch_doc: Annotated[int, typer.Option(help="Batch size for document embedding inference")] = 4,
    batch_score: Annotated[Optional[int], typer.Option(help="Batch size for score computation")] = 4,
):
    """
    Retrieve the top-k documents according to the given query.
    """

    # Create the vision retriever
    if model_name:
        retriever = load_vision_retriever_from_registry(model_class)(model_name=model_name)
    else:
        retriever = load_vision_retriever_from_registry(model_class)()


    # Load the dataset
    ds = cast(Dataset, load_dataset(dataset_name, split=split))

    # Get embeddings for the queries and documents
    emb_queries = retriever.forward_queries([query], batch_size=1)
    emb_documents = retriever.forward_documents(
        list(ds["image"]) if retriever.use_visual_embedding else list(ds["text_description"]),
        batch_size=batch_doc,
    )

    # Get the top-k documents
    top_k = get_top_k(
        retriever,
        queries=[query],
        emb_queries=emb_queries,
        emb_documents=emb_documents,
        file_names=list(ds["image_filename"]),
        k=k,
        batch_score=batch_score,
    )

    print(f"Top-{k} documents for the query '{query}':")
    for document, score in top_k[query].items():
        print(f"- Document `{document}` (score = {score})")

    print("Done.")


@app.command()
def retrieve_on_pdfs(
    model_class: Annotated[str, typer.Option(help="Model name alias (tagged with `@register_vision_retriever`)")],
    model_name: Annotated[str, typer.Option(help="Model name or path")],
    query: Annotated[str, typer.Option(help="Query to use for retrieval")],
    k: Annotated[int, typer.Option(help="Number of documents to retrieve")],
    data_dirpath: Annotated[
        str, typer.Option(help="Path to the folder containing the PDFs to use as the retrieval corpus")
    ],
    batch_doc: Annotated[int, typer.Option(help="Batch size for document embedding inference")] = 4,
    batch_score: Annotated[Optional[int], typer.Option(help="Batch size for score computation")] = 4,
):
    """
    This script is used to ask a query and retrieve the top-k documents from a given folder containing PDFs.
    The PDFs will be converted to a dataset of image pages and then used for retrieval.
    """

    if not Path(data_dirpath).is_dir():
        raise FileNotFoundError(f"Invalid data directory: `{data_dirpath}`")

    # Create the vision retriever
    if model_name:
        retriever = load_vision_retriever_from_registry(model_class)(model_name=model_name)
    else:
        retriever = load_vision_retriever_from_registry(model_class)()

    # Convert the PDFs to a collection of images
    convert_all_pdfs_to_images(data_dirpath)
    image_files = list(Path(data_dirpath).rglob("*.jpg"))
    print(f"Found {len(image_files)} images in the directory `{data_dirpath}`")

    # Generate a dataset using the images
    ds = generate_dataset_from_img_folder(data_dirpath)

    # Get embeddings for the queries and documents
    emb_queries = retriever.forward_queries([query], batch_size=1)
    emb_documents = retriever.forward_documents(
        list(ds["image"]),
        batch_size=batch_doc,
    )

    # Get the top-k documents
    top_k = get_top_k(
        retriever,
        queries=[query],
        emb_queries=emb_queries,
        emb_documents=emb_documents,
        file_names=list(ds["image_filename"]),
        k=k,
        batch_score=batch_score,
    )

    print(f"Top-{k} documents for the query '{query}':")

    for document, score in top_k[query].items():  # type: ignore
        print(f"Document: {document}, Score: {score}")

    print("Done.")


if __name__ == "__main__":
    app()
