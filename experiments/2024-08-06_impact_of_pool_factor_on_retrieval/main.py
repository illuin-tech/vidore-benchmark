from __future__ import annotations

import json
import time
from typing import Annotated, Dict, List, Optional, Tuple, cast

import typer
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from vidore_benchmark.compression.token_pooling import HierarchicalEmbeddingPooler
from vidore_benchmark.retrievers.utils.load_retriever import load_vision_retriever_from_registry
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.constants import OUTPUT_DIR

load_dotenv(override=True)


def evaluate_dataset_with_different_pool_factors(
    vision_retriever: VisionRetriever,
    ds: Dataset,
    pool_factors: List[int],
    batch_query: int,
    batch_doc: int,
    batch_score: Optional[int] = None,
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, float]]:

    # Dataset: sanity check
    col_documents = "image" if vision_retriever.use_visual_embedding else "text_description"
    required_columns = ["query", col_documents, "image_filename"]

    if not all(col in ds.column_names for col in required_columns):
        raise ValueError(f"Dataset should contain the following columns: {required_columns}")

    # Remove `None` queries (i.e. pages for which no question was generated) and duplicates
    queries = list(set(ds["query"]))
    if None in queries:
        queries.remove(None)
        if len(queries) == 0:
            raise ValueError("All queries are None")
    documents = ds[col_documents]

    # Get the embeddings for the queries and documents
    emb_queries = vision_retriever.forward_queries(queries, batch_size=batch_query)
    emb_documents = vision_retriever.forward_documents(documents, batch_size=batch_doc)

    # Get result placeholder
    pool_factor_to_metrics: Dict[int, Dict[str, float]] = {}
    pool_factor_to_scoring_latency: Dict[int, float] = {}

    # Loop over the pool factors
    pbar = tqdm(pool_factors)
    for pool_factor in pbar:
        pbar.set_description(f"Pool factor: {pool_factor}")
        emb_documents_pooled = []

        embedding_pooler = HierarchicalEmbeddingPooler(pool_factor)
        for emb_document in tqdm(emb_documents, desc="Pooling embeddings..."):
            emb_document, _ = embedding_pooler.pool_embeddings(emb_document)
            emb_documents_pooled.append(emb_document)

        # Get the similarity scores
        start_time = time.perf_counter()
        scores = vision_retriever.get_scores(emb_queries, emb_documents_pooled, batch_size=batch_score)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        pool_factor_to_scoring_latency[pool_factor] = elapsed_time

        # Get the relevant documents and results
        relevant_docs, results = vision_retriever.get_relevant_docs_results(ds, queries, scores)

        # Compute the MTEB metrics
        pool_factor_to_metrics[pool_factor] = vision_retriever.compute_metrics(relevant_docs, results)

    return pool_factor_to_metrics, pool_factor_to_scoring_latency


app = typer.Typer(
    help="CLI for running experiments.",
    no_args_is_help=True,
)


@app.command()
def run_experiment(
    model_name: Annotated[str, typer.Option(help="Model name alias (tagged with `@register_vision_retriever`)")],
    pool_factors: Annotated[List[int], typer.Option(help="Pooling factors for hierarchical token pooling")],
    dataset_name: Annotated[str, typer.Option(help="HuggingFace Hub dataset name")],
    split: Annotated[str, typer.Option(help="Dataset split")] = "test",
    batch_query: Annotated[int, typer.Option(help="Batch size for query embedding inference")] = 4,
    batch_doc: Annotated[int, typer.Option(help="Batch size for document embedding inference")] = 4,
    batch_score: Annotated[Optional[int], typer.Option(help="Batch size for score computation")] = 4,
):
    # Create the vision retriever
    retriever = load_vision_retriever_from_registry(model_name)()

    # Load the dataset
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = cast(Dataset, load_dataset(dataset_name, split=split))
    pool_factor_to_metrics, pool_factor_to_scoring_latency = evaluate_dataset_with_different_pool_factors(
        retriever,
        dataset,
        pool_factors=pool_factors,
        batch_query=batch_query,
        batch_doc=batch_doc,
        batch_score=batch_score,
    )

    for pool_factor in pool_factors:
        metrics = {dataset_name: pool_factor_to_metrics[pool_factor]}
        savepath_metrics = (
            OUTPUT_DIR
            / f"{model_name.replace('/', '_')}"
            / f"{dataset_name.replace('/', '_')}_metrics_pool_factor_{pool_factor}.json"
        )
        savepath_metrics.parent.mkdir(parents=True, exist_ok=True)
        with open(str(savepath_metrics), "w", encoding="utf-8") as f:
            json.dump(metrics, f)
    print(f"Metrics saved in `{OUTPUT_DIR}`")

    savepath_latencies = (
        OUTPUT_DIR / f"{model_name.replace('/', '_')}" / f"{dataset_name.replace('/', '_')}_scoring_latencies.json"
    )
    savepath_latencies.parent.mkdir(parents=True, exist_ok=True)
    with open(str(savepath_latencies), "w", encoding="utf-8") as f:
        json.dump(pool_factor_to_scoring_latency, f)
    print(f"Scoring latencies saved in `{savepath_latencies}`")

    print("Done!")


if __name__ == "__main__":
    app()
