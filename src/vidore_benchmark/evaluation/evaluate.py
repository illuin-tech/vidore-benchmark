from typing import Dict, List

from datasets import Dataset
from PIL import Image

from vidore_benchmark.retrievers.vision_retriever import VisionRetriever


def evaluate_dataset(
    vision_retriever: VisionRetriever,
    ds: Dataset,
    batch_query: int,
    batch_doc: int,
) -> Dict[str, float]:
    """
    Evaluate the model on a given dataset using the MTEB metrics.
    """

    # Dataset: sanity check
    col_documents = "image" if vision_retriever.use_visual_embedding else "text_description"
    col_to_check = ["query", col_documents, "image_filename"]

    if not all(col in ds.column_names for col in col_to_check):
        raise ValueError(f"Dataset should contain the following columns: {col_to_check}")

    # Remove None queries and duplicates
    queries = list(set(ds["query"]))
    if None in queries:
        queries.remove(None)
        if len(queries) == 0:
            raise ValueError("All queries are None")

    documents = ds[col_documents]

    # Get the scores - size (n_queries, n_documents)
    scores = vision_retriever.get_scores(queries, documents, batch_query=batch_query, batch_doc=batch_doc)

    # Get the relevant documents and results
    relevant_docs, results = vision_retriever.get_relevant_docs_results(ds, queries, scores)

    # compute MTEB metrics
    metrics = vision_retriever.compute_metrics(relevant_docs, results)

    return metrics


def get_top_k(
    vision_retriever: VisionRetriever,
    queries: List[str],
    documents: List["Image.Image | str"],
    file_names: List[str],
    batch_query: int,
    batch_doc: int,
    k: int,
) -> Dict[str, float]:
    """
    Get the top-k documents for a given query.
    """
    scores = vision_retriever.get_scores(queries, documents, batch_query=batch_query, batch_doc=batch_doc)

    passages2filename = {docidx: image_filename for docidx, image_filename in enumerate(file_names)}
    # Get the top-k documents

    results = {}
    for query, score_per_query in zip(queries, scores):
        for docidx, score in enumerate(score_per_query):
            filename = passages2filename[docidx]
            score_passage = float(score.item())

            if query in results:
                results[query][filename] = max(results[query].get(filename, 0), score_passage)
            else:
                results[query] = {filename: score_passage}
        # sort the results by score for each query
        results[query] = dict(sorted(results[query].items(), key=lambda item: item[1], reverse=True))
        # get the top-k documents
        results[query] = dict(list(results[query].items())[:k])

    return results
