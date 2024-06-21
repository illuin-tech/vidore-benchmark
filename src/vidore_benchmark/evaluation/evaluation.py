from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from typing import Dict
from datasets import Dataset
'''
All functions for evalaluation purposes
'''

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
    col_documents = "image" if vision_retriever.visual_embedding else "text_description"
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