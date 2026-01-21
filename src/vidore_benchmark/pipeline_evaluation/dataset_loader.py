"""
Dataset loader for vidore v3 benchmark datasets.

Handles downloading and preparing vidore v3 datasets from HuggingFace,
including queries, corpus images, and ground truth relevance judgments.
"""

from typing import Dict, List, Tuple

from datasets import load_dataset
from PIL import Image


def load_vidore_dataset(
    dataset_name: str, split: str = "test", language: str = None
) -> Tuple[List[str], List[str], List[str], List[Image.Image], Dict[str, Dict[str, int]], Dict[str, str]]:
    """
    Load a vidore v3 dataset from HuggingFace.

    Args:
        dataset_name: Name of the vidore v3 dataset (e.g., 'vidore/vidore_v3_industrial')
        split: Data split to load (default: 'test')
        language: Optional language filter (e.g., 'english', 'french').
                  If None, includes all queries. If specified, only queries
                  matching this language will be included.

    Returns:
        Tuple containing:
        - query_ids: List of query IDs (as strings)
        - queries: List of query texts
        - corpus_ids: List of corpus IDs (as strings)
        - corpus: List of PIL.Image objects
        - qrels: Ground truth relevance judgments in format {query_id: {corpus_id: score}}
        - query_languages: Mapping of query_id to language {query_id: language}

    Raises:
        ValueError: If dataset_name is not a valid vidore v3 dataset
        RuntimeError: If dataset loading fails

    Example:
        >>> query_ids, queries, corpus_ids, corpus, qrels, query_languages = load_vidore_dataset(
        ...     "vidore/vidore_v3_industrial"
        ... )
        >>> print(f"Loaded {len(queries)} queries and {len(corpus)} corpus items")

        >>> # Load only English queries
        >>> query_ids, queries, corpus_ids, corpus, qrels, query_languages = load_vidore_dataset(
        ...     "vidore/vidore_v3_hr",
        ...     language="english"
        ... )
    """
    # Validate dataset name
    available_datasets = get_available_datasets()
    if dataset_name not in available_datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}\nAvailable datasets: {', '.join(available_datasets)}")

    # Load the three components of the dataset
    try:
        queries_ds = load_dataset(dataset_name, data_dir="queries", split=split)
        corpus_ds = load_dataset(dataset_name, data_dir="corpus", split=split)
        qrels_ds = load_dataset(dataset_name, data_dir="qrels", split=split)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{dataset_name}' from HuggingFace: {e}") from e

    # Extract queries (with optional language filtering)
    query_languages = {}  # Map query_id to language

    if language:
        queries_ds = [item for item in queries_ds if item.get("language") == language]

    query_ids = [str(item["query_id"]) for item in queries_ds]
    queries = [item["query"] for item in queries_ds]

    # Store language information for each query
    for item in queries_ds:
        query_id = str(item["query_id"])
        query_languages[query_id] = item.get("language", "unknown")

    # Extract corpus (images)
    corpus_ids = [str(item["corpus_id"]) for item in corpus_ds]
    corpus = [item["image"] for item in corpus_ds]

    # Build qrels dictionary in pytrec_eval format
    # Format: {query_id: {corpus_id: relevance_score}}
    # Only include qrels for queries that passed language filter
    query_id_set = set(query_ids)
    qrels = {}
    for item in qrels_ds:
        query_id = str(item["query_id"])

        # Skip qrels for queries that were filtered out
        if query_id not in query_id_set:
            continue

        corpus_id = str(item["corpus_id"])
        score = int(item["score"])

        if query_id not in qrels:
            qrels[query_id] = {}
        qrels[query_id][corpus_id] = score

    # Validate loaded data
    if not queries:
        lang_msg = f" with language='{language}'" if language else ""
        raise ValueError(f"No queries found in dataset '{dataset_name}'{lang_msg}")
    if not corpus:
        raise ValueError(f"No corpus items found in dataset '{dataset_name}'")
    if not qrels:
        lang_msg = f" with language='{language}'" if language else ""
        raise ValueError(f"No relevance judgments found in dataset '{dataset_name}'{lang_msg}")

    return query_ids, queries, corpus_ids, corpus, qrels, query_languages


def get_available_datasets() -> List[str]:
    """
    Get list of available vidore v3 datasets.

    Returns:
        List of dataset names that can be used with load_vidore_dataset()
    """
    return [
        "vidore/vidore_v3_hr",
        "vidore/vidore_v3_finance_en",
        "vidore/vidore_v3_industrial",
        "vidore/vidore_v3_pharmaceuticals",
        "vidore/vidore_v3_computer_science",
        "vidore/vidore_v3_energy",
        "vidore/vidore_v3_physics",
        "vidore/vidore_v3_finance_fr",
    ]


def print_dataset_info(
    dataset_name: str,
    query_ids: List[str],
    queries: List[str],
    corpus_ids: List[str],
    corpus: List[Image.Image],
    qrels: Dict[str, Dict[str, int]],
) -> None:
    """
    Print summary information about a loaded dataset.

    Args:
        dataset_name: Name of the dataset
        query_ids: List of query IDs
        queries: List of query texts
        corpus_ids: List of corpus IDs
        corpus: List of corpus images
        qrels: Ground truth relevance judgments
    """
    total_judgments = sum(len(corpus_dict) for corpus_dict in qrels.values())
    avg_judgments_per_query = total_judgments / len(query_ids) if query_ids else 0

    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 60}")
    print(f"Queries:                {len(queries)}")
    print(f"Corpus items:           {len(corpus)}")
    print(f"Total relevance pairs:  {total_judgments}")
    print(f"Avg judgments/query:    {avg_judgments_per_query:.1f}")
    print(f"{'=' * 60}\n")

    # Show sample query
    if queries:
        print(f"Sample query (ID: {query_ids[0]}):")
        print(f"  {queries[0]}\n")
