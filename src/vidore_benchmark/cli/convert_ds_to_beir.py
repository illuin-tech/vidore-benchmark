from typing import Annotated, Dict, Generator, List, Optional, TypedDict, cast

import typer
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


def main(
    source_dataset: Annotated[
        str, typer.Option(..., help="Name of the Hf dataset to convert (e.g. 'vidore/syntheticDocQA').")
    ],
    split: Annotated[str, typer.Option(help="Name of the split to convert (e.g., 'test').")] = "test",
    target_dataset: Annotated[
        Optional[str],
        typer.Option(
            help="Name of the target Hf dataset to push the converted dataset to. Defaults to \
                `<source_dataset>_beir`."
        ),
    ] = None,
    query_column: Annotated[str, typer.Option(help="Name of the column containing the query.")] = "query",
    image_column: Annotated[str, typer.Option(help="Name of the column containing the image filename.")] = "image",
    image_filename_column: Annotated[
        str, typer.Option(help="Name of the column containing the image filename.")
    ] = "image_filename",
):
    """
    Convert a dataset from the QA format to the standard BEIR format.

    NOTE: To convert the OCR dataset from the "ViDoRe Chunk OCR (baseline)" collection, you should set
    `image_filename_column` to `"text_description"`.
    """

    if target_dataset is None:
        target_dataset = f"{source_dataset}_beir"
        print(f"Target dataset not provided. Using default target dataset name: `{target_dataset}`.")

    print(f"Starting conversion for dataset '{source_dataset}' and split '{split}'...")

    # Load the source dataset
    print(f"Loading source dataset '{source_dataset}'...")
    ds = cast(Dataset, load_dataset(source_dataset, split=split))

    # Initialize placeholders
    # NOTE: We will map the image filenames to the index in the dataset to efficiently
    # retrieve the image data later without overloading the RAM.
    image_filenames_to_ds_idx: Dict[str, int] = {}
    query_to_id: Dict[str, int] = {}
    image_filename_to_id: Dict[str, int] = {}

    # Process the dataset
    for idx_ds, row in tqdm(enumerate(ds), total=len(ds), desc="Processing dataset entries..."):
        if row[image_filename_column] not in image_filenames_to_ds_idx:
            image_filenames_to_ds_idx[row[image_filename_column]] = idx_ds

        if row[query_column] not in query_to_id and row[query_column] is not None:
            # NOTE: We ignore the `None` queries.
            query_to_id[row[query_column]] = len(query_to_id)  # Assign a unique ID to each query

    print(f"Total unique images: {len(image_filenames_to_ds_idx)}")
    print(f"Total unique queries: {len(query_to_id)}")

    # Create image to ID mapping
    for idx, name in enumerate(image_filenames_to_ds_idx):
        image_filename_to_id[name] = idx

    # Build corpus
    print("Building corpus...")

    CorpusItem = TypedDict("CorpusItem", {"corpus-id": int, "image": str})

    # NOTE: To prevent RAM overflow, we use a generator to create the corpus dataset
    # as images are much larger than text data.

    def corpus_generator() -> Generator[CorpusItem, None, None]:
        for name in image_filenames_to_ds_idx:
            image = ds[image_filenames_to_ds_idx[name]][image_column]
            yield {"corpus-id": image_filename_to_id[name], "image": image}

    ds_corpus = Dataset.from_generator(corpus_generator, split="test")

    # Build queries
    print("Building queries...")

    QueryItem = TypedDict("QueryItem", {"query-id": int, "query": str})
    queries: List[QueryItem] = []

    for query, idx in query_to_id.items():
        if query is not None:
            queries.append({"query-id": idx, "query": query})

    ds_queries = Dataset.from_list(queries, split="test")

    # Build query relevance judgments (qrels)
    print("Building query relevance judgments (qrels)...")

    QrelsItem = TypedDict("QrelsItem", {"query-id": int, "corpus-id": int, "score": float})
    qrels: List[QrelsItem] = []

    for row in ds:
        if row[query_column] is not None:
            qrels.append(
                {
                    "query-id": query_to_id[row[query_column]],
                    "corpus-id": image_filename_to_id[row[image_filename_column]],
                    "score": 1.0,
                }
            )

    ds_qrels = Dataset.from_list(qrels, split="test")

    # Push datasets to the Hugging Face Hub
    print(f"Pushing the 'corpus' split dataset to '{target_dataset}'...")
    ds_corpus.push_to_hub(target_dataset, config_name="corpus", private=True)

    print(f"Pushing the 'queries' split dataset to '{target_dataset}'...")
    ds_queries.push_to_hub(target_dataset, config_name="queries", private=True)

    print(f"Pushing 'the default' triples split dataset to '{target_dataset}'...")
    ds_qrels.push_to_hub(target_dataset, config_name="qrels", private=True)

    print(f"All datasets have been successfully pushed to '{target_dataset}'.")
    print("Done.")


if __name__ == "__main__":
    typer.run(main)
