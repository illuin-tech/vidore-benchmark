from typing import cast

from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from vidore_benchmark.evaluation.evaluate import evaluate_dataset
from vidore_benchmark.retrievers.jina_clip_retriever import JinaClipRetriever

load_dotenv(override=True)


def main():
    """
    Example script for a Python usage of the Vidore Benchmark.
    """
    my_retriever = JinaClipRetriever()
    dataset = cast(Dataset, load_dataset("vidore/syntheticDocQA_dummy", split="test"))
    metrics = evaluate_dataset(my_retriever, dataset, batch_query=4, batch_doc=4)
    print(metrics)


if __name__ == "__main__":
    main()
