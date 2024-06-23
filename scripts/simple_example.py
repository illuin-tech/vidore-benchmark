"""
TODO: Remove file after debugging
"""

from typing import cast

from datasets import Dataset, load_dataset
from vidore_benchmark.evaluation.evaluate import evaluate_dataset
from vidore_benchmark.retrievers.jina_clip_retriever import JinaClip


def main():
    """
    Debugging script
    """
    my_retriever = JinaClip()

    dataset = cast(Dataset, load_dataset("coldoc/shiftproject_test", split="test"))

    print("Dataset loaded")
    metrics = evaluate_dataset(my_retriever, dataset, batch_query=1, batch_doc=4)  # type: ignore

    print(metrics)


if __name__ == "__main__":
    main()