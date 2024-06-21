### DEBUG
from vidore_benchmark.retrievers.bge_m3 import BGEM3
from vidore_benchmark.retrievers.jina_clip import JinaClip
from vidore_benchmark.retrievers.nomic import NomicVision
from vidore_benchmark.retrievers.dummy import DummyRetriever
from vidore_benchmark.retrievers.siglip import SigLip
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever 
from vidore_benchmark.evaluation.evaluation import evaluate_dataset

from datasets import load_dataset, Dataset
from typing import Dict, List, cast

from typing import Dict
from vidore_benchmark.evaluation.retrieval_evaluator import CustomEvaluator




def main():
    my_retriever = JinaClip(visual_embedding=True)

    dataset = cast(Dataset, load_dataset("coldoc/shiftproject_test", split='test'))

    print("Dataset loaded")
    metrics = evaluate_dataset(my_retriever, dataset, batch_query=1, batch_doc=4) # type: ignore

    print(metrics)


if __name__ == "__main__":
    main()
