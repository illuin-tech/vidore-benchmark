import torch
from colpali_engine.models import ColIdefics3, ColIdefics3Processor
from datasets import load_dataset
from tqdm import tqdm

from vidore_benchmark.evaluation.vidore_evaluators import ViDoReEvaluatorQA, ViDoReEvaluatorBEIR
from vidore_benchmark.retrievers import VisionRetriever
from vidore_benchmark.utils.data_utils import get_datasets_from_collection

model_name = "vidore/colSmol-256M"
processor = ColIdefics3Processor.from_pretrained(model_name)
model = ColIdefics3.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
).eval()

# Get retriever instance
vision_retriever = VisionRetriever(model=model, processor=processor)

# Evaluate on a single BEIR format dataset (e.g one of the ViDoRe benchmark 2 dataset)
vidore_evaluator_beir = ViDoReEvaluatorBEIR(vision_retriever)
ds = {
    "corpus" : load_dataset("vidore/synthetic_axa_filtered_v1.0", name="corpus", split="test"),
    "queries" : load_dataset("vidore/synthetic_axa_filtered_v1.0", name="queries", split="test")
    "qrels" : load_dataset("vidore/synthetic_axa_filtered_v1.0", name="qrels", split="test")
}
metrics_dataset_beir = vidore_evaluator_beir.evaluate_dataset(
    ds=ds,
    batch_query=4,
    batch_passage=4,
)
print(metrics_dataset_beir)

# Evaluate on a single QA format dataset
vidore_evaluator_qa = ViDoReEvaluatorQA(vision_retriever)
ds = load_dataset("vidore/tabfquad_test_subsampled", split="test")
metrics_dataset_qa = vidore_evaluator_qa.evaluate_dataset(
    ds=ds,
    batch_query=4,
    batch_passage=4,
)
print(metrics_dataset_qa)

# Evaluate on a local directory or a HuggingFace collection
dataset_names = get_datasets_from_collection("vidore/vidore-benchmark-667173f98e70a1c0fa4db00d")
metrics_collection = {}
for dataset_name in tqdm(dataset_names, desc="Evaluating dataset(s)"):
    metrics_collection[dataset_name] = vidore_evaluator.evaluate_dataset(
        ds=load_dataset(dataset_name, split="test"),
        batch_query=4,
        batch_passage=4,
    )
print(metrics_collection)