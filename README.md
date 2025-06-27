# Vision Document Retrieval (ViDoRe): Benchmarks 👀

[![arXiv](https://img.shields.io/badge/arXiv-2407.01449-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2407.01449)
[![GitHub](https://img.shields.io/badge/ColPali_Engine-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/illuin-tech/colpali)
[![Hugging Face](https://img.shields.io/badge/Vidore_Hf_Space-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/vidore)

[![Test](https://github.com/illuin-tech/vidore-benchmark/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/illuin-tech/vidore-benchmark/actions/workflows/test.yml)
[![Version](https://img.shields.io/pypi/v/vidore-benchmark?color=%2334D058&label=pypi%20package)](https://pypi.org/project/vidore-benchmark/)
[![Downloads](https://static.pepy.tech/badge/vidore-benchmark)](https://pepy.tech/project/vidore-benchmark)

---

[[Model card]](https://huggingface.co/vidore/colpali)
[[ViDoRe Leaderboard]](https://huggingface.co/spaces/vidore/vidore-leaderboard)
[[Demo]](https://huggingface.co/spaces/manu/ColPali-demo)
[[Blog Post]](https://huggingface.co/blog/manu/colpali)

## Approach

The Visual Document Retrieval Benchmarks (ViDoRe v1 and v2), is introduced to evaluate the performance of document retrieval systems on visually rich documents across various tasks, domains, languages, and settings. It was used to evaluate the ColPali model, a VLM-powered retriever that efficiently retrieves documents based on their visual content and textual queries using a late-interaction mechanism.

![ViDoRe Examples](assets/vidore_examples.webp)

## ⚠️ Deprecation Warning: Moving from `vidore-benchmark` to `mteb`

Since `mteb` now supports image-text retrieval, we recommend using `mteb` to evaluate your retriever on the ViDoRe benchmark. We are deprecating `vidore-benchmark` to facilitate maintenance and have a single source of truth for the ViDoRe benchmark.

If you want your results to appear on the ViDoRe Leaderboard, you should add them to the `results` [Github Project](https://github.com/embeddings-benchmark/results). Check the *Submit your model* section of the [ViDoRe Leaderboard](https://huggingface.co/spaces/vidore/vidore-leaderboard) for more information.

### New Evaluation Process

Follow the instructions to setup `mteb` [here](https://github.com/embeddings-benchmark/mteb/tree/main?tab=readme-ov-file#installation). Then you have 2 options. 

#### Option 1: CLI

```bash
mteb run -b "ViDoRe(v1)" -m "vidore/colqwen2.5-v0.2"
mteb run -b "ViDoRe(v2)" -m "vidore/colqwen2.5-v0.2"
```

#### Option 2: Python Script

```python
import mteb
from mteb.model_meta import ModelMeta
from mteb.models.colqwen_models import ColQwen2_5Wrapper

# === Configuration ===
MODEL_NAME = "johndoe/mycolqwen2.5"
BENCHMARKS = ["ViDoRe(v1)", "ViDoRe(v2)"]

# === Model Metadata ===
custom_model_meta = ModelMeta(
    loader=ColQwen2_5Wrapper,
    name=MODEL_NAME,
    modalities=["image", "text"],
    framework="Colpali",
    similarity_fn_name="max_sim",
    # Optional metadata (fill in if available else None)
    ...
)

# === Load Model ===
custom_model = custom_model_meta.load_model(MODEL_NAME)

# === Load Tasks ===
tasks = mteb.get_benchmarks(names=BENCHMARKS)
evaluator = mteb.MTEB(tasks=tasks)

# === Run Evaluation ===
results = evaluator.run(custom_model)
```

For custom models, you should implement your own wrapper. Check the [ColPaliEngineWrapper](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/models/colpali_models.py) for an example.

## [Deprecated] Usage

This packages comes with a Python API and a CLI to evaluate your own retriever on the ViDoRe benchmark. Both are compatible with `Python>=3.9`.

### CLI mode

```bash
pip install vidore-benchmark
```

To keep this package lightweight, only the essential packages were installed. Thus, you must specify the dependency groups for models you want to evaluate with CLI (see the list in `pyproject.toml`). For instance, if you are going to evaluate the ColVision models (e.g. ColPali, ColQwen2, ColSmol, ...), you should run:

```bash
pip install "vidore-benchmark[colpali-engine]"
```

> [!WARNING]
> If possible, do not `pip install colpali-engine` directly in the env dedicated for the CLI.
>
> In particular, make sure not to install both `vidore-benchmark[colpali-engine]` and `colpali-engine[train]` simultaneously, as it will lead to a circular depencency conflict.

If you want to install all the dependencies for all the models, you can run:

```bash
pip install "vidore-benchmark[all-retrievers]"
```

Note that in order to use `BM25Retriever`, you will need to download the `nltk` resources too:

```bash
pip install "vidore-benchmark[bm25]"
python -m nltk.downloader punkt punkt_tab stopwords
```

### Library mode

Install the base package using pip:

```bash
pip install vidore-benchmark
```

## Command-line usage

### Evaluate a retriever on ViDoRE

You can evaluate any off-the-shelf retriever on the ViDoRe benchmark v1. For instance, you
can evaluate the ColPali model on the ViDoRe benchmark 1 to reproduce the results from our paper.

```bash
vidore-benchmark evaluate-retriever \
    --model-class colpali \
    --model-name vidore/colpali-v1.3 \
    --collection-name vidore/vidore-benchmark-667173f98e70a1c0fa4db00d \
    --dataset-format qa \
    --split test
```

If you want to evaluate your models on on new collection ViDoRe benchmark 2, a harder version of the previous benchmark you can execute the following command:

```bash
vidore-benchmark evaluate-retriever \
    --model-class colpali \
    --model-name vidore/colpali-v1.3 \
    --collection-name vidore/vidore-benchmark-v2-67ae03e3924e85b36e7f53b0 \
    --dataset-format beir \
    --split test
```

Alternatively, you can evaluate your model on a single dataset. If your retriver uses visual embeddings, you can use any dataset path from the [ViDoRe Benchmark v1](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d) collection or the [ViDoRe Benchmark v2](https://huggingface.co/collections/vidore/vidore-benchmark-v2-67ae03e3924e85b36e7f53b0) (beir format instead of qa), e.g.:

```bash
vidore-benchmark evaluate-retriever \
    --model-class colpali \
    --model-name vidore/colpali-v1.3 \
    --dataset-name vidore/docvqa_test_subsampled \
    --dataset-format qa \
    --split test
```

If you want to evaluate a retriever that relies on pure-text retrieval (no visual embeddings), you should use the datasets from the [ViDoRe Chunk OCR (baseline)](https://huggingface.co/collections/vidore/vidore-chunk-ocr-baseline-666acce88c294ef415548a56) instead:

```bash
vidore-benchmark evaluate-retriever \
    --model-class bge-m3 \
    --model-name BAAI/bge-m3 \
    --dataset-name vidore/docvqa_test_subsampled_tesseract \
    --dataset-format qa \
    --split test
```

All the above scripts will generate a JSON file in `outputs/{model_id}_metrics.json`. Follow the instructions on the [ViDoRe Leaderboard](https://huggingface.co/spaces/vidore/vidore-leaderboard) to learn how to publish your results on the leaderboard too!

> [!NOTE]
> The `vidore-benchmark` package supports two formats of datasets:
>
> - QA: The dataset is formatted as a question-answering task, where the queries are questions and the passages are the image pages that provide the answers.
> - BEIR: Following the [BEIR paper](https://doi.org/10.48550/arXiv.2104.08663), the dataset is formatted in 3 sub-datasets: `corpus`, `queries`, and `qrels`. The `corpus` contains the documents, the `queries` contains the queries, and the `qrels` contains the relevance scores between the queries and the documents.
>
> In the first iteration of the ViDoRe benchmark, we **arbitrarily choose** to deduplicate the queries for the QA datasets. While this made sense given our data generation process, it wasn't suited for our ViDoRe benchmark v2 which aims at being broader and multilingual. We will release the ViDoRe benchmark v2 soon.

| Dataset                                                                                                    | Dataset format | Deduplicate queries |
|------------------------------------------------------------------------------------------------------------|----------------|---------------------|
| [ViDoRe benchmark v1](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d) | QA             | ✅                   |
| [ViDoRe benchmark v2](https://huggingface.co/collections/vidore/vidore-benchmark-v2-67ae03e3924e85b36e7f53b0) (harder/multilingual)                                                | BEIR           | ❌                   |

### Documentation

To have more control over the evaluation process (e.g. the batch size used at inference), read the CLI documentation using:

```bash
vidore-benchmark evaluate-retriever --help
```

In particular, feel free to play with the `--batch-query`, `--batch-passage`, `--batch-score`, and `--num-workers` inputs to speed up the evaluation process.

## Python usage

### Quickstart example

While the CLI can be used to evaluate a fixed list of models, you can also use the Python API to evaluate your own retriever. Here is an example of how to evaluate the ColPali model on the ViDoRe benchmark. Note that your processor must implement a `process_images` and a `process_queries` methods, similarly to the ColVision processors.

```python
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
    "corpus" : load_dataset("vidore/synthetic_rse_restaurant_filtered_v1.0", name="corpus", split="test"),
    "queries" : load_dataset("vidore/synthetic_rse_restaurant_filtered_v1.0", name="queries", split="test")
    "qrels" : load_dataset("vidore/synthetic_rse_restaurant_filtered_v1.0", name="qrels", split="test")
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
```

### Implement your own retriever

If you want to evaluate your own retriever to use it with the CLI, you should clone the repository and add your own class that inherits from `BaseVisionRetriever`. You can find the detailed instructions [here](https://github.com/illuin-tech/vidore-benchmark/blob/main/src/vidore_benchmark/retrievers/README.md).

### Compare retrievers using the EvalManager

To easily process, visualize and compare the evaluation metrics of multiple retrievers, you can use the `EvalManager` class. Assume you have a list of previously generated JSON metric files, *e.g.*:

```bash
data/metrics/
├── bisiglip.json
└── colpali.json
```

The data is stored in `eval_manager.data` as a multi-column DataFrame with the following columns. Use the `get_df_for_metric`, `get_df_for_dataset`, and `get_df_for_model` methods to get the subset of the data you are interested in. For instance:

```python
from vidore_benchmark.evaluation import EvalManager

eval_manager = EvalManager.from_dir("data/metrics/")
df = eval_manager.get_df_for_metric("ndcg_at_5")
```

## Citation

**ColPali: Efficient Document Retrieval with Vision Language Models**  

Authors: **Manuel Faysse**\*, **Hugues Sibille**\*, **Tony Wu**\*, Bilel Omrani, Gautier Viaud, Céline Hudelot, Pierre Colombo (\* denotes equal contribution)

```latex
@misc{faysse2024colpaliefficientdocumentretrieval,
      title={ColPali: Efficient Document Retrieval with Vision Language Models}, 
      author={Manuel Faysse and Hugues Sibille and Tony Wu and Bilel Omrani and Gautier Viaud and Céline Hudelot and Pierre Colombo},
      year={2024},
      eprint={2407.01449},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2407.01449}, 
}

@misc{macé2025vidorebenchmarkv2raising,
      title={ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval}, 
      author={Quentin Macé and António Loison and Manuel Faysse},
      year={2025},
      eprint={2505.17166},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2505.17166}, 
}
```

If you want to reproduce the results from the ColPali paper, please read the [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) file for more information.
