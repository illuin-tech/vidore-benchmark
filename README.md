# Vision Document Retrieval (ViDoRe): Benchmark 👀

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

The Visual Document Retrieval Benchmark (ViDoRe), is introduced to evaluate the performance of document retrieval systems on visually rich documents across various tasks, domains, languages, and settings. It was used to evaluate the ColPali model, a VLM-powered retriever that efficiently retrieves documents based on their visual content and textual queries using a late-interaction mechanism.

![ViDoRe Examples](assets/vidore_examples.webp)

> [!TIP]
> If you want to fine-tune ColPali for your specific use-case, you should check the [`colpali`](https://github.com/illuin-tech/colpali) repository. It contains with the whole codebase used to train the model presented in our paper.

## Setup

We used Python 3.11.6 and PyTorch 2.2.2 to train and test our models, but the codebase is expected to be compatible with Python >=3.9 and recent PyTorch versions.

The eval codebase depends on a few Python packages, which can be downloaded using the following command:

```bash
pip install vidore-benchmark
```

> [!TIP]
> By default, the `vidore-benchmark` package already includes the dependencies for the ColVision models (e.g. ColPali, ColQwen2...).

To keep a lightweight repository, only the essential packages were installed. In particular, you must specify the dependencies for the specific non-Transformers models you want to run (see the list in `pyproject.toml`). For instance, if you are going to evaluate the BGE-M3 retriever:

```bash
pip install "vidore-benchmark[bge-m3]"
```

Or if you want to evaluate all the off-the-shelf retrievers:

```bash
pip install "vidore-benchmark[all-retrievers]"
```

## Available retrievers

The list of available retrievers can be found [here](https://github.com/illuin-tech/vidore-benchmark/tree/main/src/vidore_benchmark/retrievers). Read [this section](###Implement-your-own-retriever) to learn how to create, use, and evaluate your own retriever.

## Command-line usage

### Evaluate a retriever on ViDoRE

You can evaluate any off-the-shelf retriever on the ViDoRe benchmark. For instance, you
can evaluate the ColPali model on the ViDoRe benchmark to reproduce the results from our paper.

```bash
vidore-benchmark evaluate-retriever \
    --model-class colpali \
    --model-name vidore/colpali-v1.2 \
    --collection-name vidore/vidore-benchmark-667173f98e70a1c0fa4db00d \
    --dataset-format qa \
    --split test
```

**Note:** You should get a warning about some non-initialized weights. This is a known issue in ColPali and will
cause the metrics to be slightly different from the ones reported in the paper. We are working on fixing this issue.

Alternatively, you can evaluate your model on a single dataset. If your retriver uses visual embeddings, you can use any dataset path from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d) collection, e.g.:

```bash
vidore-benchmark evaluate-retriever \
    --model-class colpali \
    --model-name vidore/colpali-v1.2 \
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

Both scripts will generate one particular JSON file in `outputs/{model_name_all_metrics.json}`. Follow the instructions on the [ViDoRe Leaderboard](https://huggingface.co/spaces/vidore/vidore-leaderboard) to compare your model with the others.

### Evaluate a retriever using token pooling

You can use token pooling to reduce the length of the document embeddings. In production, this will significantly reduce the memory footprint of the retriever, thus reducing costs and increasing speed. You can use the `--use-token-pooling` flag to enable this feature:

```bash
vidore-benchmark evaluate-retriever \
    --model-class colpali \
    --model-name vidore/colpali-v1.2 \
    --dataset-name vidore/docvqa_test_subsampled \
    --dataset-format qa \
    --split test \
    --use-token-pooling \
    --pool-factor 3
```

### Retrieve the top-k documents from a HuggingFace dataset

```bash
vidore-benchmark retrieve-on-dataset \
    --model-class colpali \
    --model-name vidore/colpali-v1.2 \
    --query "Which hour of the day had the highest overall electricity generation in 2019?" \
    --k 5 \
    --dataset-name vidore/syntheticDocQA_energy_test \
    --split test
```

### Retrieve the top-k documents from a collection of PDF documents

```bash
vidore-benchmark retriever_on_pdfs \
    --model-class siglip \
    --model-name google/siglip-so400m-patch14-384 \
    --query "Which hour of the day had the highest overall electricity generation in 2019?" \
    --k 5 \
    --data-dirpath data/my_folder_with_pdf_documents/
```

### Documentation

To get more information about the available options, run:

```bash
vidore-benchmark --help
```

## Python usage

### Quickstart example

```python
from datasets import load_dataset
from dotenv import load_dotenv
from vidore_benchmark.evaluation.vidore_evaluators import ViDoReEvaluatorQA
from vidore_benchmark.retrievers.jina_clip_retriever import JinaClipRetriever

load_dotenv(override=True)

def main():
    """
    Example script for a Python usage of the Vidore Benchmark.
    """
    # Load the model and the dataset
    my_retriever = JinaClipRetriever("jinaai/jina-clip-v1")
    ds = load_dataset("vidore/vidore_benchmark_qa_dummy", split="test")

    # Load the ViDoRe Evaluator
    vidore_evaluator = ViDoReEvaluatorQA(vision_retriever=my_retriever)

    # Get the MTEB metrics for retrieval
    metrics = vidore_evaluator.evaluate_dataset(
        ds=ds,
        batch_query=8,
        batch_passage=8,
        batch_score=16,
    )
    print(metrics)
```

### Implement your own retriever

If you need to evaluate your own model on the ViDoRe benchmark, you can create your own instance of `VisionRetriever` to use it with the evaluation scripts in this package. You can find the detailed instructions [here](https://github.com/illuin-tech/vidore-benchmark/blob/main/src/vidore_benchmark/retrievers/README.md).

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
```

If you want to reproduce the results from the ColPali paper, please read the [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) file for more information.
