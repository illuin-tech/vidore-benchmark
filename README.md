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

The Visual Document Retrieval Benchmark (ViDoRe), is introduced to evaluate and enhance the performance of document retrieval systems on visually rich documents across various tasks, domains, languages, and settings. It was used to evaluate the ColPali model, a VLM-powered retriever that efficiently retrieves documents based on their visual content and textual queries.

> [!TIP]
> If you want to fine-tune ColPali for your specific use-case, you should check the [`colpali`](https://github.com/illuin-tech/colpali) repository. It contains with the whole codebase used to train the model presented in our paper.

## Setup

We used Python 3.11.6 and PyTorch 2.2.2 to train and test our models, but the codebase is expected to be compatible with Python >=3.9 and recent PyTorch versions.

The eval codebase depends on a few Python packages, which can be downloaded using the following command:

```bash
pip install vidore-benchmark
```

To keep a lightweight repository, only the essential packages were installed. In particular, you must specify the dependencies for the specific non-Transformers models you want to run (see the list in `pyproject.toml`). For instance, if you are going to evaluate the BGE-M3 retriever:

```bash
pip install "vidore-benchmark[bge-m3]"
```

Or if you want to evaluate all the off-the-shelf retrievers:

```bash
pip install "vidore-benchmark[all-retrievers]"
```

Finally, if you are willing to reproduce the results from the ColPali paper, you should clone the repository, checkout to the `3.3.0` tag or below, and use the `requirements-dev.txt` file to install the dependencies used at test time:

```bash
pip install -r requirements-dev.txt
```

## Available retrievers

The list of available retrievers can be found [here](https://github.com/illuin-tech/vidore-benchmark/tree/main/src/vidore_benchmark/retrievers). Read [this section](###Implement-your-own-retriever) to learn how to create, use, and evaluate your own retriever.

## Command-line usage

### Evaluate a retriever on ViDoRE

You can evaluate any off-the-shelf retriever on the ViDoRe benchmark. For instance, you
can evaluate the ColPali model on the ViDoRe benchmark to reproduce the results from our paper.

```bash
vidore-benchmark evaluate-retriever \
    --model-name vidore/colpali \
    --collection-name "vidore/vidore-benchmark-667173f98e70a1c0fa4db00d" \
    --split test
```

**Note:** You should get a warning about some non-initialized weights. This is a known issue in ColPali and will
cause the metrics to be slightly different from the ones reported in the paper. We are working on fixing this issue.

Alternatively, you can evaluate your model on a single dataset. If your retriver uses visual embeddings, you can use any dataset path from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d) collection, e.g.:

```bash
vidore-benchmark evaluate-retriever \
    --model-name vidore/colpali \
    --dataset-name vidore/docvqa_test_subsampled \
    --split test
```

If you want to evaluate a retriever that relies on pure-text retrieval (no visual embeddings), you should use the datasets from the [ViDoRe Chunk OCR (baseline)](https://huggingface.co/collections/vidore/vidore-chunk-ocr-baseline-666acce88c294ef415548a56) instead:

```bash
vidore-benchmark evaluate-retriever \
    --model-name BAAI/bge-m3 \
    --dataset-name vidore/docvqa_test_subsampled_tesseract \
    --split test
```

Both scripts will generate one particular JSON file in `outputs/{model_name_all_metrics.json}`. Follow the instructions on the [ViDoRe Leaderboard](https://huggingface.co/spaces/vidore/vidore-leaderboard) to compare your model with the others.

### Evaluate a retriever using embedding compression techniques

#### Binarization (experimental)

Binarization (or binary quantization) converts the float32 values in an embedding into 1-bit values, leading to a 32x decrease in memory and storage requirements. See [this HuggingFace blog post]((https://huggingface.co/blog/embedding-quantization#binary-quantization).) for more information on binarization. To apply binarization on your embeddings, you can use the `--quantization binarize` flag:

```bash
vidore-benchmark evaluate-retriever \
    --model-name vidore/colpali \
    --dataset-name vidore/docvqa_test_subsampled \
    --split test \
    --quantization binarize
```

#### Int8 quantization (experimental)

Int8 quantization maps the continuous float32 value range to a discrete set of int8 values, capable of representing 256 distinct levels (ranging from -128 to 127). The mapping bins are computed using the minimum and maximum values for each embedding dimension. See [this HuggingFace blog post]((https://huggingface.co/blog/embedding-quantization#scalar-int8-quantization).) for more information on int8 quantization. To apply int8 quantization on your embeddings, you can use the `--quantization int8` flag:

```bash
vidore-benchmark evaluate-retriever \
    --model-name vidore/colpali \
    --dataset-name vidore/docvqa_test_subsampled \
    --split test \
    --quantization int8
```

#### Token pooling

You can use token pooling to reduce the length of the document embeddings. In production, this will significantly reduce the memory footprint of the retriever, thus reducing costs and increasing speed. You can use the `--use-token-pooling` flag to enable this feature:

```bash
vidore-benchmark evaluate-retriever \
    --model-name vidore/colpali \
    --dataset-name vidore/docvqa_test_subsampled \
    --split test \
    --use-token-pooling \
    --pool-factor 3
```

### Retrieve the top-k documents from a HuggingFace dataset

```bash
vidore-benchmark retrieve-on-dataset \
    --model-name vidore/colpali \
    --query "Which hour of the day had the highest overall electricity generation in 2019?" \
    --k 5 \
    --dataset-name vidore/syntheticDocQA_energy_test \
    --split test
```

### Retrieve the top-k documents from a collection of PDF documents

```bash
vidore-benchmark retriever_on_pdfs \
    --model-name google/siglip-so400m-patch14-384 \
    --query "Which hour of the day had the highest overall electricity generation in 2019?" \
    --k 5 \
    --data-dirpath data/my_folder_with_pdf_documents/
```

### Documentation

To get more information about the available options, run:

```bash
❯ vidore-benchmark --help
                                                                                                                      
 Usage: vidore-benchmark [OPTIONS] COMMAND [ARGS]...                                                                       
                                                                                                                      
 CLI for evaluating retrievers on the ViDoRe benchmark.                                                               
                                                                                                                      
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                                            │
│ --show-completion             Show completion for the current shell, to copy it or customize the installation.     │
│ --help                        Show this message and exit.                                                          │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ─────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ evaluate-retriever    Evaluate the retriever on the given dataset or collection. The metrics are saved to a JSON   │
│                       file.                                                                                        │
│ retrieve-on-dataset   Retrieve the top-k documents according to the given query.                                   │
│ retrieve-on-pdfs      This script is used to ask a query and retrieve the top-k documents from a given folder      │
│                       containing PDFs. The PDFs will be converted to a dataset of image pages and then used for    │
│                       retrieval.                                                                                   │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Python usage

### Quickstart example

```python
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
from vidore_benchmark.evaluation.eval_manager import EvalManager

eval_manager = EvalManager.from_dir("data/metrics/")
df = eval_manager.get_df_for_metric("ndcg_at_5")
```

### Show the similarity maps for interpretability

By superimposing the late interaction heatmap on top of the original image, we can visualize the most salient image patches with respect to each term of the query, yielding interpretable insights into model focus zones.

You can generate similarity maps using the `generate-similarity-maps`. For instance, you can reproduce the similarity maps from the paper using the images from [`data/interpretability_examples`](https://github.com/illuin-tech/vidore-benchmark/tree/main/data/interpretability_examples) and by running the following command. You can also feed multiple documents and queries at once to generate multiple similarity maps.

```bash
generate-similarity-maps \
    --documents "data/interpretability_examples/energy_electricity_generation.jpeg" \
    --queries "Which hour of the day had the highest overall electricity generation in 2019?" \
    --documents "data/interpretability_examples/shift_kazakhstan.jpg" \
    --queries "Quelle partie de la production pétrolière du Kazakhstan provient de champs en mer ?"
```

> [!WARNING]
> The current version of `vidore-benchmark` uses a different ColPali checkpoint than the one used in the paper. As a result, the similarity maps may differ slightly from the ones presented in the paper. If you want to reproduce the exact similarity maps from the paper, you should use the `vidore/colpali` checkpoint along with `vidore-benchmark<=3.3.0`.

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
