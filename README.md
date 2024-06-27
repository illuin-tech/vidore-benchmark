# Vision Document Retrieval (ViDoRe): Benchmark

[[ColPali Model card]](https://huggingface.co/vidore/colpali)
[[ColPali Training]](https://github.com/ManuelFay/retriever-training)
[[ViDoRe Benchmark]](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d)
[[ViDoRe Leaderboard]](https://huggingface.co/spaces/vidore/vidore-leaderboard)
<!-- [[Paper]]() -->
<!-- [[Hf Blog]]() -->
<!-- [[Hf Space]]() -->
<!-- [[Colab Example]]() -->

## Approach

The Visual Document Retrieval Benchmark (ViDoRe), is introduced to evaluate and enhance the performance of document retrieval systems on visually rich documents across various tasks, domains, languages, and settings. It was used to evaluate the ColPali model, a VLM-powered retriever that efficiently retrieves documents based on their visual content and textual queries.

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

Finally, if you are willing to reproduce the results from the ColPali paper, you can use the `requirements.txt` file to install the dependencies used at test time:

```bash
pip install -r requirements.txt
```

## Available retrievers

The list of available retrievers can be found [here](https://github.com/tonywu71/vidore-benchmark/tree/main/src/vidore_benchmark/retrievers). Read [this section](###Implement-your-own-retriever) to learn how to create, use, and evaluate your own retriever.

## Command-line usage

### Evaluate a retriever on ViDoRE

You can evaluate any off-the-shelf retriever on the ViDoRe benchmark. For instance, you
can evaluate the ColPali model on the ViDoRe benchmark to reproduce the results from our paper.

```bash
vidore-benchmark evaluate-retriever \
    --model-name vidore/colpali \
    --collection-name "vidore/vidore-**benchmark**-667173f98e70a1c0fa4db00d" \
    --split test
```

**Note:** You should get a warning about some non-initialized weights. This is a known issue in ColPali and will
cause the metrics to be slightly different from the ones reported in the paper. We are working on fixing this issue.

Alternatively, you can evaluate your model on a single dataset:

```bash
vidore-benchmark evaluate-retriever \
    --model-name vidore/colpali \
    --dataset-name vidore/syntheticDocQA_dummy
```

### Retrieve the top-k documents from a HuggingFace dataset

```bash
vidore-benchmark retrieve-on-dataset \
    --model-name BAAI/bge-m3 \
    --query "Which hour of the day had the highest overall electricity generation in 2019?" \
    --k 5 \
    --dataset-name vidore/syntheticDocQA_energy_test \
    --split test
```

### Retrieve the top-k documents from a collection of PDF documents

```bash
vidore-benchmark retrieve-on-dataset \
    --model-name vidore/colpali \
    --query "How did the average miles per shipment for single modes change from 1997 to 2007?" \
    --k 2 \
    --dataset-name vidore/syntheticDocQA_dummy \
    --split test
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

If you need to evaluate your own model on the ViDoRe benchmark, you can create your own instance of `VisionRetriever` to use it with the evaluation scripts in this package. You can find the detailed instructions [here](https://github.com/tonywu71/vidore-benchmark/blob/main/src/vidore_benchmark/retrievers/README.md).

### Show the similarity maps for interpretability

By superimposing the late interaction heatmap on top of the original image, we can visualize the most salient image patches with respect to each term of the query, yielding interpretable insights into model focus zones.

You can generate similarity maps using the `generate-similarity-maps`. For instance, you can reproduce the similarity maps from the paper by downloading the images from the [`data/interpretability_examples`](https://github.com/tonywu71/vidore-benchmark/tree/main/data/interpretability_examples) folder and running the following command:

```bash
generate-similarity-maps \
    --documents "data/interpretability_examples/energy_electricity_generation.jpeg" \
    --queries "Which hour of the day had the highest overall electricity generation in 2019?" \
    --documents "data/interpretability_examples/shift_kazakhstan.jpg" \
    --queries "Quelle partie de la production pétrolière du Kazakhstan provient de champs en mer ?"
```

## Citation

**ColPali: Efficient Document Retrieval with Vision Language Models**  
First authors: Manuel Faysse, Hugues Sibille, Tony Wu  
Contributors: Bilel Omrani, Gautier Viaud, CELINE HUDELOT, Pierre Colombo
