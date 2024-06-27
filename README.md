# Vision Document Retrieval (ViDoRe): Benchmark

<!-- [[Paper]]() -->
[[ColPali Model card]](https://huggingface.co/vidore/colpali)
[[ColPali training]](https://github.com/ManuelFay/retriever-training)
[[ViDoRe Benchmark]](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d)
<!-- [[Hf Blog]]() -->
<!-- [[Hf Leaderboard]]() -->
<!-- [[Hf Space]]() -->
<!-- [[Colab example]]() -->

## Approach

The Visual Document Retrieval Benchmark (ViDoRe), is introduced to evaluate and enhance the performance of document retrieval systems on visually rich documents across various tasks, domains, languages, and settings. It was used to evaluate the ColPali model, a VLM-powered retriever that efficiently retrieves documents based on their visual content and textual queries.

## Setup

We used Python 3.11.6 and PyTorch 2.2.2 to train and test our models, but the codebase is expected to be compatible with Python >=3.9 and recent PyTorch versions.

The eval codebase depends on a few Python packages, which can be downloaded using the following command:

```bash
pip install -U vidore-benchmark
```

Alternatively, the following command will pull and install the latest commit from this repository, along with its Python dependencies:

```bash
pip install --upgrade --no-deps --force-reinstall git+https://github.com/tonywu71/vidore-benchmark.git
```

To keep a lightweight repostiory, only the essential packages were installed. In particular, you will need to specify the dependencies for the specific non-Transformers models you want to run (see the list in `pyproject.toml`). For instance, if you want to evaluate BGE-M3:

```bash
pip install -U "vidore-benchmark[bge-m3]"
```

Or if you want to evaluate all the off-the-shelf retrievers:

```bash
pip install -U "vidore-benchmark[all-retrievers]"
```

Similarly, if you need to run the interpretability scripts, you should also run:

```bash
pip install -U "vidore-benchmark[interpretability]"
```

Finally, if you are willing to reproduce the results from the ColPali paper, you can download the `requirements.txt` file to install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## Available retrievers

The list of available retrievers can be found [here](https://github.com/tonywu71/vidore-benchmark/tree/main/src/vidore_benchmark/retrievers). Read [this section](###Implement-your-own-retriever) to learn how to create, use, and evaluate your own retriever.

## Command-line usage

### Evaluate a retriever on ViDoRE

To evaluate an off-the-shelf retriever on the ViDoRe benchmark:

```bash
python scripts/evaluate_retriever.py \
    --model-name {{hf_model_name}} \
    --collection-name vidore/vidore-**benchmark**-667173f98e70a1c0fa4db00d \
    --split test
```

Alternatively, you can evaluate your model on a single dataset:

```bash
python scripts/evaluate_retriever.py \
    --model-name {{hf_model_name}} \
    --dataset-name vidore/syntheticDocQA_dummy
```

### Retrieve the top-k documents from a HuggingFace dataset

```bash
python scripts/retrieve_on_dataset.py \
    --model-name BAAI/bge-m3 \
    --query "Which hour of the day had the highest overall electricity generation in 2019?" \
    --k 5 \
    --dataset-name vidore/syntheticDocQA_energy_test \
    --split test
```

### Retrieve the top-k documents from a collection of PDF documents

```bash
    python scripts/retriever_on_pdfs.py \
    --model-name google/siglip-so400m-patch14-384 \
    --query "Which hour of the day had the highest overall electricity generation in 2019?" \
    --k 5 \
    --data-dirpath data/my_folder_with_pdf_documents/ \
```

### Reproduce the baselines

Run the following command to reproduce the results from the ColPali paper:

```bash
python scripts/evaluate_retriever.py \
    --split test \
    --collection-name coldoc/vidore-chunk-ocr-baseline-666acce88c294ef415548a56 \
    --model-name vidore/colpali-3b-mix-448
```

<!-- ### Documentation

All scripts can be found in the `scripts/` directory. To get help on a specific script, run:

```bash
python scripts/{{script_name}} --help
``` -->

## Python usage

### Implement your own retriever

Read the instructions [here](https://github.com/tonywu71/vidore-benchmark/blob/main/src/vidore_benchmark/retrievers/README.md).

### Show the similarity maps for interpretability

By superimposing the late interaction heatmap on top of the original image, we can visualize the most salient image patches with respect to each term of the query, yielding interpretable insights into model focus zones.

To generate the similarity maps for a given document-query pair:

```python
python scripts/generate_similarity_maps.py \
    --documents "data/interpretability_examples/energy_electricity_generation.jpeg" \
    --queries "Which hour of the day had the highest overall electricity generation in 2019?" \
    --documents "data/interpretability_examples/shift_kazakhstan.jpg" \
    --queries "Quelle partie de la production pétrolière du Kazakhstan provient de champs en mer ?"
```

## Citation

**ColPali: Efficient Document Retrieval with Vision Language Models**  
First authors: Manuel Faysse, Hugues Sibille, Tony Wu  
Contributors: Bilel Omrani, Gautier Viaud, CELINE HUDELOT, Pierre Colombo
