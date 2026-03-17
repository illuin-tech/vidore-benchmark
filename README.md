# ViDoRe Pipeline Evaluation Framework 🔍

[![arXiv](https://img.shields.io/badge/arXiv-2407.01449-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2407.01449)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/illuin-tech/vidore-benchmark)
[![Hugging Face](https://img.shields.io/badge/Vidore_Hf_Space-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/vidore)

[![Test](https://github.com/illuin-tech/vidore-benchmark/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/illuin-tech/vidore-benchmark/actions/workflows/test.yml)
[![Version](https://img.shields.io/pypi/v/vidore-benchmark?color=%2334D058&label=pypi%20package)](https://pypi.org/project/vidore-benchmark/)
[![Downloads](https://static.pepy.tech/badge/vidore-benchmark)](https://pepy.tech/project/vidore-benchmark)

---

> [!IMPORTANT]
> ## 🔄 Repository Focus Change
> 
> This repository is now focused on **pipeline evaluation** for visual document retrieval tasks. 
> 
> All other functionalities (vision retriever evaluation, legacy benchmarks) are kept for reproducibility purposes but are **deprecated and no longer actively maintained**.

---

## Evaluating single-model retrievers on ViDoRe v1–v3 with MTEB
We shifted from in-house evaluations to the general MTEB evaluation framework for retrieval models by moving to [MTEB](https://github.com/embeddings-benchmark/mteb/tree/main).

Here are the main steps to evaluate and submit your retriever to the ViDoRe V1-V3 leaderboards ; see the [MTEB official documentation](https://embeddings-benchmark.github.io/mteb/contributing/adding_a_model/) for full details. This section covers mteb leaderboards only; for our in-house pipeline leaderboard, see the section below.

1. Create your model implementation file (if it does not exist already) [here](https://github.com/embeddings-benchmark/mteb/tree/main/mteb/models/model_implementations), then open a PR to the [MTEB repository](https://github.com/embeddings-benchmark/mteb) with your changes; examples for Colpali-like models can be found in [this file](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/models/model_implementations/colpali_models.py).

2. Evaluate your model:
```python
import mteb
from mteb.models.model_implementations.my_custom_model import MyCustomModel

my_model = MyCustomModel(my_args)
tasks = mteb.get_tasks(["ViDoRe (v3)"])

results = mteb.evaluate(my_model, tasks=tasks)
```

3. Open a PR on the [mteb_results_repo](https://github.com/embeddings-benchmark/results/tree/main) with the generated results file to submit your results to the leaderboard

4. To evaluate on private sets, once all this is done you can ask the MTEB team to evaluate your model on private ViDoRe v3 sets by opening a dedicated issue on [their repo](https://github.com/embeddings-benchmark/mteb/issues)

## Evaluating a complex pipeline


Pipeline evaluation allows you to evaluate **complete end-to-end retrieval systems** on the ViDoRe v3 benchmark datasets. Unlike traditional retriever evaluation that focuses on individual model components, pipeline evaluation lets you test:

- **Multi-stage retrieval systems** (e.g., retrieve + rerank)
- **Hybrid approaches** (e.g., dense + sparse retrieval fusion)
- **Custom preprocessing pipelines** (e.g., OCR → chunking → embedding)
- **Arbitrary retrieval logic** that goes beyond standard dense/sparse retrievers

### 📊 Results Repository & Submission Guidelines

**This repository serves as the primary community results repository for visual document retrieval benchmarks using complex pipelines.** We encourage researchers and practitioners to submit their pipeline evaluation results to create a centralized location where the community can compare different approaches and track progress on ViDoRe v3 datasets.

### How to Submit Your Results

To contribute your pipeline results to the leaderboard:

1. **Run evaluations** using this framework on the ViDoRe v3 datasets **english splits** (`--language english` in cli). It tracks raw scores as well as indexing and search computing times.
2. **Open a Pull Request** with the following:
   - **Results files**: Add your JSON result files to the `results/metrics` folder, organized as:
     ```
     results/metrics/your_pipeline_name/
     ├── vidore_v3_hr.json
     ├── vidore_v3_finance_en.json
     ├── vidore_v3_industrial.json
     └── ... (other datasets)
     ```
   - **Pipeline description**: Include a `description.json` file in the same PR that describes the architecture used. A pipeline is represented as a graph of a set of modules (OCR, retriever, reranker, mcp server... linked together via edges)
     Some pipeline descriptions files example are written in `results/pipeline_descriptions`

    We encourage adding as much hardware information as possible in the description to enable the community to get a feel about the latency of each pipeline.

### Installation

```bash
pip install vidore-benchmark
```

### List Available Datasets

List all ViDoRe v3 datasets:

```bash
vidore-benchmark pipeline list-datasets
```

Available datasets:
- `vidore/vidore_v3_hr` - Human Resources documents
- `vidore/vidore_v3_finance_en` - Financial documents (English)
- `vidore/vidore_v3_industrial` - Industrial documents
- `vidore/vidore_v3_pharmaceuticals` - Pharmaceutical documents
- `vidore/vidore_v3_computer_science` - Computer Science documents
- `vidore/vidore_v3_energy` - Energy sector documents
- `vidore/vidore_v3_physics` - Physics documents
- `vidore/vidore_v3_finance_fr` - Financial documents (French)

### Evaluate a Pipeline

You can evaluate any pipeline that inherits from `BasePipeline`:

Some pipelines are already implemented in the `pipeline_implementations` folder.

## Custom Pipeline

Evaluate your own pipeline implementation:

```bash
vidore-benchmark pipeline evaluate \
    --dataset-name vidore/vidore_v3_hr \
    --module-path path/to/my_pipeline.py \
    --class-name MyCustomPipeline \
    --language english \
    --pipeline-args '{"model_name": "my-model"}'
```

**Your pipeline file** (`my_pipeline.py`):
```python
from vidore_benchmark.pipeline_evaluation import BasePipeline

class MyCustomPipeline(BasePipeline):
    def __init__(self, model_name):
        self.model_name = model_name
        # Initialize your model here

    def index(self, corpus_ids, corpus_images, corpus_texts, dataset_name: str = None):
        # Indexing function to process corpus, should store anything
        # relevant as class attributes
        self.corpus_ids = corpus_ids
        ...

    def search(self, query_ids, queries):
        # Your search logic, returns scores dict (see BasePipeline file for description)
        return {query_id: {corpus_id: score}}
```

### Language Filtering

Some datasets contain multilingual queries. You can filter by language:

```bash
vidore-benchmark pipeline evaluate \
    --dataset-name vidore/vidore_v3_hr \
    --pipeline-type random \
    --language english
```

### Evaluate on All Datasets

Evaluate your pipeline on all ViDoRe v3 datasets:

**With built-in pipeline:**
```bash
vidore-benchmark pipeline evaluate-all \
    --pipeline-type random \
    --pipeline-args '{"seed": 42}' \
    --output-dir results/
```

**With custom pipeline:**
```bash
vidore-benchmark pipeline evaluate-all \
    --module-path my_pipeline.py \
    --class-name MyCustomPipeline \
    --output-dir results/
```

## Python API

### Implementing Your Own Pipeline

To evaluate a custom pipeline, inherit from `BasePipeline` and implement the `index()` and `search()` methods:

### Running Evaluation

```python
from path_to_pipeline import MyCustomPipeline
from vidore_benchmark.pipeline_evaluation import (
    load_vidore_dataset,
    evaluate_retrieval,
    aggregate_results,
)

# Load dataset
query_ids, queries, corpus_ids, corpus_images, corpus_texts, qrels = load_vidore_dataset(
    dataset_name="vidore/vidore_v3_hr",
    split="test"
)

# Initialize your pipeline
pipeline = MyCustomPipeline(retriever=my_retriever, reranker=my_reranker)

# Run evaluation
results = evaluate_retrieval(
    pipeline=pipeline,
    query_ids=query_ids,
    queries=queries,
    corpus_ids=corpus_ids,
    corpus_images=corpus_images,
    corpus_texts=corpus_texts,
    qrels=qrels,
    metrics=["ndcg_cut_10", "recall_10"]
)

# Get aggregate scores
aggregated = aggregate_results(results)
print(f"NDCG@10: {aggregated['ndcg_cut_10']:.4f}")
```

Some examples of pipeline implementations can be found in the `pipeline_implementations` folder

## Advanced Usage

### Tracking Additional Metrics (Optional)

Pipelines can optionally return additional tracking information alongside retrieval results. This is useful for monitoring costs, timing, resource usage, or other custom metrics:

```python
from typing import Dict, List, Any, Optional, Tuple

class PipelineWithMetrics(BasePipeline):
    def index(
        self,
        corpus_ids: List[str],
        corpus_images: List[Any],
        corpus_texts: List[str],
    ) -> None:
        # Indexing logic
        ...

    def search(
        self,
        query_ids: List[str],
        queries: List[str],
    ) -> Tuple[Dict[str, Dict[str, float]], Optional[Dict[str, Any]]]:
        """
        Return both retrieval results and optional tracking metrics.
        
        Returns:
            Tuple of (results, infos) where infos can contain:
            - Cost tracking (e.g., API costs, GPU hours)
            - Granular timing information
            - Resource usage (num_gpus, memory, etc.)
            - Model-specific metadata
        """
        # Your retrieval logic here
        results = {...}
        
        # Optional: track additional metrics
        infos = {
            "estimated_cost_usd": 0.05,
            "num_gpus": 1,
            "total_time_ms": 1234.5,
            "model_name": "my-model-v1",
        }
        
        return results, infos
```

The `infos` dictionary will be stored in the evaluation results under the `_infos` key. This is completely **optional** - pipelines can still return just the results dictionary for backward compatibility:

```python
class SimplePipeline(BasePipeline):
    def search(...) -> Dict[str, Dict[str, float]]:
        # Just return results, no tracking needed
        return results
```

See [`example_pipelines/pipeline_with_metrics.py`](example_pipelines/pipeline_with_metrics.py) for a complete example.

### Dataset Information

```python
from vidore_benchmark.pipeline_evaluation import (
    load_vidore_dataset,
    print_dataset_info,
    get_available_datasets,
)

# List available datasets
datasets = get_available_datasets()
print(datasets)

# Load and inspect a dataset
query_ids, queries, corpus_ids, corpus, qrels = load_vidore_dataset(
    "vidore/vidore_v3_industrial"
)

print_dataset_info(
    dataset_name="vidore/vidore_v3_industrial",
    query_ids=query_ids,
    queries=queries,
    corpus_ids=corpus_ids,
    corpus=corpus,
    qrels=qrels,
)
```

### Custom Metrics

You can specify custom metrics to evaluate if you want to:

```python
results = evaluate_retrieval(
    pipeline=pipeline,
    query_ids=query_ids,
    queries=queries,
    corpus_ids=corpus_ids,
    corpus=corpus,
    qrels=qrels,
    metrics=[
        "ndcg_cut_5",
        "ndcg_cut_10",
        "recall_5",
        "recall_10",
        "map",
    ]
)
```

All metrics supported by [pytrec_eval](https://github.com/cvangysel/pytrec_eval) are available.

## Architecture

The pipeline evaluation framework consists of:

1. **`BasePipeline`**: Abstract base class for implementing custom pipelines
2. **Dataset Loaders**: Functions to load ViDoRe v3 datasets from HuggingFace
3. **Evaluator**: Uses `pytrec_eval` to compute retrieval metrics
4. **CLI**: Commands for evaluating any custom pipeline

```
vidore_benchmark/
├── pipeline_evaluation/
│   ├── base_pipeline.py          # BasePipeline abstract class
│   ├── dataset_loader.py          # ViDoRe v3 dataset loading
│   ├── evaluator.py               # Evaluation orchestration
│   ├── utils.py                   # Helper utilities
└── cli/
    └── pipeline_evaluation.py     # CLI for pipeline evaluation
```

## Reproducibility & Legacy Features

This repository previously focused on evaluating vision retrievers on the ViDoRe benchmarks v1 and v2. All code related to these functionalities is **still available but deprecated**:

- **Vision Retriever Evaluation**: See [`README_OLD.md`](README_OLD.md)
- **ViDoRe Benchmarks v1/v2**: Now maintained in [MTEB](https://github.com/embeddings-benchmark/mteb)
- **Model Implementations**: Available in `src/vidore_benchmark/retrievers/` (for reference only)

**⚠️ For new projects**, we recommend:
- Using **MTEB** for vision retriever evaluation on ViDoRe v1/v2
- Using **this framework** for pipeline evaluation on ViDoRe v3

For reproducibility of published results, see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Contributing

We welcome contributions for:
- New example pipelines
- Additional evaluation results
- Dataset utilities
- Documentation improvements

Please open an issue or PR on [GitHub](https://github.com/illuin-tech/vidore-benchmark).

## Citation

If you use this framework or the ViDoRe benchmark in your research, please cite:

**ColPali: Efficient Document Retrieval with Vision Language Models**

```bibtex
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

**ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval**

```bibtex
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

**ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios**

```bibtex
@misc{loison2026vidore,
      title={ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
      author={Loison, Ant{\'o}nio and Mac{\'e}, Quentin and Edy, Antoine and Xing, Victor and Balough, Tom and Moreira, Gabriel and Liu, Bo and Faysse, Manuel and Hudelot, C{\'e}line and Viaud, Gautier},
      journal={arXiv preprint arXiv:2601.08620},
      year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Links

- [[ViDoRe Leaderboard]](https://huggingface.co/spaces/vidore/vidore-leaderboard)
- [[ColPali Model Card]](https://huggingface.co/vidore/colpali)
- [[ColPali Engine]](https://github.com/illuin-tech/colpali)
- [[Blog Post]](https://huggingface.co/blog/manu/colpali)
- [[Demo]](https://huggingface.co/spaces/manu/ColPali-demo)
