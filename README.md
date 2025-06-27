# Jina VDR (Jina Visual Document Retrieval)

[![arXiv](https://img.shields.io/badge/arXiv-2506.18902-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2506.18902)
[![GitHub](https://img.shields.io/badge/Jina%20VDR-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/jina-ai/jina-vdr)
[![Hugging Face](https://img.shields.io/badge/Jina%20VDR%20Collecttion-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/jinaai/jinavdr-visual-document-retrieval-684831c022c53b21c313b449)
---

## Scope of this Benchmark

Jina VDR is a multilingual, multi-domain benchmark for visual document retrieval.
In contrast to other VDR  benchmarks that focus on question answering and OCR-related tasks, it has an expanded scope of visual document benchmarking:
more query types, a much more diverse array of materials (e.g. maps, markdown documents, advertisements, scans of historical documents), and includes tasks available in many languages.

The code is based on the [vidore-benchmark](https://github.com/illuin-tech/vidore-benchmark) code.

## How to run the Vidore Evaluation

To evaluate models on the Jina VDR benchmark, install the requirements corresponding to your selected model:

### Installation


```bash
# For Jina Embeddings v4
pip install ".[jina-v4]"

# For Jina Embeddings v3
pip install ".[jina-v3]"

# For BM25
pip install ".[bm25]"
# Additionally install stopwords if you haven't already
python -c "import nltk; nltk.download('stopwords')"

# For Jina CLIP
pip install ".[jina-clip]"

# For Colpali
pip install ".[colpali-engine]"

# For DSE-Qwen2b
pip install ".[dse]"
```

### Running Evaluation

Use the base command below with the appropriate model configuration:

```bash
vidore-benchmark evaluate-retriever \
    --model-class <MODEL_CLASS> \
    --model-name <MODEL_NAME> \
    --collection-name jinaai/jinavdr-visual-document-retrieval-684831c022c53b21c313b449 \
    --dataset-format qa \
    --split test
```

### Model Configurations
| Model              | --model-class | --model-name                |
|--------------------|---------------|-----------------------------|
| Jina Embeddings v4 | jev4          | jinaai/jina-embeddings-v4   |
| Jina Embeddings v3 | jev3          | jinaai/jina-embeddings-v3   |
| BM25               | bm25          | bm25                        |
| Jina CLIP          | jina-clip     | jinaai/jina-clip-v2         |
| Colpali            | colpali       | vidore/colpali-v1.2         |
| DSE-Qwen2-2b       | dse-qwen2     | MrLight/dse-qwen2-2b-mrl-v1 |

#### Example for Jina Embeddings v4:
```bash
# For single vector
vidore-benchmark evaluate-retriever \
    --model-class jev4 \
    --model-name jinaai/jina-embeddings-v4 \
    --collection-name jinaai/jinavdr-visual-document-retrieval-684831c022c53b21c313b449 \
    --dataset-format qa \
    --languages ar,bn,de,en,es,fr,hi,hu,id,it,jp,ko,my,nl,pt,ru,th,ur,vi,zh
    --split test
    
# For multi vector"
vidore-benchmark evaluate-retriever \
    --model-class jev4 \
    --model-name jinaai/jina-embeddings-v4 \
    --collection-name jinaai/jinavdr-visual-document-retrieval-684831c022c53b21c313b449 \
    --dataset-format qa \
    --split test
    --languages ar,bn,de,en,es,fr,hi,hu,id,it,jp,ko,my,nl,pt,ru,th,ur,vi,zh
    --vector-type multi_vector
```

## Overview of the Dataset Collection

| Dataset Name | Domain | Document Format | Query Format | Number of Queries / Documents | Languages |
|---|---|---|---|---|---|
| jinaai/airbnb-synthetic-retrieval† | Housing | Tables | Instruction | 4953 / 10000 | ar, de, en, es, fr, hi, hu, ja ru, zh |
| jinaai/arabic_chartqa_ar | Mixed | Charts | Question | 745 / 745 | ar |
| jinaai/arabic_infographicsvqa_ar | Mixed | Illustrations | Question | 120 / 40 | ar |
| jinaai/automobile_catalogue_jp | Marketing | Catalog | Question | 45 / 15 | ja |
| jinaai/arxivqa | Science | Mixed | Question | 30 / 499 | en |
| jinaai/beverages_catalogue_ru | Marketing | Digital Docs | Question | 100 / 34 | ru |
| jinaai/ChartQA | Mixed | Charts | Question | 7996 / 1000 | en |
| jinaai/CharXiv-en | Science | Charts | Question | 999 / 1000 | en |
| jinaai/docvqa | Mixed | Scans | Question | 39 / 499 | en |
| jinaai/donut_vqa | Medical | Scans / Handwriting | Question | 704 / 800 | en |
| jinaai/docqa_artificial_intelligence | Software / IT | Digital Docs | Question | 70 / 962 | en |
| jinaai/docqa_energy | Energy | Digital Docs | Question | 69 / 972 | en |
| jinaai/docqa_gov_report | Government | Digital Docs | Question | 77 / 970 | en |
| jinaai/docqa_healthcare_industry | Medical | Digital Docs | Question | 90 / 963 | en |
| jinaai/europeana-de-news | Historic | Scans / News Articles | Question | 379 / 137 | de |
| jinaai/europeana-es-news | Historic | Scans / News Articles | Question | 474 / 179 | es |
| jinaai/europeana-fr-news | Historic | Scans / News Articles | Question | 237 / 145 | fr |
| jinaai/europeana-it-scans | Historic | Scans | Question | 618 / 265 | it |
| jinaai/europeana-nl-legal | Legal | Scans | Question | 199 / 300 | nl |
| jinaai/github-readme-retrieval-multilingual† | Software / IT | Markdown Docs | Description | 16755 / 4398 | ar, bn, de, en, es, fr, hi, id, it, ja, ko, nl pt, ru, th, vi, zh |
| jinaai/hindi-gov-vqa | Governmental | Digital Docs | Question | 454 / 340 | hi |
| jinaai/hungarian_doc_qa_hu | Mixed | Digital Docs | Question | 54 / 54 | hu |
| jinaai/infovqa | Mixed | Illustrations | Question | 363 / 500 | en |
| jinaai/jdocqa | News | Digital Docs | Question | 744 / 758 | ja |
| jinaai/jina_2024_yearly_book | Software / IT | Digital Docs | Question | 75 / 33 | en |
| jinaai/medical-prescriptions | Medical | Digital Docs | Question | 100 / 100 | en |
| jinaai/mpmqa-small | Manuals | Digital Docs | Question | 155 / 782 | en |
| jinaai/MMTab | Mixed | Tables | Fact | 987 / 906 | en |
| jinaai/openai-news | Software / IT | Digital Docs | Question | 31 / 30 | en |
| jinaai/owid_charts_en | Mixed | Charts | Question | 132 / 972 | en |
| jinaai/plotqa | Mixed | Charts | Question | 610 / 986 | en |
| jinaai/ramen_benchmark_jp | Marketing | Catalog | Question | 29 / 10 | ja |
| jinaai/shanghai_master_plan | Governmental | Digital Docs | Question / Key Phrase | 57 / 23 | zh, en |
| jinaai/wikimedia-commons-documents-ml† | Mixed | Mixed | Description | 14061 / 14661 | ar, bn, de, en, es, fr, hi, hu, id, it, ja, ko, my, nl, pt, ru, th, ur, vi, zh |
| jinaai/shiftproject | Environmental Documents | Digital Docs | Question | 89 / 998 | fr |
| jinaai/stanford_slide | Education | Slides | Question | 14 / 1000 | en |
| jinaai/student-enrollment | Demographics | Charts | Question | 1000 / 489 | en |
| jinaai/tabfquad | Mixed | Tables | Question | 126 / 70 | fr, en |
| jinaai/table-vqa | Science | Tables | Question | 992 / 1000 | en |
| jinaai/tatqa | Finance | Digital Docs | Question | 121 / 176 | en |
| jinaai/tqa | Education | Illustrations | Question | 981 / 394 | en |
| jinaai/tweet-stock-synthetic-retrieval† | Finance | Charts | Question | 6278 / 10000 | ar, de, en, es, hi, hu, ja, ru, zh |
| jinaai/wikimedia-commons-maps | Mixed | Maps | Description | 443 / 455 | en |

---


## Citation

**jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval**  

```latex
@misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      title={jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval}, 
      author={Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Sedigheh Eslami and Scott Martens and Bo Wang and Nan Wang and Han Xiao},
      year={2025},
      eprint={2506.18902},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.18902}, 
}
```
