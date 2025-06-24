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

# How to run the Vidore Evaluation

If you want to run the vidore evaluation on the jina-embeddings-v4 model (and on the Document Retrieval Benchmark curated by Jina AI), you need to install requirements:

```
pip install ".[jina-v4]"
```

Then, you can run the evaluation with the following command:

```
vidore-benchmark evaluate-retriever \
    --model-class jev4 \
    --model-name jinaai/jina-embeddings-v4 \
    --collection-name jinaai/jinavdr-visual-document-retrieval-684831c022c53b21c313b449 \
    --dataset-format qa \
    --split test
```



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
