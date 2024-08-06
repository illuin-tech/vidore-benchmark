# Token Pooling: Impact of Pool Factor on Retrieval

**Author:** Tony Wu

**Creation date:** Tue, 6 Aug 2024 10:21:00 GMT+2

## Introduction

ColPali achieves strong retrieval performances on document retrieval tasks (see the ViDoRe benchmark).

However, your multi-vector embeddings scale linearly in size withe number of tokens! A direct consequence of having larger embeddings is longer indexing and search time. This is where indexing helps: we will approximate our embeddings to make everything go faster. To improve ColBERT, researchers have experimented with ColBERTv2  and Performance-optimized Late Interaction Driver (PLAID) and managed to use centroid and compression tricks to build a multi-step retrieval pipeline that is both efficient and accurate. While PLAID seems great on paper, because it’s centroid-based, it’s not tractable to create (C), update (U), nor to delete (D) a document from the vector store (or else, how would you update the centroids after each operation?). Hierarchical Navigable Small Worlds (HNSW) is fully CRUD but doesn’t scale well.

In his blog ["A little pooling goes a long way for multi-vector representations"](https://www.answer.ai/posts/colbert-pooling.html), Benjamin Clavié proposes a simple but effective to a similar problem with ColBERT: to simply reduce the sequence length of the multi-vector embedding using pooling. Thus our problematic:

> Can we use token pooling to reduce the size of our embeddings and improve the retrieval performance and latency of ColPali?

## Objectives

- (1) Investigate the impact of the pool factor on the retrieval performance and latency of the model.
- (2) Find the optimal pool factor for ColPali.
- (3) Interpretate the clustering of the embeddings.

## Assumptions

- For simplicity, we will only evaluate ColPali on DocVQA with the `vidore/docvqa_test_subsampled` dataset. We will assume that the results will generalize to other datasets.
- Because 

## Methodology

Notes:
- ColPali outputs document embeddings with length 32*32=1024 patches + 6 memory tokens = 1030 tokens.

To run/reproduce the experiments, follow these steps:

1. Checkout to the branch `2024-08-06_impact_of_pool_factor_on_retrieval`

```bash
python experiments/2024-08-06_impact_of_pool_factor_on_retrieval/main.py \
    --model-name vidore/colpali \
    --pool-factors 1 \
    --pool-factors 2 \
    --pool-factors 3 \
    --pool-factors 4 \
    --pool-factors 5 \
    --pool-factors 6 \
    --pool-factors 7 \
    --pool-factors 8 \
    --pool-factors 9 \
    --pool-factors 10 \
    --pool-factors 20 \
    --pool-factors 30 \
    --dataset-name vidore/docvqa_test_subsampled \
    --split test

python experiments/2024-08-06_impact_of_pool_factor_on_retrieval/main.py \
    --model-name vidore/colpali \
    --pool-factors 1 \
    --pool-factors 2 \
    --pool-factors 3 \
    --pool-factors 4 \
    --pool-factors 5 \
    --pool-factors 6 \
    --pool-factors 7 \
    --pool-factors 8 \
    --pool-factors 9 \
    --pool-factors 10 \
    --pool-factors 20 \
    --pool-factors 30 \
    --dataset-name vidore/syntheticDocQA_energy_test \
    --split test

python experiments/2024-08-06_impact_of_pool_factor_on_retrieval/main.py \
    --model-name vidore/colpali \
    --pool-factors 1 \
    --pool-factors 2 \
    --pool-factors 3 \
    --pool-factors 4 \
    --pool-factors 5 \
    --pool-factors 6 \
    --pool-factors 7 \
    --pool-factors 8 \
    --pool-factors 9 \
    --pool-factors 10 \
    --pool-factors 20 \
    --pool-factors 30 \
    --dataset-name vidore/shiftproject_test \
    --split test
```

## Results

### Impact of Pool Factor on Retrieval Performance

### Interpretation of the clustering of the document embeddings

There are no clear clusters in the document embeddings.

## Discussion
