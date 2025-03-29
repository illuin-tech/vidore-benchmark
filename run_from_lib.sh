vidore-benchmark evaluate-retriever \
    --model-class test_retriever \
    --model-name test_retriever \
    --dataset-name vidore/docvqa_test_subsampled \
    --dataset-format qa \
    --split test


# Available models: ['bge-m3-colbert', 'bge-m3', 'biqwen2', 'bm25', 'cohere', 'colidefics3', 'colpali', 
#                    'colqwen2', 'dse-qwen2', 'dummy_vision_retriever', 'jina-clip-v1', 'nomic-embed-vision', 'siglip']