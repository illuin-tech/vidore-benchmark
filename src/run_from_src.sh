python -m vidore_benchmark.cli.main evaluate-retriever \
    --model-class llava-onevision \
    --model-name llava-hf/llava-onevision-qwen2-0.5b-ov-hf \
    --dataset-name vidore/docvqa_test_subsampled \
    --dataset-format qa \
    --split test 
    # --batch-query 4 \
    # --batch-passage 4 \
    # --batch-score 4
    # --model-name test_retriever \

# vidore/vidore-benchmark-667173f98e70a1c0fa4db00d
