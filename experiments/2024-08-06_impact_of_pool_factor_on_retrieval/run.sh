#!/bin/bash

# Array of dataset names from "vidore/vidore-benchmark-667173f98e70a1c0fa4db00d"
datasets=(
    "vidore/arxivqa_test_subsampled"
    "vidore/docvqa_test_subsampled"
    "vidore/infovqa_test_subsampled"
    "vidore/tabfquad_test_subsampled"
    # "vidore/tatdqa_test"
    "vidore/shiftproject_test"
    # "vidore/syntheticDocQA_artificial_intelligence_test"
    "vidore/syntheticDocQA_energy_test"
    # "vidore/syntheticDocQA_government_reports_test"
    # "vidore/syntheticDocQA_healthcare_industry_test"
)

# Loop through each dataset
for dataset in "${datasets[@]}"; do
    python experiments/2024-08-06_impact_of_pool_factor_on_retrieval/main.py \
        --model-name vidore/colpali \
        --pool-factors 1 \
        --pool-factors 2 \
        --pool-factors 3 \
        --pool-factors 4 \
        --pool-factors 5 \
        --pool-factors 6 \
        --pool-factors 7 \
        --dataset-name "$dataset" \
        --split test
done
