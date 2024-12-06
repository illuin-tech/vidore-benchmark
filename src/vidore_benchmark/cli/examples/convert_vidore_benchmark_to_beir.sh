#!/bin/bash
set -e  # Exit the script if any command fails

# Instructions:
#
# Run this script to convert all the ViDoRe benchmark datasets to the BEIR format.
#
# ```bash
# bash src/vidore_benchmark/cli/convert_vidore_benchmark_to_beir.sh > convert_vidore_benchmark_to_beir.log 2>&1 &
# ```

# List of source datasets
datasets=(
    "vidore/arxivqa_test_subsampled"
    "vidore/docvqa_test_subsampled"
    "vidore/infovqa_test_subsampled"
    "vidore/tabfquad_test_subsampled"
    "vidore/tatdqa_test"
    "vidore/shiftproject_test"
    "vidore/syntheticDocQA_artificial_intelligence_test"
    "vidore/syntheticDocQA_energy_test"
    "vidore/syntheticDocQA_government_reports_test"
    "vidore/syntheticDocQA_healthcare_industry_test"
)

# Iterate over each dataset and run the conversion script
for dataset in "${datasets[@]}"; do
    echo "=================   Converting dataset: $dataset   ================="
    python src/vidore_benchmark/cli/convert_ds_to_beir.py --source-dataset "$dataset"
done
