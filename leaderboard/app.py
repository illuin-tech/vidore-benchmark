import json
import os

import gradio as gr
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.repocard import metadata_load


def make_clickable_model(model_name, link=None):
    if link is None:
        link = "https://huggingface.co/" + model_name
    # Remove user from model name
    # return (
    #     f'<a target="_blank" style="text-decoration: underline" href="{link}">{model_name.split("/")[-1]}</a>'
    # )
    return f'<a target="_blank" style="text-decoration: underline" href="{link}">{model_name}</a>'


def add_rank(df):
    cols_to_rank = [
        col
        for col in df.columns
        if col
        not in [
            "Model",
            "Model Size (Million Parameters)",
            "Memory Usage (GB, fp32)",
            "Embedding Dimensions",
            "Max Tokens",
        ]
    ]
    if len(cols_to_rank) == 1:
        df.sort_values(cols_to_rank[0], ascending=False, inplace=True)
    else:
        df.insert(len(df.columns) - len(cols_to_rank), "Average", df[cols_to_rank].mean(axis=1, skipna=False))
        df.sort_values("Average", ascending=False, inplace=True)
    df.insert(0, "Rank", list(range(1, len(df) + 1)))
    df = df.round(2)
    # Fill NaN after averaging
    df.fillna("", inplace=True)
    return df


def get_vidore_data():
    api = HfApi()

    # local cache path
    model_infos_path = "model_infos.json"
    MODEL_INFOS = {}
    if os.path.exists(model_infos_path):
        with open(model_infos_path) as f:
            MODEL_INFOS = json.load(f)

    models = api.list_models(filter="vidore")

    for model in models:
        if model.modelId not in MODEL_INFOS:
            readme_path = hf_hub_download(model.modelId, filename="README.md")
            meta = metadata_load(readme_path)
            try:
                result_path = hf_hub_download(model.modelId, filename="results.json")

                with open(result_path) as f:
                    results = json.load(f)
                # keep only ndcg_at_5
                for dataset in results:
                    results[dataset] = {key: value for key, value in results[dataset].items() if "ndcg_at_5" in key}

                MODEL_INFOS[model.modelId] = {"metadata": meta, "results": results}
            except:
                continue

    model_res = {}
    df = None
    if len(MODEL_INFOS) > 0:
        for model in MODEL_INFOS.keys():
            res = MODEL_INFOS[model]["results"]
            dataset_res = {}
            for dataset in res.keys():
                if "validation_set" == dataset:
                    continue
                dataset_res[dataset] = res[dataset]["ndcg_at_5"]
            model_res[model] = dataset_res

        df = pd.DataFrame(model_res).T

        # add average
        # df["average"] = df.mean(axis=1)
        # df = df.sort_values(by="average", ascending=False)
        # # round to 2 decimals
        # df = df.round(2)
    return df


def add_rank_and_format(df):
    df = df.reset_index()
    df = df.rename(columns={"index": "Model"})
    df = add_rank(df)
    df["Model"] = df["Model"].apply(make_clickable_model)
    return df


# 1. Force headers to wrap
# 2. Force model column (maximum) width
# 3. Prevent model column from overflowing, scroll instead
# 4. Prevent checkbox groups from taking up too much space

css = """
table > thead {
    white-space: normal
}

table {
    --cell-width-1: 250px
}

table > tbody > tr > td:nth-child(2) > div {
    overflow-x: auto
}

.filter-checkbox-group {
    max-width: max-content;
}

"""


def get_refresh_function():
    def _refresh():
        data_task_category = get_vidore_data()
        return add_rank_and_format(data_task_category)

    return _refresh


def get_refresh_overall_function():
    return lambda: get_refresh_function()


data = get_vidore_data()
data = add_rank_and_format(data)

NUM_DATASETS = len(data.columns) - 3
NUM_SCORES = len(data) * NUM_DATASETS
NUM_MODELS = len(data)

with gr.Blocks(css=css) as block:
    gr.Markdown("# ViDoRe: The Visual Document Retrieval Benchmark 📚🔍")
    gr.Markdown("## From the paper - ColPali: Efficient Document Retrieval with Vision Language Models 👀")

    gr.Markdown(
        f"""
    Visual Document Retrieval Benchmark leaderboard. To submit, refer to the <a href="https://github.com/tonywu71/vidore-benchmark/" target="_blank" style="text-decoration: underline">ViDoRe GitHub repository</a>.  Refer to the [ColPali paper](https://arxiv.org/abs/XXXX.XXXXX) for details on metrics, tasks and models.
    """
    )

    with gr.Row():
        datatype = ["number", "markdown"] + ["number"] * (NUM_DATASETS + 1)
        dataframe = gr.Dataframe(data, datatype=datatype, type="pandas", height=500)

    with gr.Row():
        refresh_button = gr.Button("Refresh")
        refresh_button.click(get_refresh_function(), inputs=None, outputs=dataframe, concurrency_limit=20)

    gr.Markdown(
        f"""
    - **Total Datasets**: {NUM_DATASETS}
    - **Total Scores**: {NUM_SCORES}
    - **Total Models**: {NUM_MODELS}
    """
        + r"""
    Please consider citing:

    ```bibtex
    @article{}
    ```
    """
    )


if __name__ == "__main__":
    block.queue(max_size=10).launch(debug=True)
