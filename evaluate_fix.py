import argparse
import json
import os
from typing import List

import torch

torch.manual_seed(42)

from typing import cast

import pytrec_eval
from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2, ColQwen2Processor
from datasets import load_dataset
from tqdm import tqdm

ALL_DATASET = [
    "mixedbread-ai/vidore-arxivqa_test_subsampled",
    "mixedbread-ai/vidore-docvqa_test_subsampled",
    "mixedbread-ai/vidore-infovqa_test_subsampled",
    "mixedbread-ai/vidore-tabfquad_test_subsampled",
    "mixedbread-ai/vidore-tatdqa_test",
    "mixedbread-ai/vidore-shiftproject_test",
    "mixedbread-ai/vidore-syntheticDocQA_artificial_intelligence_test",
    "mixedbread-ai/vidore-syntheticDocQA_energy_test",
    "mixedbread-ai/vidore-syntheticDocQA_government_reports_test",
    "mixedbread-ai/vidore-syntheticDocQA_healthcare_industry_test",
]


def get_torch_device(device: str = "auto") -> str:
    """
    Returns the device (string) to be used by PyTorch.

    Defaults to "auto" which will use:
    - "cuda:0" if available
    - else "mps" if available
    - "cpu" otherwise.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        elif torch.backends.mps.is_available():  # for Apple Silicon
            return "mps"
        else:
            return "cpu"
    else:
        return device


def score_single_vector(qs: List[torch.Tensor], ps: List[torch.Tensor]) -> torch.Tensor:
    """
    Evaluate the similarity scores using a simple dot product between the query and
    the passage embeddings.
    """
    qs_stacked = torch.stack(qs)
    ps_stacked = torch.stack(ps)
    scores = torch.einsum("bd,cd->bc", qs_stacked, ps_stacked)
    return scores


def score_multi_vector(
    qs: List[torch.Tensor],
    ps: List[torch.Tensor],
    batch_size: int = 128,
    device: str = "auto",
) -> torch.Tensor:
    """
    Evaluate the similarity scores using the ColBERT scoring function.
    """
    device = get_torch_device(device)
    scores: List[torch.Tensor] = []
    if not qs:
        raise ValueError("No queries provided.")
    if not ps:
        raise ValueError("No passages provided.")

    if qs[0].dtype != ps[0].dtype:
        raise ValueError("Queries and passages must have the same dtype.")

    is_dtype_int8 = ps[0].dtype == torch.int8 and qs[0].dtype == torch.int8

    for i in range(0, len(qs), batch_size):
        scores_batch: List[torch.Tensor] = []

        qs_batch = torch.nn.utils.rnn.pad_sequence(
            qs[i : i + batch_size],
            batch_first=True,
            padding_value=0,
        )

        for j in range(0, len(ps), batch_size):
            ps_batch = torch.nn.utils.rnn.pad_sequence(
                ps[j : j + batch_size],
                batch_first=True,
                padding_value=0,
            )

            if is_dtype_int8:
                # NOTE: CUDA does not support int8 operations, so we need to move the
                # tensors to the CPU for the MaxSim operation.
                scores_batch.append(
                    torch.einsum(
                        "bnd,csd->bcns",
                        qs_batch.to("cpu"),
                        ps_batch.to("cpu"),
                    )
                    .max(dim=3)[0]
                    .sum(dim=2, dtype=torch.int16)  # prevent int8 overflow
                    .to(device)
                )
            else:
                scores_batch.append(
                    torch.einsum(
                        "bnd,csd->bcns",
                        qs_batch.to(device),
                        ps_batch.to(device),
                    )
                    .max(dim=3)[0]
                    .sum(dim=2)
                )

        scores.append(torch.cat(scores_batch, dim=1).cpu())

    return torch.cat(scores, dim=0)


def evaluate(dataset_name: str) -> float:
    print(f"Evaluate {dataset_name}...")
    ds_corpus = load_dataset(dataset_name, "corpus", split="train")
    image_ids = list(ds_corpus["corpus-id"])
    images = list(ds_corpus["image"])

    queries = load_dataset(dataset_name, "queries")["train"]
    qids, query = queries["query-id"], queries["query"]
    qrels_data = load_dataset(dataset_name, "default")["train"]

    qrels = {}
    for qrel in qrels_data:
        qid = qrel["query-id"]
        cid = qrel["corpus-id"]
        if f"{qid}" not in qrels:
            qrels[f"{qid}"] = {}
        qrels[f"{qid}"][f"{cid}"] = int(qrel["score"])

    all_query_embeddings = []
    for i in tqdm(range(0, len(query), args.batch_size)):
        batch_queries = processor.process_queries(query[i : i + args.batch_size]).to(model.device)

        with torch.no_grad():
            query_embeddings = model(**batch_queries)

        all_query_embeddings.extend(list(torch.unbind(query_embeddings)))

    all_image_embeddings = []
    for i in tqdm(range(0, len(images), args.batch_size)):
        batch_images = images[i : i + args.batch_size]
        batch_images = processor.process_images(batch_images).to(model.device)
        with torch.no_grad():
            image_embeddings = model(**batch_images)
        all_image_embeddings.extend(list(torch.unbind(image_embeddings)))

    scores = score_multi_vector(all_query_embeddings, all_image_embeddings)

    results = {}
    for i, qid in enumerate(qids):
        qid = str(qid)
        results[qid] = {}
        _, indices = torch.sort(scores[i], descending=True)
        for idx in indices:
            # cid = doc2id[idx.item()]
            cid = str(image_ids[idx])
            results[qid][cid] = scores[i][idx].item()

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, pytrec_eval.supported_measures)
    k_values = [1, 3, 5, 10]

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})

    scores_eval = evaluator.evaluate(results)

    all_ndcgs, all_aps, all_recalls, all_precisions = {}, {}, {}, {}

    for k in k_values:
        all_ndcgs[f"NDCG@{k}"] = []
        all_aps[f"MAP@{k}"] = []
        all_recalls[f"Recall@{k}"] = []
        all_precisions[f"P@{k}"] = []

    for query_id in scores_eval.keys():
        for k in k_values:
            all_ndcgs[f"NDCG@{k}"].append(scores_eval[query_id]["ndcg_cut_" + str(k)])
            all_aps[f"MAP@{k}"].append(scores_eval[query_id]["map_cut_" + str(k)])
            all_recalls[f"Recall@{k}"].append(scores_eval[query_id]["recall_" + str(k)])
            all_precisions[f"P@{k}"].append(scores_eval[query_id]["P_" + str(k)])

    ndcg, _map, recall, precision = (
        all_ndcgs.copy(),
        all_aps.copy(),
        all_recalls.copy(),
        all_precisions.copy(),
    )

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(sum(ndcg[f"NDCG@{k}"]) / len(scores_eval), 5)
        _map[f"MAP@{k}"] = round(sum(_map[f"MAP@{k}"]) / len(scores_eval), 5)
        recall[f"Recall@{k}"] = round(sum(recall[f"Recall@{k}"]) / len(scores_eval), 5)
        precision[f"P@{k}"] = round(sum(precision[f"P@{k}"]) / len(scores_eval), 5)

    print(f"Results of {dataset_name}:")
    print(ndcg)
    print(_map)
    print(recall)
    print(precision)
    os.makedirs("results", exist_ok=True)
    with open(
        f'results/result-{args.adapter_name.replace("/", "--")}-{dataset_name.split("/")[-1]}.jsonl', "w"
    ) as writer:
        json.dump({"NDCG": ndcg, "MAP": _map, "Recall": recall, "Precision": precision}, writer, indent=4)
    return ndcg["NDCG@5"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=True, help="Specify model name or path to set transformer backbone, required"
    )
    parser.add_argument(
        "--adapter_name",
        type=str,
        required=True,
        help="Specify adapter name or path to set transformer backbone, required",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["ColPali", "DSE", "ColQwen2"],
        help="Specify model_type, choices ['Colpali', 'ColQwen2']",
    )
    parser.add_argument("--dataset_name", type=str, default="all", help="Specify dataset name to evaluate. Default all")
    parser.add_argument("--batch_size", type=int, default=4, help="Specify batch size. Default 4")
    args = parser.parse_args()
    print(f"Args: {args}")

    if args.model_type == "ColPali":
        MODEL_CLS = ColPali
        PROCESSOR_CLS = ColPaliProcessor
    elif args.model_type == "ColQwen2":
        MODEL_CLS = ColQwen2
        PROCESSOR_CLS = ColQwen2Processor
    else:
        raise NotImplementedError

    model = cast(
        MODEL_CLS,
        MODEL_CLS.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).cuda(),
    ).eval()
    print(f"Loaded {args.model_type} model (non-trained weights) from `{args.model_name}`")

    if args.adapter_name is not None and args.adapter_name != "None":
        model.load_adapter(args.adapter_name)
        print(f"Loaded {args.model_type} adapter from `{args.adapter_name}`")

    model = model.eval()
    processor = PROCESSOR_CLS.from_pretrained(args.model_name or args.adapter_name)

    if args.dataset_name == "all":
        main_scores = []
        for dataset_name in ALL_DATASET:
            main_scores.append(evaluate(dataset_name) * 100)
        print("\t".join(ALL_DATASET))
        print("\t".join([str(round(v, 2)) for v in main_scores]))
    else:
        evaluate(args.dataset_name)
