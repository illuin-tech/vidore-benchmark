import json
from pathlib import Path


def log_metrics(metrics, dataset_name, log_file="metrics.json") -> None:
    if Path(log_file).exists():
        loaded_metrics = json.load(open(log_file, "r", encoding="utf-8"))
        loaded_metrics[dataset_name] = metrics
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(loaded_metrics, f)
    else:
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump({dataset_name: metrics}, f)
    return
