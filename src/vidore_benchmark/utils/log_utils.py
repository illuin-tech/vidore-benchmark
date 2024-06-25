from pathlib import Path
import json


def log_metrics(metrics, dataset_name, log_file="metrics.json"):
    if Path(log_file).exists():
        loaded_metrics = json.load(open(log_file))
        loaded_metrics[dataset_name] = metrics
        with open(log_file, "w") as f:
            json.dump(loaded_metrics, f)
    else:
        with open(log_file, "w") as f:
            json.dump({dataset_name: metrics}, f)
