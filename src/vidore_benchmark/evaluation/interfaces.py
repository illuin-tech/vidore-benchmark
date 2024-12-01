from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel


class MetricsModel(BaseModel):
    """
    Metrics model for the ViDoRe benchmark results.
    Contains the default MTEB metrics: nDCG, MAP, Recall, Precision, MRR, and NAUCS.

    Example usage:

    ```python
    >>> metrics = MetricsModel(
            ndcg_at_1=0.5,
            ndcg_at_3=0.7,
            map_at_1=0.4,
        )
    >>> # Serialize the model to JSON
    >>> json_data = metrics.model_dump_json()
    """

    # nDCG metrics
    ndcg_at_1: Optional[float] = None
    ndcg_at_3: Optional[float] = None
    ndcg_at_5: Optional[float] = None
    ndcg_at_10: Optional[float] = None
    ndcg_at_20: Optional[float] = None
    ndcg_at_50: Optional[float] = None
    ndcg_at_100: Optional[float] = None

    # MAP metrics
    map_at_1: Optional[float] = None
    map_at_3: Optional[float] = None
    map_at_5: Optional[float] = None
    map_at_10: Optional[float] = None
    map_at_20: Optional[float] = None
    map_at_50: Optional[float] = None
    map_at_100: Optional[float] = None

    # Recall metrics
    recall_at_1: Optional[float] = None
    recall_at_3: Optional[float] = None
    recall_at_5: Optional[float] = None
    recall_at_10: Optional[float] = None
    recall_at_20: Optional[float] = None
    recall_at_50: Optional[float] = None
    recall_at_100: Optional[float] = None

    # Precision metrics
    precision_at_1: Optional[float] = None
    precision_at_3: Optional[float] = None
    precision_at_5: Optional[float] = None
    precision_at_10: Optional[float] = None
    precision_at_20: Optional[float] = None
    precision_at_50: Optional[float] = None
    precision_at_100: Optional[float] = None

    # MRR metrics
    mrr_at_1: Optional[float] = None
    mrr_at_3: Optional[float] = None
    mrr_at_5: Optional[float] = None
    mrr_at_10: Optional[float] = None
    mrr_at_20: Optional[float] = None
    mrr_at_50: Optional[float] = None
    mrr_at_100: Optional[float] = None

    # NAUCS metrics
    naucs_at_1_max: Optional[float] = None
    naucs_at_1_std: Optional[float] = None
    naucs_at_1_diff1: Optional[float] = None
    naucs_at_3_max: Optional[float] = None
    naucs_at_3_std: Optional[float] = None
    naucs_at_3_diff1: Optional[float] = None
    naucs_at_5_max: Optional[float] = None
    naucs_at_5_std: Optional[float] = None
    naucs_at_5_diff1: Optional[float] = None
    naucs_at_10_max: Optional[float] = None
    naucs_at_10_std: Optional[float] = None
    naucs_at_10_diff1: Optional[float] = None
    naucs_at_20_max: Optional[float] = None
    naucs_at_20_std: Optional[float] = None
    naucs_at_20_diff1: Optional[float] = None
    naucs_at_50_max: Optional[float] = None
    naucs_at_50_std: Optional[float] = None
    naucs_at_50_diff1: Optional[float] = None
    naucs_at_100_max: Optional[float] = None
    naucs_at_100_std: Optional[float] = None
    naucs_at_100_diff1: Optional[float] = None


class MetadataModel(BaseModel):
    """
    Metadata data for the ViDoRe benchmark results.

    Example usage:

    ```python
    >>> from datetime import datetime
    >>> metadata = MetadataModel(
            timestamp=datetime.now(),
            vidore_benchmark_version="0.0.1.dev7+g462dc4f.d20241102",
        )
    >>> # Serialize the model to JSON
    >>> json_data = metadata.model_dump_json()
    ```
    """

    timestamp: datetime
    vidore_benchmark_version: str

    # NOTE: `Config` is used to allow extra fields in the model
    class Config:
        extra = "allow"


class ViDoReBenchmarkResults(BaseModel):
    """
    ViDoRe benchmark results.

    This Pydantic model contains:
    - Metadata: Information about the benchmark run, including the timestamp and the `vidore-benchmark` package version.
    - Metrics: Dictionary of metrics: the keys are the dataset names and the values are the metrics for that dataset.

    Example usage:

    ```python
    >>> root = RootModel(
            metadata=Metadata(
                timestamp=datetime.now(),
                vidore_benchmark_version="0.0.1.dev7+g462dc4f.d20241102",
            ),
            metrics={
                "vidore/syntheticDocQA_dummy": MetricsModel(
                    ndcg_at_1=1.0,
                    ndcg_at_3=1.0,
                )
            }
        )
    >>> # Serialize the model to JSON
    >>> json_data = root.model_dump_json()
    ```
    """

    metadata: MetadataModel
    metrics: Dict[str, MetricsModel]

    @classmethod
    def merge(cls, results: List["ViDoReBenchmarkResults"]) -> "ViDoReBenchmarkResults":
        """
        Merge multiple `ViDoReBenchmarkResults` instances into a single one.

        Uses the latest timestamp from the input results and combines all metrics.
        If there are conflicting metrics for the same benchmark, the last one in the list takes precedence.

        Args:
            results: List of ViDoReBenchmarkResults to merge

        Returns:
            A new `ViDoReBenchmarkResults` instance containing the merged data

        Raises:
            ValueError: If the input list is empty
        """
        if not results:
            raise ValueError("Cannot merge an empty list of results")

        # Use the metadata from the result with the latest timestamp
        latest_result = max(results, key=lambda x: x.metadata.timestamp)
        merged_metadata = latest_result.metadata

        # Merge all metrics, later results override earlier ones for the same benchmark
        merged_metrics: Dict[str, MetricsModel] = {}
        for result in results:
            if set(merged_metrics.keys()).intersection(set(result.metrics.keys())):
                raise ValueError("Duplicate dataset keys found in the input results")
            merged_metrics.update(result.metrics)

        return cls(metadata=merged_metadata, metrics=merged_metrics)
