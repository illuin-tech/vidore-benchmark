from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel


class MetricsModel(BaseModel):
    """
    Metrics model for the ViDoRe benchmark results. Contains the default MTER metrics.
    """

    # NDCG metrics
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


class Metadata(BaseModel):
    """
    Metadata for the ViDoRe benchmark results.
    """

    timestamp: datetime
    vidore_benchmark_hash: str

    class Config:
        extra = "allow"


class ViDoReBenchmarkResults(BaseModel):
    """
    A Pydantic model for the ViDoRe benchmark results.

    Example usage:

    ```python
    >>> root = RootModel(
            metadata=Metadata(
                timestamp=datetime.now(),
                vidore_benchmark_hash="1234567890abcdef",
            ),
            metrics={
                "vidore/syntheticDocQA_dummy": MetricsModel(
                    ndcg_at_1=1.0,
                    ndcg_at_3=1.0,
                )
            }
        )
    >>> json_data = root.model_dump_json()
    ```
    """

    metadata: Metadata
    metrics: Dict[str, MetricsModel]
