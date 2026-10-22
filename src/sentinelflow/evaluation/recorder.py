"""Prediction recorder for batch-level model evaluation."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from sentinelflow.evaluation.metrics import calculate_metrics
from sentinelflow.streaming.contracts import Batch


class EvaluationRecorder:
    """Collect per-model metrics for every scored evaluation batch."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.records: list[dict[str, object]] = []

    def record(
        self,
        batch: Batch,
        predictions: dict[str, np.ndarray],
    ) -> None:
        """Calculate and store metrics for each model in one batch."""

        for model_name, probabilities in predictions.items():
            metrics = calculate_metrics(
                batch.labels,
                probabilities,
                threshold=self.threshold,
            )

            self.records.append(
                {
                    "model": model_name,
                    "batch": batch.batch_id,
                    "progress": batch.progress,
                    "phase": batch.phase,
                    **asdict(metrics),
                }
            )