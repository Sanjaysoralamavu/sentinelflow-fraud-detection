"""Prediction recorder for batch-level model evaluation."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
from sentinelflow.config import ProjectConfig
from sentinelflow.streaming.contracts import Batch
from sentinelflow.evaluation.metrics import (
    MetricAccumulator,
    MetricResult,
    calculate_metrics,
)


class EvaluationRecorder:
    """Collect per-model metrics for every scored evaluation batch."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.records: list[dict[str, object]] = []

        self._overall: dict[str, MetricAccumulator] = {}
        self._phases: dict[
            tuple[str, str],
            MetricAccumulator,
        ] = {}
    
    @classmethod
    
    def from_config(cls, config: ProjectConfig) -> EvaluationRecorder:
        """Create an evaluation recorder from project configuration."""
            
        model_settings = config.section("models")
            
        return cls(
            threshold=float(model_settings["threshold"]),
        )

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

            overall = self._overall.setdefault(
                model_name,
                MetricAccumulator(threshold=self.threshold),
            )
            overall.update(
                batch.labels,
                probabilities,
            )

            phase_key = (model_name, batch.phase)

            phase = self._phases.setdefault(
                phase_key,
                MetricAccumulator(threshold=self.threshold),
            )
            phase.update(
                batch.labels,
                probabilities,
            )

    def overall_metrics(self) -> dict[str, MetricResult]:
        """Return metrics over all recorded rows for each model."""

        return {
            model_name: accumulator.calculate()
            for model_name, accumulator in self._overall.items()
        }

    def phase_metrics(self) -> dict[tuple[str, str], MetricResult]:
        """Return metrics over all rows within each model and phase."""

        return {
            key: accumulator.calculate()
            for key, accumulator in self._phases.items()
        }