"""Unified coordinator for SentinelFlow fraud-detection models."""

from __future__ import annotations

from typing import Self

import numpy as np

from sentinelflow.config import ProjectConfig
from sentinelflow.models.adaptive import build_adaptive_models
from sentinelflow.models.common import DelayedBatch, DriftEvent, StreamModel
from sentinelflow.models.static import FeatureMatrix, StaticModelSuite


class ModelSuite:
    """Coordinate frozen static baselines and adaptive streaming models."""

    def __init__(
        self,
        static_models: StaticModelSuite,
        adaptive_models: dict[str, StreamModel],
    ) -> None:
        overlapping_names = set(static_models.names).intersection(adaptive_models)

        if overlapping_names:
            names = ", ".join(sorted(overlapping_names))
            raise ValueError(f"Duplicate model names: {names}")

        self.static_models = static_models
        self.adaptive_models = adaptive_models
        self._is_warmed_up = False

    @classmethod
    def from_config(cls, config: ProjectConfig) -> Self:
        """Build all enabled models from the shared project configuration."""

        models = config.section("models")
        random_forest = models["random_forest"]
        max_depth = random_forest.get("max_depth")

        static_models = StaticModelSuite(
            seed=config.seed,
            positive_weight_cap=float(models["positive_weight_cap"]),
            rf_enabled=bool(random_forest["enabled"]),
            rf_training_rows=int(random_forest["training_rows"]),
            rf_n_estimators=int(random_forest["n_estimators"]),
            rf_max_depth=None if max_depth is None else int(max_depth),
            rf_min_samples_leaf=int(random_forest["min_samples_leaf"]),
        )

        adaptive_models = build_adaptive_models(
            models=models,
            seed=config.seed,
        )

        return cls(
            static_models=static_models,
            adaptive_models=adaptive_models,
        )

    @property
    def names(self) -> list[str]:
        """Return every model name in deterministic reporting order."""

        return [
            *self.static_models.names,
            *self.adaptive_models.keys(),
        ]

    @property
    def is_warmed_up(self) -> bool:
        """Indicate whether all models completed warm-up training."""

        return self._is_warmed_up

    @property
    def drift_events(self) -> list[DriftEvent]:
        """Return detector events from all adaptive models in stream order."""

        events: list[DriftEvent] = []

        for model in self.adaptive_models.values():
            events.extend(getattr(model, "events", ()))

        return sorted(
            events,
            key=lambda event: (
                event.observed_batch,
                event.source_batch,
                event.model,
            ),
        )

    def warm_up(
        self,
        features: FeatureMatrix,
        labels: np.ndarray,
    ) -> Self:
        """Train every model on the same preprocessed warm-up window."""

        if self._is_warmed_up:
            raise RuntimeError("ModelSuite has already completed warm-up training.")

        labels = np.asarray(labels, dtype=np.int8).reshape(-1)

        if features.ndim != 2:
            raise ValueError("features must be a two-dimensional matrix.")

        if features.shape[0] != labels.shape[0]:
            raise ValueError("features and labels must contain the same number of rows.")

        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("Warm-up labels must contain both legitimate and fraud classes.")

        self.static_models.fit(features, labels)

        for model in self.adaptive_models.values():
            model.warm_up(features, labels)

        self._is_warmed_up = True
        return self

    def predict_probabilities(
        self,
        features: FeatureMatrix,
    ) -> dict[str, np.ndarray]:
        """Return one fraud probability per row from every model."""

        self._require_warmed_up()

        probabilities = self.static_models.predict_probabilities(features)

        for name, model in self.adaptive_models.items():
            probabilities[name] = model.predict_proba(features)

        row_count = int(features.shape[0])

        for name, values in probabilities.items():
            scores = np.asarray(values, dtype=np.float32).reshape(-1)

            if scores.shape != (row_count,):
                raise ValueError(
                    f"Model '{name}' returned {scores.shape[0]} probabilities for {row_count} rows."
                )

            if not np.all(np.isfinite(scores)):
                raise ValueError(f"Model '{name}' returned non-finite probabilities.")

            if np.any((scores < 0.0) | (scores > 1.0)):
                raise ValueError(f"Model '{name}' returned probabilities outside [0, 1].")

            probabilities[name] = scores

        return probabilities

    def learn(self, batch: DelayedBatch) -> list[DriftEvent]:
        """Update adaptive models from one released delayed-label batch.

        Static models are intentionally excluded because they must remain
        frozen after warm-up.
        """

        self._require_warmed_up()

        events: list[DriftEvent] = []

        for model in self.adaptive_models.values():
            event = model.learn(batch)

            if event is not None:
                events.append(event)

        return events

    def _require_warmed_up(self) -> None:
        if not self._is_warmed_up:
            raise RuntimeError(
                "ModelSuite must complete warm-up training before prediction or delayed learning."
            )
