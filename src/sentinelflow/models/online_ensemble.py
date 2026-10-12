"""Online bagging ensemble of incremental fraud learners.

Each of the ``n_estimators`` learners (five by default) receives every
released row with its own independent Poisson(1) weight, following Oza and
Russell's online bagging. Each learner has its own seeded random generator,
so the Poisson weights are reproducible for a fixed seed. Poisson weights are
multiplied by the same capped fraud-class weight the adaptive models use. The
ensemble fraud probability is the mean of the learners' probabilities. The
ensemble has no drift detector and never resets.
"""

from __future__ import annotations

import numpy as np

from sentinelflow.models.common import (
    DelayedBatch,
    fit_in_chunks,
    fraud_probability,
    new_learner,
    positive_class_weight,
    sample_weights,
)


class OnlineBaggingEnsemble:
    """Poisson-weighted online bagging over incremental logistic regressions."""

    name = "online_ensemble"

    def __init__(
        self,
        n_estimators: int,
        alpha: float,
        positive_weight_cap: float,
        seed: int,
    ) -> None:
        if n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero.")
        if alpha <= 0:
            raise ValueError("alpha must be greater than zero.")
        if positive_weight_cap < 1.0:
            raise ValueError("positive_weight_cap must be at least one.")

        self.n_estimators = n_estimators
        self.positive_weight_cap = positive_weight_cap
        self.learners = [new_learner(alpha, seed + index) for index in range(n_estimators)]
        self._rngs = [np.random.default_rng(seed + 10_000 + index) for index in range(n_estimators)]
        self._fraud_weight = 1.0
        self._is_fitted = False

    @classmethod
    def from_config(cls, models: dict, seed: int) -> OnlineBaggingEnsemble:
        ensemble = models["online_ensemble"]
        return cls(
            n_estimators=int(ensemble["n_estimators"]),
            alpha=float(ensemble["alpha"]),
            positive_weight_cap=float(models["positive_weight_cap"]),
            seed=seed,
        )

    def warm_up(self, features: np.ndarray, labels: np.ndarray) -> None:
        labels = np.asarray(labels).astype(int)
        self._fraud_weight = positive_class_weight(labels, self.positive_weight_cap)
        self._update(features, labels)
        self._is_fitted = True

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("online_ensemble must be warmed up before scoring.")
        return np.mean([fraud_probability(learner, features) for learner in self.learners], axis=0)

    def learn(self, batch: DelayedBatch) -> None:
        """Update every learner with released labels; the ensemble never resets."""

        if not self._is_fitted:
            raise RuntimeError("online_ensemble must be warmed up before learning.")
        self._update(batch.features, np.asarray(batch.labels).astype(int))

    def _update(self, features: np.ndarray, labels: np.ndarray) -> None:
        class_weights = sample_weights(labels, self._fraud_weight)

        for learner, rng in zip(self.learners, self._rngs, strict=True):
            poisson = rng.poisson(1.0, size=len(labels))
            kept = poisson > 0
            if not kept.any():
                if hasattr(learner, "coef_"):
                    continue
                poisson = np.ones_like(poisson)
                kept = poisson > 0
            fit_in_chunks(
                learner,
                features[kept],
                labels[kept],
                (poisson * class_weights)[kept],
            )
