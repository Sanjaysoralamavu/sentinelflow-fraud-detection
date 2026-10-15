from __future__ import annotations

from pathlib import Path
from typing import Self

import joblib
import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

FeatureMatrix = np.ndarray | sparse.spmatrix


def positive_class_weights(labels: np.ndarray, weight_cap: float) -> np.ndarray:
    """Create sample weights that increase the influence of fraud examples."""

    labels = np.asarray(labels, dtype=np.int8).reshape(-1)

    if weight_cap < 1:
        raise ValueError("positive_weight_cap must be at least 1.")

    fraud_count = int(np.count_nonzero(labels == 1))
    legitimate_count = int(np.count_nonzero(labels == 0))

    if fraud_count == 0 or legitimate_count == 0:
        raise ValueError("Training labels must contain both fraud and legitimate rows.")

    fraud_weight = min(max(legitimate_count / fraud_count, 1.0), weight_cap)

    return np.where(labels == 1, fraud_weight, 1.0).astype(np.float64)


class StaticModelSuite:
    """Frozen Logistic Regression and Random Forest baselines."""

    def __init__(
        self,
        *,
        seed: int,
        positive_weight_cap: float,
        rf_enabled: bool,
        rf_training_rows: int,
        rf_n_estimators: int,
        rf_max_depth: int | None,
        rf_min_samples_leaf: int,
    ) -> None:
        if positive_weight_cap < 1:
            raise ValueError("positive_weight_cap must be at least 1.")
        if rf_training_rows < 2:
            raise ValueError("rf_training_rows must be at least 2.")
        if rf_n_estimators < 1:
            raise ValueError("rf_n_estimators must be positive.")
        if rf_max_depth is not None and rf_max_depth < 1:
            raise ValueError("rf_max_depth must be positive or None.")
        if rf_min_samples_leaf < 1:
            raise ValueError("rf_min_samples_leaf must be positive.")

        self.seed = int(seed)
        self.positive_weight_cap = float(positive_weight_cap)
        self.rf_training_rows = int(rf_training_rows)
        self._is_fitted = False

        # This is logistic regression trained efficiently for a large,
        # potentially sparse warm-up dataset.
        self.logistic = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.0001,
            max_iter=1000,
            tol=0.001,
            average=True,
            random_state=self.seed,
        )

        self.random_forest = (
            RandomForestClassifier(
                n_estimators=int(rf_n_estimators),
                max_depth=rf_max_depth,
                min_samples_leaf=int(rf_min_samples_leaf),
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=self.seed,
            )
            if rf_enabled
            else None
        )

    @property
    def names(self) -> list[str]:
        """Return the names used in result tables and visualizations."""

        names = ["static_logistic"]

        if self.random_forest is not None:
            names.append("static_random_forest")

        return names

    @property
    def is_fitted(self) -> bool:
        """Indicate whether the warm-up training has been completed."""

        return self._is_fitted

    def fit(self, features: FeatureMatrix, labels: np.ndarray) -> Self:
        """Train both baselines once on the warm-up window."""

        if self._is_fitted:
            raise RuntimeError(
                "Static models are frozen after warm-up training and cannot be refitted."
            )

        labels = np.asarray(labels, dtype=np.int8).reshape(-1)

        if features.ndim != 2:
            raise ValueError("features must be a two-dimensional matrix.")
        if features.shape[0] != labels.shape[0]:
            raise ValueError("features and labels must contain the same number of rows.")
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("Training labels must contain binary classes 0 and 1.")

        weights = positive_class_weights(labels, self.positive_weight_cap)
        self.logistic.fit(features, labels, sample_weight=weights)

        if self.random_forest is not None:
            indices = self._random_forest_indices(labels)
            self.random_forest.fit(features[indices], labels[indices])

        self._is_fitted = True
        return self

    def predict_probabilities(
        self,
        features: FeatureMatrix,
    ) -> dict[str, np.ndarray]:
        """Return one fraud probability per row for every enabled baseline."""

        self._require_fitted()

        probabilities = {
            "static_logistic": self.logistic.predict_proba(features)[:, 1].astype(np.float32)
        }

        if self.random_forest is not None:
            probabilities["static_random_forest"] = self.random_forest.predict_proba(features)[
                :, 1
            ].astype(np.float32)

        return probabilities

    def save(self, directory: str | Path) -> None:
        """Save the fitted static models for reproducibility."""

        self._require_fitted()

        output_directory = Path(directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            self.logistic,
            output_directory / "static_logistic.joblib",
        )

        if self.random_forest is not None:
            joblib.dump(
                self.random_forest,
                output_directory / "static_random_forest.joblib",
            )

    def _random_forest_indices(self, labels: np.ndarray) -> np.ndarray:
        """Select a deterministic warm-up subset while retaining both classes."""

        row_count = len(labels)
        selected_count = min(self.rf_training_rows, row_count)

        if selected_count == row_count:
            return np.arange(row_count)

        generator = np.random.default_rng(self.seed)
        indices = generator.choice(
            row_count,
            size=selected_count,
            replace=False,
        )

        selected_labels = labels[indices]

        if len(np.unique(selected_labels)) == 1:
            missing_class = 1 - int(selected_labels[0])
            missing_indices = np.flatnonzero(labels == missing_class)
            indices[-1] = missing_indices[0]

        return indices

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Static models must be fitted before prediction or saving.")
