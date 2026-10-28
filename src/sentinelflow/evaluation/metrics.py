"""Evaluation metrics for fraud-detection model predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PRECISION_AT_K = 100


@dataclass(frozen=True)
class MetricResult:
    """Metrics calculated for one model over one set of predictions."""

    tp: int
    tn: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    fpr: float
    roc_auc: float
    average_precision: float
    precision_at_100: float


def precision_at_k(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    k: int = PRECISION_AT_K,
) -> float:
    """Return fraud precision among the k highest-scored transactions."""

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    if y_true.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("y_true and probabilities must be one-dimensional.")

    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same length.")

    if len(y_true) == 0:
        raise ValueError("Cannot calculate precision@k for an empty batch.")

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    effective_k = min(
        k,
        len(y_true),
    )

    # Stable descending sort so tied scores are handled reproducibly.
    ranked_indices = np.argsort(
        -probabilities,
        kind="stable",
    )

    top_indices = ranked_indices[:effective_k]
    top_labels = y_true[top_indices]

    return float(np.sum(top_labels == 1) / effective_k)


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> MetricResult:
    """Calculate binary fraud-detection metrics."""

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    if y_true.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("y_true and probabilities must be one-dimensional.")

    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same length.")

    if len(y_true) == 0:
        raise ValueError("Cannot calculate metrics for an empty batch.")

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between zero and one.")

    predicted = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predicted,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_true,
        predicted,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predicted,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predicted,
        zero_division=0,
    )

    negative_count = tn + fp

    fpr = fp / negative_count if negative_count else 0.0

    # AUC is undefined when the data contains only one class.
    if len(np.unique(y_true)) < 2:
        roc_auc = float("nan")
        average_precision = float("nan")
    else:
        roc_auc = roc_auc_score(
            y_true,
            probabilities,
        )

        average_precision = average_precision_score(
            y_true,
            probabilities,
        )

    p_at_100 = precision_at_k(
        y_true,
        probabilities,
        k=PRECISION_AT_K,
    )

    return MetricResult(
        tp=int(tp),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        fpr=float(fpr),
        roc_auc=float(roc_auc),
        average_precision=float(average_precision),
        precision_at_100=float(p_at_100),
    )


class MetricAccumulator:
    """Accumulate labels and probabilities for a combined metric summary."""

    def __init__(
        self,
        threshold: float = 0.5,
    ) -> None:
        self.threshold = threshold
        self._labels: list[np.ndarray] = []
        self._probabilities: list[np.ndarray] = []

    def update(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
    ) -> None:
        """Add one batch of labels and prediction probabilities."""

        y_true = np.asarray(y_true)
        probabilities = np.asarray(probabilities)

        if y_true.ndim != 1 or probabilities.ndim != 1:
            raise ValueError("y_true and probabilities must be one-dimensional.")

        if len(y_true) != len(probabilities):
            raise ValueError("y_true and probabilities must have the same length.")

        if len(y_true) == 0:
            raise ValueError("Cannot accumulate an empty batch.")

        self._labels.append(y_true.copy())

        self._probabilities.append(probabilities.copy())

    def calculate(self) -> MetricResult:
        """Calculate metrics across all accumulated rows."""

        if not self._labels:
            raise ValueError("No evaluation rows have been accumulated.")

        y_true = np.concatenate(self._labels)

        probabilities = np.concatenate(self._probabilities)

        return calculate_metrics(
            y_true,
            probabilities,
            threshold=self.threshold,
        )

    @property
    def row_count(self) -> int:
        """Return the number of accumulated evaluation rows."""

        return sum(len(labels) for labels in self._labels)
