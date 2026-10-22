"""Tests for fraud-detection evaluation metrics."""

import numpy as np
import pytest

from sentinelflow.evaluation.metrics import (
    MetricAccumulator,
    calculate_metrics,
)


def test_calculate_metrics_matches_hand_computed_example() -> None:
    """Metric calculations should match a small hand-computed example."""

    y_true = np.array([1, 1, 0, 0])
    probabilities = np.array([0.9, 0.4, 0.8, 0.1])

    result = calculate_metrics(
        y_true,
        probabilities,
        threshold=0.5,
    )

    assert result.tp == 1
    assert result.tn == 1
    assert result.fp == 1
    assert result.fn == 1

    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
    assert result.f1 == pytest.approx(0.5)
    assert result.fpr == pytest.approx(0.5)

    assert result.roc_auc == pytest.approx(0.75)
    assert result.pr_auc == pytest.approx(0.833333333)

def test_calculate_metrics_handles_single_class_batch() -> None:
    """AUC metrics should be NaN when only one class is present."""

    y_true = np.array([0, 0, 0, 0])
    probabilities = np.array([0.1, 0.2, 0.3, 0.4])

    result = calculate_metrics(y_true, probabilities)

    assert result.tp == 0
    assert result.tn == 4
    assert result.fp == 0
    assert result.fn == 0

    assert result.precision == pytest.approx(0.0)
    assert result.recall == pytest.approx(0.0)
    assert result.f1 == pytest.approx(0.0)
    assert result.fpr == pytest.approx(0.0)

    assert np.isnan(result.roc_auc)
    assert np.isnan(result.pr_auc)


def test_calculate_metrics_rejects_different_lengths() -> None:
    """Labels and probabilities must contain the same number of rows."""

    y_true = np.array([0, 1, 0])
    probabilities = np.array([0.2, 0.8])

    with pytest.raises(
        ValueError,
        match="same length",
    ):
        calculate_metrics(y_true, probabilities)


def test_calculate_metrics_rejects_empty_input() -> None:
    """Metrics cannot be calculated without evaluation rows."""

    with pytest.raises(
        ValueError,
        match="empty batch",
    ):
        calculate_metrics(
            np.array([]),
            np.array([]),
        )

def test_metric_accumulator_combines_batches_before_calculating() -> None:
    """Combined metrics should be calculated over all rows, not batch averages."""

    accumulator = MetricAccumulator(threshold=0.5)

    accumulator.update(
        np.array([1, 0]),
        np.array([0.9, 0.8]),
    )

    accumulator.update(
        np.array([1, 0, 0]),
        np.array([0.7, 0.2, 0.1]),
    )

    result = accumulator.calculate()

    assert accumulator.row_count == 5

    assert result.tp == 2
    assert result.tn == 2
    assert result.fp == 1
    assert result.fn == 0

    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(1.0)
    assert result.f1 == pytest.approx(0.8)
    assert result.fpr == pytest.approx(1 / 3)