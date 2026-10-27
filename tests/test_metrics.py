"""Tests for fraud-detection evaluation metrics."""

import numpy as np
import pytest

from sentinelflow.evaluation.metrics import (
    MetricAccumulator,
    calculate_metrics,
    precision_at_k,
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

    # There are fewer than 100 rows, so all four rows are used.
    assert result.precision_at_100 == pytest.approx(0.5)


def test_precision_at_k_uses_highest_scored_transactions() -> None:
    """Precision@k should use only the k highest fraud scores."""

    y_true = np.array(
        [1, 0, 1, 0, 1, 0]
    )

    probabilities = np.array(
        [0.95, 0.90, 0.85, 0.80, 0.75, 0.10]
    )

    result = precision_at_k(
        y_true,
        probabilities,
        k=4,
    )

    # Top four labels are [1, 0, 1, 0].
    assert result == pytest.approx(0.5)


def test_precision_at_k_uses_all_rows_when_fewer_than_k() -> None:
    """Precision@k should use all rows when fewer than k are available."""

    y_true = np.array(
        [1, 0, 1, 1]
    )

    probabilities = np.array(
        [0.9, 0.8, 0.7, 0.6]
    )

    result = precision_at_k(
        y_true,
        probabilities,
        k=100,
    )

    assert result == pytest.approx(0.75)


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
    assert result.precision_at_100 == pytest.approx(0.0)


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

    # All five rows are used because there are fewer than 100.
    assert result.precision_at_100 == pytest.approx(2 / 5)


def test_metric_accumulator_ranks_across_batches() -> None:
    """Accumulated Precision@k should rank predictions across all batches."""

    accumulator = MetricAccumulator(threshold=0.5)

    accumulator.update(
        np.array([0, 1]),
        np.array([0.99, 0.80]),
    )

    accumulator.update(
        np.array([1, 0]),
        np.array([0.95, 0.10]),
    )

    result = accumulator.calculate()

    # With fewer than 100 rows, Precision@100 uses all four rows.
    assert result.precision_at_100 == pytest.approx(0.5)