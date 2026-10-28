"""Tests for adaptation evaluation across drift phases."""

import pytest

from sentinelflow.evaluation.adaptation import (
    calculate_relative_loss,
)


def test_relative_loss_when_performance_degrades() -> None:
    """Positive relative loss should indicate degradation."""

    result = calculate_relative_loss(
        pre_drift=0.90,
        drifted=0.81,
    )

    assert result == pytest.approx(0.10)


def test_relative_loss_when_performance_improves() -> None:
    """Negative relative loss should indicate improvement."""

    result = calculate_relative_loss(
        pre_drift=0.90,
        drifted=0.95,
    )

    assert result == pytest.approx((0.90 - 0.95) / 0.90)


def test_relative_loss_when_performance_is_unchanged() -> None:
    """Zero relative loss should indicate unchanged performance."""

    result = calculate_relative_loss(
        pre_drift=0.90,
        drifted=0.90,
    )

    assert result == pytest.approx(0.0)


def test_relative_loss_rejects_zero_pre_drift() -> None:
    """Relative loss is undefined when pre-drift performance is zero."""

    with pytest.raises(
        ValueError,
        match="nonzero",
    ):
        calculate_relative_loss(
            pre_drift=0.0,
            drifted=0.50,
        )
