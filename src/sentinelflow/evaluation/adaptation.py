"""Adaptation evaluation across controlled-drift phases."""

from __future__ import annotations

from dataclasses import dataclass

from sentinelflow.evaluation.recorder import EvaluationRecorder

PRE_DRIFT = "pre_drift"
DRIFTED = "drifted"


@dataclass(frozen=True)
class RelativePerformanceLoss:
    """Relative metric loss from pre-drift to fully drifted data."""

    model: str
    metric: str
    pre_drift: float
    drifted: float
    relative_loss: float


def calculate_relative_loss(
    pre_drift: float,
    drifted: float,
) -> float:
    """Return relative performance loss after drift."""

    if pre_drift == 0.0:
        raise ValueError("Pre-drift performance must be nonzero.")

    return (pre_drift - drifted) / pre_drift


def evaluate_relative_performance_loss(
    recorder: EvaluationRecorder,
    *,
    metric: str = "precision_at_100",
) -> list[RelativePerformanceLoss]:
    """Compare model performance before and after controlled drift."""

    phase_metrics = recorder.phase_metrics()

    model_names = sorted({model_name for model_name, _ in phase_metrics})

    results: list[RelativePerformanceLoss] = []

    for model_name in model_names:
        pre_key = (
            model_name,
            PRE_DRIFT,
        )
        drifted_key = (
            model_name,
            DRIFTED,
        )

        if pre_key not in phase_metrics:
            continue

        if drifted_key not in phase_metrics:
            continue

        pre_metrics = phase_metrics[pre_key]
        drifted_metrics = phase_metrics[drifted_key]

        if not hasattr(pre_metrics, metric):
            raise ValueError(f"Unknown evaluation metric: {metric}")

        pre_value = float(getattr(pre_metrics, metric))

        drifted_value = float(getattr(drifted_metrics, metric))

        relative_loss = calculate_relative_loss(
            pre_value,
            drifted_value,
        )

        results.append(
            RelativePerformanceLoss(
                model=model_name,
                metric=metric,
                pre_drift=pre_value,
                drifted=drifted_value,
                relative_loss=relative_loss,
            )
        )

    return results
