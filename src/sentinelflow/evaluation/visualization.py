"""Static evaluation figures for SentinelFlow experiment results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sentinelflow.models.common import (
    ALARM_CONFIRMED,
    DriftEvent,
)


def _metric_axis_limits(
    values: pd.Series,
) -> tuple[float, float]:
    """Return readable y-axis limits around observed metric values."""

    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return 0.0, 1.0

    minimum = float(numeric.min())
    maximum = float(numeric.max())
    span = maximum - minimum

    padding = max(
        span * 0.15,
        0.001,
    )

    lower = max(
        0.0,
        minimum - padding,
    )

    upper = min(
        1.0,
        maximum + padding,
    )

    if lower >= upper:
        lower = max(
            0.0,
            minimum - 0.001,
        )
        upper = min(
            1.0,
            maximum + 0.001,
        )

    return lower, upper


def plot_model_comparison(
    model_summary: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot overall fraud-detection metrics for every model."""

    required_columns = {
        "model",
        "precision",
        "recall",
        "f1",
        "average_precision",
        "roc_auc",
    }

    missing = required_columns.difference(model_summary.columns)

    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Model summary is missing required columns: {names}")

    if model_summary.empty:
        raise ValueError("Model summary cannot be empty.")

    plot_frame = model_summary.set_index("model")[
        [
            "precision",
            "recall",
            "f1",
            "average_precision",
            "roc_auc",
        ]
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    axis = plot_frame.plot(
        kind="bar",
        figsize=(12, 6),
    )

    axis.set_title("Overall Model Performance")
    axis.set_xlabel("Model")
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.05)

    axis.legend(
        title="Metric",
        loc="lower right",
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


def plot_drift_timeline(
    batch_metrics: pd.DataFrame,
    output_path: Path,
    *,
    metric: str = "f1",
    drift_start_batch: int | None = None,
    drift_events: list[DriftEvent] | tuple[DriftEvent, ...] = (),
) -> Path:
    """Plot one model metric across evaluation batches."""

    required_columns = {
        "model",
        "batch",
        "phase",
        metric,
    }

    missing = required_columns.difference(batch_metrics.columns)

    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Batch metrics are missing required columns: {names}")

    if batch_metrics.empty:
        raise ValueError("Batch metrics cannot be empty.")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(figsize=(12, 6))

    for model_name, frame in batch_metrics.groupby("model"):
        ordered = frame.sort_values("batch")

        axis.plot(
            ordered["batch"],
            ordered[metric],
            label=model_name,
        )

    if drift_start_batch is not None:
        axis.axvline(
            drift_start_batch,
            linestyle="--",
            label="Drift start",
        )

    confirmed_events = [event for event in drift_events if event.alarm_state == ALARM_CONFIRMED]

    labelled_detectors: set[str] = set()

    for event in confirmed_events:
        label = None

        if event.detector not in labelled_detectors:
            label = f"{event.detector.upper()} alarm"
            labelled_detectors.add(event.detector)

        axis.axvline(
            event.source_batch,
            linestyle=":",
            alpha=0.7,
            label=label,
        )

    lower, upper = _metric_axis_limits(batch_metrics[metric])

    display_name = (
        "Precision@100" if metric == "precision_at_100" else metric.replace("_", "-").upper()
    )

    axis.set_title(f"{display_name} Across Evaluation Batches")

    axis.set_xlabel("Batch")
    axis.set_ylabel(display_name)

    axis.set_ylim(
        lower,
        upper,
    )

    axis.legend(
        loc="best",
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def plot_relative_performance_loss(
    relative_loss: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot relative performance loss after controlled drift."""

    required_columns = {
        "model",
        "metric",
        "pre_drift",
        "drifted",
        "relative_loss",
    }

    missing = required_columns.difference(relative_loss.columns)

    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Relative performance loss is missing required columns: {names}")

    if relative_loss.empty:
        raise ValueError("Relative performance loss cannot be empty.")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metric_order = [
        "f1",
        "precision",
        "recall",
        "roc_auc",
        "average_precision",
        "precision_at_100",
    ]

    available_metrics = [
        metric for metric in metric_order if metric in set(relative_loss["metric"])
    ]

    plot_frame = relative_loss.pivot(
        index="model",
        columns="metric",
        values="relative_loss",
    ).reindex(columns=available_metrics)

    display_names = {
        "f1": "F1",
        "precision": "Precision",
        "recall": "Recall",
        "roc_auc": "ROC-AUC",
        "average_precision": "Average Precision",
        "precision_at_100": "P@100",
    }

    plot_frame = plot_frame.rename(columns=display_names)

    axis = plot_frame.plot(
        kind="bar",
        figsize=(14, 7),
    )

    axis.axhline(
        0.0,
        linestyle="--",
    )

    axis.set_title("Relative Performance Loss After Controlled Drift")

    axis.set_xlabel("Model")
    axis.set_ylabel("Relative Loss (positive = degradation, negative = improvement)")

    axis.legend(
        title="Metric",
        loc="best",
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    return output_path
