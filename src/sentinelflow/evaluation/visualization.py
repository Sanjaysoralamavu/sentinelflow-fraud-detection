"""Static evaluation figures for SentinelFlow experiment results."""

from __future__ import annotations

from pathlib import Path
from sentinelflow.models.common import (
    ALARM_CONFIRMED,
    DriftEvent,
)
import matplotlib.pyplot as plt
import pandas as pd


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
        "pr_auc",
        "roc_auc",
    }

    missing = required_columns.difference(model_summary.columns)

    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(
            f"Model summary is missing required columns: {names}"
        )

    if model_summary.empty:
        raise ValueError("Model summary cannot be empty.")

    plot_frame = model_summary.set_index("model")[
        [
            "precision",
            "recall",
            "f1",
            "pr_auc",
            "roc_auc",
        ]
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

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
    """Plot model performance across batches with controlled drift marked."""

    required_columns = {
        "model",
        "batch",
        "phase",
        metric,
    }

    missing = required_columns.difference(batch_metrics.columns)

    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(
            f"Batch metrics are missing required columns: {names}"
        )

    if batch_metrics.empty:
        raise ValueError("Batch metrics cannot be empty.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    confirmed_events = [
        event
        for event in drift_events
        if event.alarm_state == ALARM_CONFIRMED
    ]

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

    axis.set_title(
        f"{metric.upper()} Across Evaluation Batches"
    )
    axis.set_xlabel("Batch")
    axis.set_ylabel(metric.upper())
    axis.set_ylim(0.0, 1.05)

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