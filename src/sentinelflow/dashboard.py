"""Self-contained interactive dashboard for SentinelFlow evaluation results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio

from sentinelflow.models.common import (
    ALARM_CONFIRMED,
    DriftEvent,
)


def generate_dashboard(
    batch_metrics: pd.DataFrame,
    output_path: Path,
    *,
    metric: str = "f1",
    drift_start_batch: int | None = None,
    drift_events: list[DriftEvent] | tuple[DriftEvent, ...] = (),
) -> Path:
    """Generate a self-contained interactive evaluation dashboard."""

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

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ordered = batch_metrics.sort_values(
        ["model", "batch"]
    ).copy()

    figure = px.line(
        ordered,
        x="batch",
        y=metric,
        color="model",
        markers=True,
        hover_data=["phase"],
        title=f"SentinelFlow — {metric.upper()} Across Batches",
    )

    # Show the stream phases directly on the performance timeline.
    phase_ranges = (
        ordered.groupby("phase")["batch"]
        .agg(["min", "max"])
        .reset_index()
    )

    for _, row in phase_ranges.iterrows():
        figure.add_vrect(
            x0=row["min"] - 0.5,
            x1=row["max"] + 0.5,
            opacity=0.08,
            line_width=0,
            annotation_text=str(row["phase"]),
            annotation_position="top left",
            layer="below",
        )

    # Ground-truth start of injected concept drift.
    if drift_start_batch is not None:
        figure.add_vline(
            x=drift_start_batch,
            line_dash="dash",
            annotation_text="Drift start",
            annotation_position="top",
        )

    # Detector signals are logged even when suppressed. For the main
    # dashboard, show only confirmed alarms that actually triggered recovery.
    confirmed_events = [
        event
        for event in drift_events
        if event.alarm_state == ALARM_CONFIRMED
    ]

    labelled_detectors: set[str] = set()

    for event in confirmed_events:
        detector_name = event.detector.upper()

        annotation = detector_name

        if event.detector in labelled_detectors:
            annotation = ""

        figure.add_vline(
            x=event.source_batch,
            line_dash="dot",
            annotation_text=annotation,
            annotation_position="bottom",
        )

        labelled_detectors.add(event.detector)

    figure.update_layout(
        xaxis_title="Batch",
        yaxis_title=metric.upper(),
        yaxis_range=[0.0, 1.05],
        hovermode="x unified",
        legend_title_text="Model",
    )

    html = pio.to_html(
        figure,
        full_html=True,
        include_plotlyjs=True,
    )

    output_path.write_text(
        html,
        encoding="utf-8",
    )

    return output_path