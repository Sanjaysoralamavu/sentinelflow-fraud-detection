"""Self-contained interactive dashboard for SentinelFlow evaluation results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
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
    """Generate a self-contained dashboard with cumulative batch replay."""

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
        ["batch", "model"]
    ).copy()

    batches = sorted(ordered["batch"].unique())
    models = sorted(ordered["model"].unique())

    figure = go.Figure()

    first_batch = batches[0]

    # Initial state: only the first batch is visible.
    for model_name in models:
        model_frame = ordered[
            (ordered["model"] == model_name)
            & (ordered["batch"] <= first_batch)
        ]

        figure.add_trace(
            go.Scatter(
                x=model_frame["batch"],
                y=model_frame[metric],
                mode="lines+markers",
                name=model_name,
                customdata=model_frame[["phase"]],
                hovertemplate=(
                    "Batch: %{x}<br>"
                    f"{metric.upper()}: %{{y:.4f}}<br>"
                    "Phase: %{customdata[0]}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )

    # One cumulative frame per batch.
    frames: list[go.Frame] = []

    for current_batch in batches:
        frame_data = []

        for model_name in models:
            model_frame = ordered[
                (ordered["model"] == model_name)
                & (ordered["batch"] <= current_batch)
            ]

            frame_data.append(
                go.Scatter(
                    x=model_frame["batch"],
                    y=model_frame[metric],
                    mode="lines+markers",
                    name=model_name,
                    customdata=model_frame[["phase"]],
                    hovertemplate=(
                        "Batch: %{x}<br>"
                        f"{metric.upper()}: %{{y:.4f}}<br>"
                        "Phase: %{customdata[0]}"
                        "<extra>%{fullData.name}</extra>"
                    ),
                )
            )

        frames.append(
            go.Frame(
                data=frame_data,
                name=str(current_batch),
            )
        )

    figure.frames = frames

    # Phase shading remains fixed while the slider moves.
    phase_ranges = (
        ordered.groupby("phase")["batch"]
        .agg(["min", "max"])
        .reset_index()
    )

    for _, row in phase_ranges.iterrows():
        figure.add_vrect(
            x0=float(row["min"]) - 0.5,
            x1=float(row["max"]) + 0.5,
            opacity=0.08,
            line_width=0,
            annotation_text=str(row["phase"]),
            annotation_position="top left",
            layer="below",
        )

    # Ground-truth drift start.
    if drift_start_batch is not None:
        figure.add_vline(
            x=drift_start_batch,
            line_dash="dash",
            annotation_text="Drift start",
            annotation_position="top",
        )

    # Confirmed detector alarms.
    confirmed_events = [
        event
        for event in drift_events
        if event.alarm_state == ALARM_CONFIRMED
    ]

    labelled_detectors: set[str] = set()

    for event in confirmed_events:
        annotation = ""

        if event.detector not in labelled_detectors:
            annotation = event.detector.upper()
            labelled_detectors.add(event.detector)

        figure.add_vline(
            x=event.source_batch,
            line_dash="dot",
            annotation_text=annotation,
            annotation_position="bottom",
        )

    # Slider moves directly between cumulative frames.
    slider_steps = [
        {
            "method": "animate",
            "label": str(batch),
            "args": [
                [str(batch)],
                {
                    "mode": "immediate",
                    "frame": {
                        "duration": 0,
                        "redraw": True,
                    },
                    "transition": {
                        "duration": 0,
                    },
                },
            ],
        }
        for batch in batches
    ]

    figure.update_layout(
        title=f"SentinelFlow — {metric.upper()} Experiment Replay",
        xaxis={
            "title": "Batch",
            "range": [
                float(min(batches)) - 0.5,
                float(max(batches)) + 0.5,
            ],
        },
        yaxis={
            "title": metric.upper(),
            "range": [0.0, 1.05],
        },
        hovermode="x unified",
        legend={
            "title": {
                "text": "Model",
            },
        },
        sliders=[
            {
                "active": 0,
                "currentvalue": {
                    "prefix": "Batch: ",
                },
                "pad": {
                    "t": 50,
                },
                "steps": slider_steps,
            }
        ],
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