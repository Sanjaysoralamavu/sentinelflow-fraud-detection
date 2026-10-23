"""Self-contained interactive dashboard for SentinelFlow evaluation results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio


def generate_dashboard(
    batch_metrics: pd.DataFrame,
    output_path: Path,
    *,
    metric: str = "f1",
    drift_start_batch: int | None = None,
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

    if drift_start_batch is not None:
        figure.add_vline(
            x=drift_start_batch,
            line_dash="dash",
            annotation_text="Drift start",
            annotation_position="top",
        )

    figure.update_layout(
        xaxis_title="Batch",
        yaxis_title=metric.upper(),
        yaxis_range=[0.0, 1.05],
        hovermode="x unified",
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