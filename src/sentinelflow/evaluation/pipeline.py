"""End-to-end evaluation output generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sentinelflow.dashboard import generate_dashboard
from sentinelflow.evaluation.adaptation import (
    evaluate_relative_performance_loss,
)
from sentinelflow.evaluation.detectors import evaluate_all_detectors
from sentinelflow.evaluation.export import (
    export_batch_metrics,
    export_detector_quality,
    export_model_summary,
    export_phase_summary,
    export_relative_performance_loss,
)
from sentinelflow.evaluation.recorder import EvaluationRecorder
from sentinelflow.evaluation.visualization import (
    plot_drift_timeline,
    plot_model_comparison,
    plot_relative_performance_loss,
)
from sentinelflow.experiment import ExperimentResult


TIMELINE_METRICS = (
    "f1",
    "precision",
    "recall",
    "fpr",
    "roc_auc",
    "pr_auc",
    "precision_at_100",
)

RELATIVE_LOSS_METRICS = (
    "f1",
    "precision",
    "recall",
    "roc_auc",
    "pr_auc",
    "precision_at_100",
)

DASHBOARD_METRICS = (
    "f1",
    "precision",
    "recall",
    "fpr",
    "roc_auc",
    "pr_auc",
)


def generate_evaluation_outputs(
    recorder: EvaluationRecorder,
    result: ExperimentResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Generate evaluation CSV and figure artifacts for one experiment."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    detector_quality = evaluate_all_detectors(
        result.drift_events,
        drift_start_batch=recorder.drift_start_batch,
        expected_detectors=(
            ("adaptive_adwin", "adwin"),
            ("adaptive_ddm", "ddm"),
        ),
    )

    # Compare pre-drift and fully drifted performance for metrics
    # where a larger value represents better model performance.
    relative_performance_loss = []

    for metric in RELATIVE_LOSS_METRICS:
        relative_performance_loss.extend(
            evaluate_relative_performance_loss(
                recorder,
                metric=metric,
            )
        )

    paths = {
        "batch_metrics": export_batch_metrics(
            recorder,
            output_dir / "batch_metrics.csv",
        ),
        "phase_summary": export_phase_summary(
            recorder,
            output_dir / "phase_summary.csv",
        ),
        "model_summary": export_model_summary(
            recorder,
            output_dir / "model_summary.csv",
        ),
        "detector_quality": export_detector_quality(
            detector_quality,
            output_dir / "detector_quality.csv",
        ),
        "relative_performance_loss": (
            export_relative_performance_loss(
                relative_performance_loss,
                output_dir
                / "relative_performance_loss.csv",
            )
        ),
    }

    model_summary_frame = pd.read_csv(
        paths["model_summary"]
    )

    batch_metrics_frame = pd.read_csv(
        paths["batch_metrics"]
    )

    paths["model_comparison_figure"] = (
        plot_model_comparison(
            model_summary_frame,
            output_dir / "model_comparison.png",
        )
    )

    # Generate one batch-level timeline for every evaluation metric.
    for metric in TIMELINE_METRICS:
        path_key = f"{metric}_timeline_figure"

        paths[path_key] = plot_drift_timeline(
            batch_metrics_frame,
            output_dir / f"{metric}_timeline.png",
            metric=metric,
            drift_start_batch=recorder.drift_start_batch,
            drift_events=result.drift_events,
        )

    # Keep the original artifact name for compatibility.
    paths["drift_timeline_figure"] = paths[
        "f1_timeline_figure"
    ]

    # Generate relative-loss visualization when both pre-drift
    # and fully drifted phases are available.
    if relative_performance_loss:
        relative_loss_frame = pd.read_csv(
            paths["relative_performance_loss"]
        )

        paths["relative_performance_loss_figure"] = (
            plot_relative_performance_loss(
                relative_loss_frame,
                output_dir
                / "relative_performance_loss.png",
            )
        )

    # Generate interactive replay dashboards.
    for metric in DASHBOARD_METRICS:
        dashboard_name = (
            "index.html"
            if metric == "f1"
            else f"{metric}.html"
        )

        paths[f"{metric}_dashboard"] = (
            generate_dashboard(
                batch_metrics_frame,
                output_dir
                / "dashboard"
                / dashboard_name,
                metric=metric,
                drift_start_batch=(
                    recorder.drift_start_batch
                ),
                drift_events=result.drift_events,
            )
        )

    # Keep the original dashboard artifact name for compatibility.
    paths["dashboard"] = paths[
        "f1_dashboard"
    ]

    return paths