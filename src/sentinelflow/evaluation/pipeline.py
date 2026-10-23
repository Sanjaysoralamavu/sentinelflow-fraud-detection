"""End-to-end evaluation output generation."""

from __future__ import annotations

from pathlib import Path
from sentinelflow.dashboard import generate_dashboard
import pandas as pd

from sentinelflow.evaluation.detectors import evaluate_all_detectors
from sentinelflow.evaluation.export import (
    export_batch_metrics,
    export_detector_quality,
    export_model_summary,
    export_phase_summary,
)
from sentinelflow.evaluation.recorder import EvaluationRecorder
from sentinelflow.evaluation.visualization import (
    plot_drift_timeline,
    plot_model_comparison,
)
from sentinelflow.experiment import ExperimentResult


def generate_evaluation_outputs(
    recorder: EvaluationRecorder,
    result: ExperimentResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Generate evaluation CSV and figure artifacts for one experiment."""

    output_dir.mkdir(parents=True, exist_ok=True)

    detector_quality = evaluate_all_detectors(
        result.drift_events,
        drift_start_batch=recorder.drift_start_batch,
        expected_detectors=(
            ("adaptive_adwin", "adwin"),
            ("adaptive_ddm", "ddm"),
        ),
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
    }

    model_summary_frame = pd.read_csv(
        paths["model_summary"]
    )
    batch_metrics_frame = pd.read_csv(
        paths["batch_metrics"]
    )

    paths["model_comparison_figure"] = plot_model_comparison(
        model_summary_frame,
        output_dir / "model_comparison.png",
    )

    paths["drift_timeline_figure"] = plot_drift_timeline(
        batch_metrics_frame,
        output_dir / "drift_timeline.png",
        metric="f1",
        drift_start_batch=recorder.drift_start_batch,
        drift_events=result.drift_events,
    )

    paths["dashboard"] = generate_dashboard(
        batch_metrics_frame,
        output_dir / "dashboard" / "index.html",
        metric="f1",
        drift_start_batch=recorder.drift_start_batch,
        drift_events=result.drift_events,
    )

    return paths