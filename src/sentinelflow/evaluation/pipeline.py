"""End-to-end evaluation output generation."""

from __future__ import annotations

from pathlib import Path

from sentinelflow.evaluation.detectors import evaluate_all_detectors
from sentinelflow.evaluation.export import (
    export_batch_metrics,
    export_detector_quality,
    export_model_summary,
    export_phase_summary,
)
from sentinelflow.evaluation.recorder import EvaluationRecorder
from sentinelflow.experiment import ExperimentResult


def generate_evaluation_outputs(
    recorder: EvaluationRecorder,
    result: ExperimentResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Generate evaluation CSV artifacts for one completed experiment."""

    output_dir.mkdir(parents=True, exist_ok=True)

    detector_quality = evaluate_all_detectors(
        result.drift_events,
        drift_start_batch=recorder.drift_start_batch,
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

    return paths