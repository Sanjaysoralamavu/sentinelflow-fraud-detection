"""CSV export helpers for evaluation results."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from sentinelflow.evaluation.adaptation import (
    RelativePerformanceLoss,
)
from sentinelflow.evaluation.detectors import DetectorQuality
from sentinelflow.evaluation.recorder import EvaluationRecorder


def export_batch_metrics(
    recorder: EvaluationRecorder,
    output_path: Path,
) -> Path:
    """Write per-batch model metrics to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(recorder.records)
    frame.to_csv(output_path, index=False)

    return output_path


def export_model_summary(
    recorder: EvaluationRecorder,
    output_path: Path,
) -> Path:
    """Write overall metrics for every model to CSV."""

    rows: list[dict[str, object]] = []

    for model_name, metrics in recorder.overall_metrics().items():
        rows.append(
            {
                "model": model_name,
                **asdict(metrics),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)

    return output_path


def export_phase_summary(
    recorder: EvaluationRecorder,
    output_path: Path,
) -> Path:
    """Write per-phase metrics for every model to CSV."""

    rows: list[dict[str, object]] = []

    for (model_name, phase), metrics in recorder.phase_metrics().items():
        rows.append(
            {
                "model": model_name,
                "phase": phase,
                **asdict(metrics),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)

    return output_path


def export_detector_quality(
    results: list[DetectorQuality],
    output_path: Path,
) -> Path:
    """Write detector-quality summaries to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [asdict(result) for result in results]
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)

    return output_path


def export_relative_performance_loss(
    results: list[RelativePerformanceLoss],
    output_path: Path,
) -> Path:
    """Write relative performance-loss results to CSV."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = [asdict(result) for result in results]

    frame = pd.DataFrame(rows)

    frame.to_csv(
        output_path,
        index=False,
    )

    return output_path
