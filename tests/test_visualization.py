"""Tests for evaluation visualization."""

import pandas as pd

from sentinelflow.evaluation.visualization import (
    plot_drift_timeline,
    plot_model_comparison,
)


def test_plot_model_comparison_creates_figure(tmp_path) -> None:
    """Model comparison should generate a nonempty image file."""

    frame = pd.DataFrame(
        {
            "model": [
                "static_logistic",
                "adaptive_adwin",
            ],
            "precision": [0.80, 0.90],
            "recall": [0.70, 0.88],
            "f1": [0.75, 0.89],
            "average_precision": [0.82, 0.93],
            "roc_auc": [0.86, 0.95],
        }
    )

    output_path = plot_model_comparison(
        frame,
        tmp_path / "model_comparison.png",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_drift_timeline_creates_figure(tmp_path) -> None:
    """Drift timeline should plot batch performance and drift start."""

    frame = pd.DataFrame(
        {
            "model": [
                "static_logistic",
                "static_logistic",
                "adaptive_adwin",
                "adaptive_adwin",
            ],
            "batch": [
                0,
                1,
                0,
                1,
            ],
            "phase": [
                "pre_drift",
                "ramp_drift",
                "pre_drift",
                "ramp_drift",
            ],
            "f1": [
                0.90,
                0.65,
                0.89,
                0.85,
            ],
        }
    )

    output_path = plot_drift_timeline(
        frame,
        tmp_path / "drift_timeline.png",
        metric="f1",
        drift_start_batch=1,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
