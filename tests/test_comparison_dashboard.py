"""Tests for the experiment scenario comparison dashboard."""

from pathlib import Path

import pandas as pd

from sentinelflow.evaluation.comparison_dashboard import (
    generate_scenario_comparison_dashboard,
)


def test_generate_scenario_comparison_dashboard(tmp_path: Path) -> None:
    phase_one = pd.DataFrame(
        {
            "model": [
                "static_random_forest",
                "adaptive_adwin",
                "adaptive_ddm",
            ],
            "recall": [
                0.99,
                0.98,
                0.97,
            ],
        }
    )

    phase_two = pd.DataFrame(
        {
            "model": [
                "static_random_forest",
                "adaptive_adwin",
                "adaptive_ddm",
            ],
            "recall": [
                0.61,
                0.92,
                0.91,
            ],
        }
    )

    phase_one_path = tmp_path / "phase_one.csv"
    phase_two_path = tmp_path / "phase_two.csv"

    phase_one.to_csv(
        phase_one_path,
        index=False,
    )
    phase_two.to_csv(
        phase_two_path,
        index=False,
    )

    output_path = tmp_path / "comparison.html"

    result = generate_scenario_comparison_dashboard(
        phase_one_path,
        phase_two_path,
        output_path,
        phase_one_name="Original Drift",
        phase_two_name="Camouflage Drift",
        metric="recall",
    )

    assert result == output_path
    assert output_path.exists()

    html = output_path.read_text(
        encoding="utf-8",
    )

    assert "Original Drift" in html
    assert "Camouflage Drift" in html
    assert "Static Random Forest" in html
    assert "Adaptive ADWIN" in html
    assert "Adaptive DDM" in html
