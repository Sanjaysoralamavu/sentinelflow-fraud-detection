"""Tests for the experiment scenario comparison dashboard."""

from pathlib import Path

import pandas as pd

from sentinelflow.evaluation.comparison_dashboard import (
    METRICS,
    generate_final_experiment_dashboard,
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


def test_generate_final_experiment_dashboard(tmp_path: Path) -> None:
    models = [
        "static_random_forest",
        "adaptive_adwin",
        "adaptive_ddm",
    ]
    phases = [
        "pre_drift",
        "ramp_drift",
        "drifted",
    ]

    rows = []

    for phase_index, phase in enumerate(phases):
        for model_index, model in enumerate(models):
            row = {
                "model": model,
                "phase": phase,
            }

            for metric in METRICS:
                if metric == "fpr":
                    row[metric] = 0.01 + (phase_index * 0.01)
                else:
                    row[metric] = 0.99 - (phase_index * 0.10) - (model_index * 0.01)

            rows.append(row)

    initial = pd.DataFrame(rows)
    revised = pd.DataFrame(rows)

    batch_metrics = pd.DataFrame(
        {
            "batch": [0, 1],
            "model": ["static_random_forest", "adaptive_adwin"],
        }
    )

    detector_quality = pd.DataFrame(
        {
            "model": ["adaptive_adwin", "adaptive_ddm"],
            "detector": ["adwin", "ddm"],
            "total_alarms": [1, 1],
            "false_alarms": [0, 0],
        }
    )

    initial_batch_path = tmp_path / "initial_batch.csv"
    initial_phase_path = tmp_path / "initial_phase.csv"
    revised_batch_path = tmp_path / "revised_batch.csv"
    revised_phase_path = tmp_path / "revised_phase.csv"
    detector_quality_path = tmp_path / "detector_quality.csv"
    drift_events_path = tmp_path / "drift_events.json"
    run_manifest_path = tmp_path / "run_manifest.json"
    output_path = tmp_path / "index.html"

    batch_metrics.to_csv(initial_batch_path, index=False)
    initial.to_csv(initial_phase_path, index=False)
    batch_metrics.to_csv(revised_batch_path, index=False)
    revised.to_csv(revised_phase_path, index=False)
    detector_quality.to_csv(detector_quality_path, index=False)

    drift_events_path.write_text(
        """
[
  {
    "model": "adaptive_adwin",
    "source_batch": 17,
    "observed_batch": 24,
    "retrain_rows": 200000
  },
  {
    "model": "adaptive_ddm",
    "source_batch": 18,
    "observed_batch": 26,
    "retrain_rows": 150000
  }
]
""".strip(),
        encoding="utf-8",
    )

    run_manifest_path.write_text(
        """
{
  "configuration": {
    "models": {
      "adaptive": {
        "recent_window_rows": 200000
      }
    },
    "stream": {
      "label_delay_batches": 7
    }
  },
  "summary": {
    "batch_count": 30,
    "evaluation_rows": 1500000,
    "seed": 543
  }
}
""".strip(),
        encoding="utf-8",
    )

    result = generate_final_experiment_dashboard(
        separability_batch_metrics=initial_batch_path,
        separability_phase_summary=initial_phase_path,
        camouflage_batch_metrics=revised_batch_path,
        camouflage_phase_summary=revised_phase_path,
        detector_quality_path=detector_quality_path,
        drift_events_path=drift_events_path,
        run_manifest_path=run_manifest_path,
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()

    html = output_path.read_text(encoding="utf-8")

    assert "Results Between the Two Evaluations" in html
    assert "Adaptive ADWIN" in html
    assert "Adaptive DDM" in html

    # Detector details must come from the supplied artifacts.
    assert "Source batch:</strong> 17" in html
    assert "Observed batch:</strong> 24" in html
    assert "Detection delay:</strong> 7 batches" in html
    assert "200,000 recent labeled rows" in html

    assert "Source batch:</strong> 18" in html
    assert "Observed batch:</strong> 26" in html
    assert "Detection delay:</strong> 8 batches" in html
    assert "150,000 recent labeled rows" in html
    assert "Seed 543" in html
    assert "1,500,000 evaluation rows" in html
    assert "30 batches" in html
    assert "complete three-seed study" in html
