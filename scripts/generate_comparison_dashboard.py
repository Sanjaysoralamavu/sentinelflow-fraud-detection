"""Generate the representative concept-drift comparison dashboard."""

from pathlib import Path

from sentinelflow.evaluation.comparison_dashboard import (
    generate_final_experiment_dashboard,
)


def main() -> None:
    """Generate the comparison dashboard from representative-run artifacts."""
    generate_final_experiment_dashboard(
        separability_batch_metrics=Path(
            "results/final_separability/metrics/evaluation/batch_metrics.csv"
        ),
        separability_phase_summary=Path(
            "results/final_separability/metrics/evaluation/phase_summary.csv"
        ),
        camouflage_batch_metrics=Path(
            "results/final_camouflage/metrics/evaluation/batch_metrics.csv"
        ),
        camouflage_phase_summary=Path(
            "results/final_camouflage/metrics/evaluation/phase_summary.csv"
        ),
        detector_quality_path=Path(
            "results/final_camouflage/metrics/evaluation/detector_quality.csv"
        ),
        drift_events_path=Path(
            "results/final_camouflage/metrics/experiment/drift_events.json"
        ),
        run_manifest_path=Path(
            "results/final_camouflage/metrics/experiment/run_manifest.json"
        ),
        output_path=Path("results/experiment_comparison/index.html"),
    )


if __name__ == "__main__":
    main()