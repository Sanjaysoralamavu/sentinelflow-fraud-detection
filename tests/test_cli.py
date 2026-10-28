"""Command-line interface tests."""

import json
from pathlib import Path
from typing import Any

import pytest

import sentinelflow.cli as cli
from sentinelflow.config import ProjectConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_show_config_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli.main(
        [
            "show-config",
            "--config",
            str(PROJECT_ROOT / "configs" / "full.yaml"),
        ]
    )

    captured = capsys.readouterr()
    output = json.loads(captured.out)

    assert exit_code == 0
    assert output["project"] == "SentinelFlow"
    assert output["random_seed"] == 543
    assert output["target_column"] == "is_fraud"
    assert output["cnp_values"] == ["cnp_online", "cnp_moto"]
    assert output["warmup_rows"] == 500000


def test_run_command_passes_configuration_and_smoke_limit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured_arguments: dict[str, Any] = {}

    class FakeResult:
        def to_dict(self) -> dict[str, Any]:
            return {
                "summary": {
                    "seed": 543,
                    "evaluation_rows": 250,
                    "batch_count": 1,
                },
                "drift_events": [],
                "feature_manifest": {
                    "n_output_features": 12,
                },
            }

    def fake_run_experiment(
        config: ProjectConfig,
        *,
        recorder: object | None = None,
        max_evaluation_rows: int | None = None,
    ) -> FakeResult:
        captured_arguments["seed"] = config.seed
        captured_arguments["recorder"] = recorder
        captured_arguments["max_evaluation_rows"] = max_evaluation_rows

        return FakeResult()

    def fake_write_experiment_artifacts(
        result: FakeResult,
        config: ProjectConfig,
        output_directory: str | Path,
    ) -> dict[str, Path]:
        output = Path(output_directory)

        captured_arguments["artifact_seed"] = config.seed
        captured_arguments["artifact_result"] = result
        captured_arguments["output_directory"] = output

        return {
            "run_manifest": output / "run_manifest.json",
            "feature_manifest": output / "feature_manifest.json",
            "drift_events": output / "drift_events.json",
        }

    def fake_generate_evaluation_outputs(
        recorder: object,
        result: FakeResult,
        output_directory: str | Path,
    ) -> dict[str, Path]:
        output = Path(output_directory)

        captured_arguments["evaluation_recorder"] = recorder
        captured_arguments["evaluation_result"] = result
        captured_arguments["evaluation_output_directory"] = output

        return {
            "batch_metrics": output / "batch_metrics.csv",
            "model_summary": output / "model_summary.csv",
            "f1_timeline_figure": output / "f1_timeline.png",
            "dashboard": output / "dashboard" / "index.html",
        }

    monkeypatch.setattr(
        cli,
        "run_experiment",
        fake_run_experiment,
    )

    monkeypatch.setattr(
        cli,
        "write_experiment_artifacts",
        fake_write_experiment_artifacts,
    )

    monkeypatch.setattr(
        cli,
        "generate_evaluation_outputs",
        fake_generate_evaluation_outputs,
    )

    exit_code = cli.main(
        [
            "run",
            "--config",
            str(PROJECT_ROOT / "configs" / "full.yaml"),
            "--max-evaluation-rows",
            "250",
        ]
    )

    captured = capsys.readouterr()
    output = json.loads(captured.out)

    assert exit_code == 0

    assert captured_arguments["seed"] == 543
    assert captured_arguments["recorder"] is not None
    assert captured_arguments["max_evaluation_rows"] == 250

    assert captured_arguments["artifact_seed"] == 543

    assert isinstance(
        captured_arguments["artifact_result"],
        FakeResult,
    )

    expected_experiment_directory = PROJECT_ROOT / "results" / "metrics" / "experiment"

    assert captured_arguments["output_directory"] == expected_experiment_directory

    expected_evaluation_directory = PROJECT_ROOT / "results" / "metrics" / "evaluation"

    assert captured_arguments["evaluation_output_directory"] == expected_evaluation_directory

    assert captured_arguments["evaluation_recorder"] is captured_arguments["recorder"]

    assert captured_arguments["evaluation_result"] is captured_arguments["artifact_result"]

    assert output["summary"]["seed"] == 543
    assert output["summary"]["evaluation_rows"] == 250
    assert output["summary"]["batch_count"] == 1

    assert output["drift_events"] == []

    assert "feature_manifest" not in output

    assert output["artifacts"] == {
        "run_manifest": str(expected_experiment_directory / "run_manifest.json"),
        "feature_manifest": str(expected_experiment_directory / "feature_manifest.json"),
        "drift_events": str(expected_experiment_directory / "drift_events.json"),
    }

    assert output["evaluation_artifacts"] == {
        "batch_metrics": str(expected_evaluation_directory / "batch_metrics.csv"),
        "model_summary": str(expected_evaluation_directory / "model_summary.csv"),
        "f1_timeline_figure": str(expected_evaluation_directory / "f1_timeline.png"),
        "dashboard": str(expected_evaluation_directory / "dashboard" / "index.html"),
    }


def test_run_command_generates_evaluation_artifacts(
    synthetic_parquet: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Official run command should generate CSV, PNG, and HTML evaluation outputs."""

    from copy import deepcopy

    from sentinelflow.config import ProjectConfig, load_config

    base_config = load_config(PROJECT_ROOT / "configs" / "full.yaml")

    raw = deepcopy(base_config.raw)

    raw["paths"]["dataset"] = str(synthetic_parquet)

    raw["paths"]["results"] = str(tmp_path / "results")

    raw["data"].update(
        {
            "batch_size": 100,
            "warmup_rows": 300,
        }
    )

    raw["stream"].update(
        {
            "label_delay_batches": 2,
            "row_group_shuffle": True,
            "within_batch_shuffle": True,
        }
    )

    raw["drift"].update(
        {
            "enabled": True,
            "start_fraction": 0.25,
            "end_fraction": 0.75,
            "max_intensity": 0.60,
            "random_seed": 1543,
        }
    )

    raw["models"]["random_forest"].update(
        {
            "enabled": True,
            "training_rows": 250,
            "n_estimators": 8,
            "max_depth": 5,
            "min_samples_leaf": 2,
        }
    )

    raw["models"]["adaptive"].update(
        {
            "detector_sample_size": 50,
            "recent_window_rows": 300,
            "ddm_warm_start": 10,
            "detector_grace_batches": 100,
            "detector_cooldown_batches": 3,
        }
    )

    raw["models"]["online_ensemble"].update(
        {
            "enabled": True,
            "n_estimators": 3,
        }
    )

    config = ProjectConfig(
        raw=raw,
        root=base_config.root,
    )

    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _: config,
    )

    exit_code = cli.main(
        [
            "run",
            "--config",
            "synthetic-test.yaml",
            "--max-evaluation-rows",
            "400",
        ]
    )

    captured = capsys.readouterr()
    output = json.loads(captured.out)

    assert exit_code == 0
    assert output["summary"]["evaluation_rows"] == 400
    assert output["summary"]["batch_count"] == 4

    evaluation_artifacts = {
        name: Path(path) for name, path in output["evaluation_artifacts"].items()
    }

    # CSV evaluation artifacts.
    assert evaluation_artifacts["batch_metrics"].is_file()

    assert evaluation_artifacts["phase_summary"].is_file()

    assert evaluation_artifacts["model_summary"].is_file()

    assert evaluation_artifacts["detector_quality"].is_file()

    # Static PNG visualization.
    assert evaluation_artifacts["f1_timeline_figure"].is_file()

    assert evaluation_artifacts["f1_timeline_figure"].suffix == ".png"

    # Interactive HTML dashboard.
    assert evaluation_artifacts["dashboard"].is_file()

    assert evaluation_artifacts["dashboard"].suffix == ".html"

    assert evaluation_artifacts["dashboard"].stat().st_size > 0
