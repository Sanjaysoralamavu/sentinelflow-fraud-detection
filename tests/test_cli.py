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

    monkeypatch.setattr(
        cli,
        "run_experiment",
        fake_run_experiment,
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
    assert captured_arguments == {
        "seed": 543,
        "recorder": None,
        "max_evaluation_rows": 250,
    }
    assert output["summary"]["seed"] == 543
    assert output["summary"]["evaluation_rows"] == 250
    assert output["summary"]["batch_count"] == 1
    assert output["drift_events"] == []
    assert output["feature_manifest"]["n_output_features"] == 12
