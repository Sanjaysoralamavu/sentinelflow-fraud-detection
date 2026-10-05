import json
from pathlib import Path

import pytest

from sentinelflow.cli import main

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_show_config_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
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
