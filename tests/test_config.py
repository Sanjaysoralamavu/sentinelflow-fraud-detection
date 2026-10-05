from pathlib import Path

import pytest
import yaml

from sentinelflow.config import ConfigurationError, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_full_config() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "full.yaml")

    assert config.seed == 543
    assert config.batch_size == 50000
    assert config.warmup_rows == 500000
    assert config.label_delay_batches == 7
    assert config.target_column == "is_fraud"
    assert config.cnp_column == "card_present_cnp"
    assert config.cnp_values == ("cnp_online", "cnp_moto")
    assert (
        config.dataset_path
        == (PROJECT_ROOT / "data" / "raw" / "credit_card_fraud.parquet").resolve()
    )


def test_rejects_invalid_drift_fraction(tmp_path: Path) -> None:
    source_path = PROJECT_ROOT / "configs" / "full.yaml"

    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["drift"]["start_fraction"] = 0.90
    raw["drift"]["end_fraction"] = 0.50

    invalid_path = tmp_path / "invalid.yaml"

    with invalid_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle)

    with pytest.raises(ConfigurationError):
        load_config(invalid_path)
