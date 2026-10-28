"""Issue #9: full and reduced feature configurations, and the value-level signal screen."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from conftest import make_frame, write_parquet

from sentinelflow.config import load_config
from sentinelflow.data.loader import DataSettings, frame_hash, iter_cnp_chunks
from sentinelflow.data.preprocessing import (
    FraudPreprocessor,
    infer_column_roles,
    parse_exclusions,
    write_feature_manifest,
)
from sentinelflow.data.signal_audit import screen_signals, write_signal_outputs

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "configs" / "full.yaml"
ABLATIONS = {
    "ablation_reduced": ROOT / "configs" / "ablation_reduced.yaml",
    "ablation_strict": ROOT / "configs" / "ablation_strict.yaml",
}
REDUCED = {
    "session_duration_bin",
    "session_duration_sec",
    "geo_velocity_bin",
    "geo_velocity_kmh",
    "geo_distance_bin",
    "geo_distance_km",
    "merchant_category_vs_history",
    "chargeback_history_count",
}
STRICT = REDUCED | {"cards_on_device_30d", "failed_attempts_before_success"}
PROTECTED = {"is_fraud", "customer_id", "transaction_id"}


def _raw(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _exclusions(path: Path) -> dict[str, str]:
    return _raw(path)["data"]["preprocessing"]["exclude_columns"]


def _real_schema() -> list[tuple[str, str]]:
    """Column names and Arrow types of the verified file, from the committed audit."""

    with (ROOT / "results" / "metrics" / "audit" / "column_profile.csv").open() as handle:
        return [(row["column"], row["arrow_type"]) for row in csv.DictReader(handle)]


@pytest.mark.parametrize("name", sorted(ABLATIONS))
def test_ablation_differs_from_full_only_in_exclusions_and_output(name: str) -> None:
    full, ablation = _raw(FULL), _raw(ABLATIONS[name])

    preprocessing = ablation["data"].pop("preprocessing")
    output = ablation["paths"].pop("experiment_output")

    assert ablation == full
    assert set(preprocessing) == {"exclude_columns"}
    assert output == f"results/metrics/experiment/{name}"


def test_exclusion_sets_match_the_documented_audit() -> None:
    assert set(_exclusions(ABLATIONS["ablation_reduced"])) == REDUCED
    assert set(_exclusions(ABLATIONS["ablation_strict"])) == STRICT
    for path in ABLATIONS.values():
        assert all(reason.strip() for reason in _exclusions(path).values())


def test_full_configuration_is_unchanged() -> None:
    full = _raw(FULL)
    assert "preprocessing" not in full["data"]
    assert "experiment_output" not in full["paths"]
    assert (
        load_config(FULL).experiment_output_path
        == (ROOT / "results" / "metrics" / "experiment").resolve()
    )


def test_every_excluded_column_exists_in_the_verified_dataset() -> None:
    columns = {name for name, _ in _real_schema()}
    assert STRICT <= columns
    assert not STRICT & PROTECTED


@pytest.fixture
def real_schema_parquet(tmp_path: Path) -> Path:
    """Synthetic rows with the verified file's exact column names and types."""

    rng = np.random.default_rng(9)
    rows = 3_000
    data: dict[str, object] = {}
    for name, arrow_type in _real_schema():
        if name == "is_fraud":
            data[name] = (rng.random(rows) < 0.15).astype(np.int8)
        elif name == "card_present_cnp":
            data[name] = rng.choice(["cnp_online", "cnp_moto", "card_present_chip"], rows)
        elif arrow_type in {"string", "large_string"}:
            data[name] = rng.choice([f"{name}_a", f"{name}_b", f"{name}_c"], rows)
        elif arrow_type.startswith("int"):
            data[name] = rng.integers(0, 10, rows)
        else:
            data[name] = rng.gamma(2.0, 10.0, rows)
    return write_parquet(pd.DataFrame(data), tmp_path / "data" / "raw" / "cc.parquet", 500)


def _config_for(source: Path, dataset: Path, tmp_path: Path) -> Path:
    raw = _raw(source)
    raw["paths"]["dataset"] = str(dataset)
    raw["data"]["warmup_rows"] = 1_000
    path = tmp_path / "configs" / source.name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


@pytest.mark.parametrize("source", [FULL, *ABLATIONS.values()], ids=lambda p: p.stem)
def test_preprocessing_works_with_each_configuration(
    source: Path, real_schema_parquet: Path, tmp_path: Path
) -> None:
    config_path = _config_for(source, real_schema_parquet, tmp_path)
    manifest = write_feature_manifest(config_path, tmp_path / "manifest.json")
    record = json.loads((tmp_path / "manifest.json").read_text())

    expected = set() if source == FULL else set(_exclusions(source))
    assert set(manifest.excluded) == expected
    assert PROTECTED.isdisjoint(record["input_columns"])
    assert PROTECTED.isdisjoint(record["included"])
    assert expected.isdisjoint(record["included"])
    for column in expected:
        assert record["dropped"][column].startswith("excluded by configuration: ")
        stems = [name.split("__", 1)[1] for name in record["output_features"]]
        assert not any(stem.startswith(column) for stem in stems)
    assert record["n_output_features"] > 0


def test_ablation_does_not_change_the_stream(real_schema_parquet: Path, tmp_path: Path) -> None:
    """Exclusion happens after loading and drift, so every batch is byte-identical."""

    def hashes(source: Path) -> list[str]:
        config = load_config(_config_for(source, real_schema_parquet, tmp_path / source.stem))
        return [
            frame_hash(chunk.features, chunk.labels)
            for chunk in iter_cnp_chunks(DataSettings.from_config(config))
        ]

    full = hashes(FULL)
    for path in ABLATIONS.values():
        assert hashes(path) == full


def test_experiment_output_override(real_schema_parquet: Path, tmp_path: Path) -> None:
    config = load_config(_config_for(ABLATIONS["ablation_strict"], real_schema_parquet, tmp_path))
    assert config.experiment_output_path == (
        tmp_path / "results" / "metrics" / "experiment" / "ablation_strict"
    )


def test_parse_exclusions_accepts_list_and_mapping() -> None:
    assert parse_exclusions(None) == ((), {})
    assert parse_exclusions(["a", "b"]) == (("a", "b"), {})
    assert parse_exclusions({"a": "why"}) == (("a",), {"a": "why"})
    with pytest.raises(ValueError, match="list or a mapping"):
        parse_exclusions("a")


def test_unknown_excluded_column_is_rejected() -> None:
    frame = pd.DataFrame({"amount": [1.0, 2.0, 3.0], "kind": list("xyx")})
    with pytest.raises(ValueError, match="not in the features: amonut"):
        FraudPreprocessor(exclude_columns=("amonut",)).fit(frame)


@pytest.mark.parametrize("column", sorted(PROTECTED))
def test_protected_columns_cannot_be_listed_as_exclusions(column: str) -> None:
    with pytest.raises(ValueError, match="always removed"):
        FraudPreprocessor(exclude_columns=(column,))


def test_exclusion_reason_recorded_in_manifest() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 1.0, 2.0]})
    manifest = infer_column_roles(frame, exclude=("b",), exclusion_reasons={"b": "proxy"})
    assert manifest.dropped["b"] == "excluded by configuration: proxy"
    assert manifest.numeric == ["a"]


@pytest.fixture
def signal_parquet(tmp_path: Path) -> Path:
    frame = make_frame(rows=4_000, seed=11)
    rng = np.random.default_rng(3)
    fraud = frame["is_fraud"].to_numpy() == 1
    # A fraud-only numeric tail and a bin that is a function of it.
    speed = np.where(fraud & (rng.random(len(frame)) < 0.5), rng.uniform(600, 900, len(frame)), 0.0)
    speed = np.where(speed == 0.0, rng.uniform(0, 400, len(frame)), speed)
    frame["speed_kmh"] = speed
    frame["speed_bin"] = np.where(speed > 500, "impossible", np.where(speed > 50, "fast", "normal"))
    return write_parquet(frame, tmp_path / "signals.parquet")


def test_signal_screen_finds_planted_signals(signal_parquet: Path, tmp_path: Path) -> None:
    report = screen_signals(DataSettings(dataset_path=signal_parquet), min_support=50)
    summary = report["summary"]

    assert "speed_bin=impossible" in summary["near_deterministic_values"]
    assert "speed_kmh" in summary["columns_with_fraud_only_tail"]
    assert "speed_bin <- speed_kmh" in summary["derived_bins"]
    assert "device_type" not in summary["columns_with_near_deterministic_values"]

    paths = write_signal_outputs(report, tmp_path / "out")
    written = paths["values"].read_text() + paths["report"].read_text()
    for value in ("t0000001", "c00001"):
        assert value not in written


def test_signal_screen_never_sees_target_or_identifiers(signal_parquet: Path) -> None:
    report = screen_signals(DataSettings(dataset_path=signal_parquet), min_support=50)
    screened = {row["column"] for row in report["value_screen"]}
    screened |= {row["column"] for row in report["numeric_support"]}
    assert PROTECTED.isdisjoint(screened)


def test_ablation_probe_writes_metrics_and_manifests(
    real_schema_parquet: Path, tmp_path: Path
) -> None:
    from sentinelflow.data.ablation_probe import main

    configs = [
        _config_for(path, real_schema_parquet, tmp_path / path.stem)
        for path in (FULL, *ABLATIONS.values())
    ]
    for path in configs:
        raw = _raw(path)
        raw["models"]["random_forest"]["n_estimators"] = 5
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    output = tmp_path / "probe"
    args = ["--configs", *map(str, configs), "--eval-rows", "300", "--output", str(output)]
    assert main(args) == 0

    with (output / "ablation_probe.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {row["config"] for row in rows} == {"full", "ablation_reduced", "ablation_strict"}
    assert {row["model"] for row in rows} == {"static_logistic", "static_random_forest"}
    excluded = {row["config"]: int(row["excluded"]) for row in rows}
    assert excluded == {"full": 0, "ablation_reduced": len(REDUCED), "ablation_strict": len(STRICT)}
    for name in ("full", "ablation_reduced", "ablation_strict"):
        manifest = json.loads((output / "manifests" / f"{name}.json").read_text())
        assert PROTECTED.isdisjoint(manifest["included"])
