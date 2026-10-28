"""Tests for the three controlled-drift scenarios: none, separability and camouflage."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import roc_auc_score

from sentinelflow.config import ConfigurationError, ProjectConfig, load_config
from sentinelflow.models.static import StaticModelSuite
from sentinelflow.streaming.diagnostics import (
    class_separation,
    separation_by_phase,
    standardized_mean_difference,
    total_variation_distance,
)
from sentinelflow.streaming.drift_injector import (
    SCENARIO_CAMOUFLAGE,
    SCENARIO_NONE,
    SCENARIO_SEPARABILITY,
    DriftInjector,
    DriftSettings,
    drift_intensity,
    drift_phase,
)
from sentinelflow.streaming.label_delay import StreamSettings, batch_stream
from sentinelflow.streaming.scenario_report import build_report, main, scenario_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_ROWS = 4000
BATCH_SIZE = 400
SIGNAL_COLUMNS = ["amount", "velocity"]
NUMERIC_COLUMNS = [*SIGNAL_COLUMNS, "noise"]


def make_stream_frame(seed: int = 3, n_rows: int = N_ROWS) -> tuple[pd.DataFrame, np.ndarray]:
    """Numeric + categorical features where fraud differs clearly from legitimate rows."""

    rng = np.random.default_rng(seed)
    labels = (rng.random(n_rows) < 0.25).astype(int)
    fraud = labels == 1
    features = pd.DataFrame(
        {
            "amount": rng.normal(10.0, 2.0, n_rows) + 5.0 * fraud,
            "velocity": rng.normal(0.0, 1.0, n_rows) + 3.0 * fraud,
            "noise": rng.normal(0.0, 1.0, n_rows),
            "count": rng.integers(0, 5, n_rows) + 3 * fraud,
            "device": np.where(
                fraud,
                rng.choice(["new", "known"], n_rows, p=[0.9, 0.1]),
                rng.choice(["new", "known"], n_rows, p=[0.1, 0.9]),
            ),
            "country": rng.choice(["us", "ca", "gb"], n_rows),
        }
    )
    return features, labels


def make_settings(scenario: str | None, **overrides) -> DriftSettings:
    values = {
        "enabled": scenario not in (None, SCENARIO_NONE) if scenario else True,
        "start_fraction": 0.40,
        "end_fraction": 0.70,
        "max_intensity": 0.90,
        "random_seed": 1543,
        "scenario": scenario,
    }
    values.update(overrides)
    return DriftSettings(**values)


def stream(scenario: str | None, **overrides):
    features, labels = make_stream_frame()
    injector = DriftInjector(make_settings(scenario, **overrides))
    stream_settings = StreamSettings(
        batch_size=BATCH_SIZE, label_delay_batches=2, seed=543, within_batch_shuffle=True
    )
    return (
        features,
        labels,
        injector,
        list(batch_stream(features, labels, stream_settings, injector)),
    )


def with_scenario(config_text: str, scenario: str) -> str:
    """Add ``drift.scenario`` to the shipped config text, keeping ``enabled: true`` beside it."""

    marker = "drift:\n  enabled: true"
    assert config_text.count(marker) == 1
    return config_text.replace(marker, f"{marker}\n  scenario: {scenario}")


def by_schedule(batches: list) -> list:
    """Relabel batches with the drift schedule's phases.

    The no-drift control keeps every batch in ``pre_drift``, so comparing it with the
    other scenarios phase by phase needs a common, schedule-based grouping.
    """

    schedule = make_settings(SCENARIO_SEPARABILITY)
    return [
        SimpleNamespace(
            phase=drift_phase(batch.progress, schedule),
            features=batch.features,
            labels=batch.labels,
        )
        for batch in batches
    ]


# ---------------------------------------------------------------------------
# Scenario selection and backward compatibility
# ---------------------------------------------------------------------------


def test_legacy_enabled_flag_maps_to_original_scenarios() -> None:
    assert make_settings(None, enabled=True).mode == SCENARIO_SEPARABILITY
    assert make_settings(None, enabled=False).mode == SCENARIO_NONE


def test_explicit_scenarios_are_selectable() -> None:
    for scenario in (SCENARIO_NONE, SCENARIO_SEPARABILITY, SCENARIO_CAMOUFLAGE):
        assert make_settings(scenario).mode == scenario


def test_explicit_scenario_wins_over_inherited_enabled_flag() -> None:
    base = {"start_fraction": 0.4, "end_fraction": 0.7, "max_intensity": 0.9, "random_seed": 1}

    assert DriftSettings.from_config({**base, "enabled": True, "scenario": "none"}).mode == "none"
    camouflage = DriftSettings.from_config({**base, "enabled": False, "scenario": "camouflage"})
    assert (camouflage.mode, camouflage.enabled) == ("camouflage", True)


@pytest.mark.parametrize(("scenario", "enabled"), [("none", True), ("camouflage", False)])
def test_direct_construction_rejects_contradictory_flags(scenario: str, enabled: bool) -> None:
    with pytest.raises(ValueError, match="contradicts"):
        make_settings(scenario, enabled=enabled)


def test_from_config_derives_enabled_from_scenario() -> None:
    base = {"start_fraction": 0.4, "end_fraction": 0.7, "max_intensity": 0.9, "random_seed": 1}

    assert DriftSettings.from_config({**base, "scenario": "camouflage"}).enabled is True
    assert DriftSettings.from_config({**base, "scenario": "none"}).enabled is False
    assert DriftSettings.from_config({**base, "enabled": True}).mode == SCENARIO_SEPARABILITY


@pytest.mark.parametrize(
    "drift",
    [
        {"scenario": "sideways"},
        {"scenario": "camouflage", "camouflage": {"strength": 1.5}},
        {"scenario": "camouflage", "camouflage": {"top_k": 0}},
        {"scenario": "camouflage", "camouflage": {"columns": []}},
        {},
    ],
)
def test_invalid_drift_configuration_is_rejected(drift: dict) -> None:
    base = {"start_fraction": 0.4, "end_fraction": 0.7, "max_intensity": 0.9, "random_seed": 1}

    with pytest.raises((ValueError, KeyError)):
        DriftSettings.from_config({**base, **drift})


def test_full_config_keeps_the_legacy_default_and_every_scenario_is_selectable(
    tmp_path: Path,
) -> None:
    source = PROJECT_ROOT / "configs" / "full.yaml"
    default = DriftSettings.from_config(load_config(source).section("drift"))
    assert default.mode == SCENARIO_SEPARABILITY

    text = source.read_text(encoding="utf-8")
    for scenario in (SCENARIO_NONE, SCENARIO_SEPARABILITY, SCENARIO_CAMOUFLAGE):
        config_dir = tmp_path / scenario / "configs"
        config_dir.mkdir(parents=True)
        path = config_dir / "full.yaml"
        path.write_text(with_scenario(text, scenario), encoding="utf-8")
        loaded = load_config(path)
        assert DriftSettings.from_config(loaded.section("drift")).mode == scenario


def test_config_rejects_unknown_scenario(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    path = config_dir / "full.yaml"
    text = (PROJECT_ROOT / "configs" / "full.yaml").read_text(encoding="utf-8")
    path.write_text(with_scenario(text, "sideways"), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="drift"):
        load_config(path)


# ---------------------------------------------------------------------------
# No-drift control and the original scenario stay unchanged
# ---------------------------------------------------------------------------


def test_no_drift_scenario_leaves_every_batch_unchanged() -> None:
    features, _, _, batches = stream(SCENARIO_NONE)

    assert {batch.phase for batch in batches} == {"pre_drift"}
    for batch in batches:
        original = features.iloc[batch.source_rows].reset_index(drop=True)
        pd.testing.assert_frame_equal(batch.features, original)


def test_original_scenario_is_identical_with_or_without_explicit_name() -> None:
    _, _, _, legacy = stream(None)
    _, _, _, explicit = stream(SCENARIO_SEPARABILITY)

    for left, right in zip(legacy, explicit, strict=True):
        assert left.phase == right.phase
        pd.testing.assert_frame_equal(left.features, right.features)


def test_original_scenario_ignores_camouflage_options() -> None:
    _, _, _, plain = stream(SCENARIO_SEPARABILITY)
    _, _, _, tuned = stream(SCENARIO_SEPARABILITY, camouflage_top_k=1, camouflage_strength=0.2)

    for left, right in zip(plain, tuned, strict=True):
        pd.testing.assert_frame_equal(left.features, right.features)


def test_original_scenario_does_not_resample_categories() -> None:
    features, _, _, batches = stream(SCENARIO_SEPARABILITY)

    for batch in batches:
        original = features.iloc[batch.source_rows].reset_index(drop=True)
        pd.testing.assert_series_equal(batch.features["device"], original["device"])


# ---------------------------------------------------------------------------
# Camouflage: labels, rows, reproducibility, targets
# ---------------------------------------------------------------------------


def test_camouflage_preserves_labels_rows_and_columns() -> None:
    features, labels, _, batches = stream(SCENARIO_CAMOUFLAGE)

    assert sum(batch.row_count for batch in batches) == N_ROWS
    for batch in batches:
        assert list(batch.features.columns) == list(features.columns)
        assert len(batch.features) == len(batch.labels)
        np.testing.assert_array_equal(batch.labels, labels[batch.source_rows])


def test_camouflage_never_uses_the_target_as_a_feature() -> None:
    features, labels = make_stream_frame()
    injector = DriftInjector(make_settings(SCENARIO_CAMOUFLAGE))

    result = injector.apply(features, labels, 0.9)

    assert "is_fraud" not in result.features.columns
    assert list(result.features.columns) == list(features.columns)
    assert set(injector.camouflage_columns or ()) <= set(features.columns)


def test_camouflage_leaves_legitimate_rows_untouched() -> None:
    features, labels = make_stream_frame()
    injector = DriftInjector(make_settings(SCENARIO_CAMOUFLAGE))

    result = injector.apply(features, labels, 1.0)

    legit = labels == 0
    pd.testing.assert_frame_equal(
        result.features[legit].reset_index(drop=True),
        features[legit].reset_index(drop=True),
        check_dtype=False,  # integer columns are promoted to float64 for the whole column
    )


def test_camouflage_only_changes_selected_columns_of_fraud_rows() -> None:
    features, labels = make_stream_frame()
    injector = DriftInjector(
        make_settings(SCENARIO_CAMOUFLAGE, camouflage_top_k=2, camouflage_categorical_top_k=1)
    )

    result = injector.apply(features, labels, 1.0)

    changed = {
        column
        for column in features.columns
        if not result.features[column].equals(
            features[column].astype(result.features[column].dtype)
        )
    }
    assert changed == set(result.affected_columns)
    assert changed == {"velocity", "amount", "device"}


def test_camouflage_is_reproducible_across_independent_runs() -> None:
    _, _, first_injector, first = stream(SCENARIO_CAMOUFLAGE)
    _, _, second_injector, second = stream(SCENARIO_CAMOUFLAGE)

    assert first_injector.camouflage_columns == second_injector.camouflage_columns
    assert (
        first_injector.camouflage_categorical_columns
        == second_injector.camouflage_categorical_columns
    )
    for left, right in zip(first, second, strict=True):
        assert left.phase == right.phase
        pd.testing.assert_frame_equal(left.features, right.features)


def test_camouflage_batch_does_not_depend_on_other_batches() -> None:
    features, labels = make_stream_frame()
    settings = make_settings(SCENARIO_CAMOUFLAGE)
    chunk, chunk_labels = features.iloc[:BATCH_SIZE], labels[:BATCH_SIZE]

    alone = DriftInjector(settings)
    warm = DriftInjector(settings)
    warm.apply(chunk, chunk_labels, 0.1)
    warm.apply(chunk, chunk_labels, 0.5)

    pd.testing.assert_frame_equal(
        alone.apply(chunk, chunk_labels, 0.9).features,
        warm.apply(chunk, chunk_labels, 0.9).features,
    )


def test_camouflage_random_seed_changes_only_the_category_draws() -> None:
    features, labels = make_stream_frame()
    first = DriftInjector(make_settings(SCENARIO_CAMOUFLAGE, random_seed=1)).apply(
        features, labels, 0.9
    )
    second = DriftInjector(make_settings(SCENARIO_CAMOUFLAGE, random_seed=2)).apply(
        features, labels, 0.9
    )

    for column in NUMERIC_COLUMNS:
        pd.testing.assert_series_equal(first.features[column], second.features[column])
    assert not first.features["device"].equals(second.features["device"])


def test_camouflage_selects_the_most_separating_columns_and_freezes_them() -> None:
    features, labels = make_stream_frame()
    injector = DriftInjector(
        make_settings(SCENARIO_CAMOUFLAGE, camouflage_top_k=2, camouflage_categorical_top_k=1)
    )

    injector.apply(features, labels, 0.1)
    chosen = injector.camouflage_columns
    injector.apply(features.assign(noise=features["noise"] + 100 * labels), labels, 0.9)

    assert set(chosen or ()) == {"velocity", "amount"}
    assert injector.camouflage_columns == chosen
    assert injector.camouflage_categorical_columns == ("device",)


def test_camouflage_honours_an_explicit_column_list() -> None:
    features, labels = make_stream_frame()
    injector = DriftInjector(
        make_settings(SCENARIO_CAMOUFLAGE, camouflage_columns=("amount", "device"))
    )

    result = injector.apply(features, labels, 1.0)

    assert injector.camouflage_columns == ("amount",)
    assert injector.camouflage_categorical_columns == ("device",)
    assert set(result.affected_columns) == {"amount", "device"}


def test_camouflage_rejects_unknown_explicit_columns() -> None:
    features, labels = make_stream_frame()
    injector = DriftInjector(make_settings(SCENARIO_CAMOUFLAGE, camouflage_columns=("missing",)))

    with pytest.raises(ValueError, match="missing"):
        injector.apply(features, labels, 0.9)


def test_camouflage_handles_missing_values_and_integer_columns() -> None:
    features, labels = make_stream_frame()
    features.loc[::7, "amount"] = np.nan
    features.loc[::11, "device"] = None
    injector = DriftInjector(make_settings(SCENARIO_CAMOUFLAGE))

    result = injector.apply(features, labels, 1.0)

    assert result.features["count"].dtype == np.float64
    assert result.features["amount"].isna().sum() == features["amount"].isna().sum()
    assert np.isfinite(result.features["velocity"]).all()


def test_camouflage_skips_batches_without_both_classes() -> None:
    features, labels = make_stream_frame()
    injector = DriftInjector(make_settings(SCENARIO_CAMOUFLAGE))

    result = injector.apply(features, np.zeros_like(labels), 0.9)

    assert result.affected_rows == 0
    pd.testing.assert_frame_equal(result.features, features)
    assert injector.camouflage_columns is None


# ---------------------------------------------------------------------------
# Phase boundaries and affected rows
# ---------------------------------------------------------------------------


def test_camouflage_phase_boundaries_and_intensity() -> None:
    settings = make_settings(SCENARIO_CAMOUFLAGE)

    assert drift_phase(0.39, settings) == "pre_drift"
    assert drift_phase(0.40, settings) == "ramp_drift"
    assert drift_phase(0.69, settings) == "ramp_drift"
    assert drift_phase(0.70, settings) == "drifted"
    assert drift_intensity(0.39, settings) == 0.0
    assert drift_intensity(0.55, settings) == pytest.approx(0.45)
    assert drift_intensity(0.70, settings) == pytest.approx(0.90)
    assert drift_intensity(1.00, settings) == pytest.approx(0.90)


def test_camouflage_affected_rows_follow_the_phases() -> None:
    features, labels = make_stream_frame()
    injector = DriftInjector(make_settings(SCENARIO_CAMOUFLAGE))
    fraud_rows = int(labels.sum())

    pre = injector.apply(features, labels, 0.2)
    ramp = injector.apply(features, labels, 0.55)
    drifted = injector.apply(features, labels, 0.95)

    assert (pre.phase, pre.affected_rows, pre.affected_columns) == ("pre_drift", 0, ())
    pd.testing.assert_frame_equal(pre.features, features)
    assert ramp.phase == "ramp_drift"
    assert ramp.affected_rows == fraud_rows
    assert drifted.phase == "drifted"
    assert drifted.affected_rows == fraud_rows
    assert ramp.intensity < drifted.intensity
    assert pre.scenario == ramp.scenario == drifted.scenario == SCENARIO_CAMOUFLAGE


def test_camouflage_stream_changes_only_after_the_start_fraction() -> None:
    features, _, _, batches = stream(SCENARIO_CAMOUFLAGE)
    settings = make_settings(SCENARIO_CAMOUFLAGE)

    for batch in batches:
        original = features.iloc[batch.source_rows].reset_index(drop=True)
        untouched = batch.features.equals(original)
        assert untouched == (drift_intensity(batch.progress, settings) == 0.0)


def test_disabled_scenarios_report_no_affected_rows() -> None:
    features, labels = make_stream_frame()
    result = DriftInjector(make_settings(SCENARIO_NONE)).apply(features, labels, 1.0)

    assert result.affected_rows == 0
    assert result.affected_columns == ()
    assert result.phase == "pre_drift"


# ---------------------------------------------------------------------------
# Class-separation diagnostics
# ---------------------------------------------------------------------------


def test_diagnostics_measure_known_separation() -> None:
    features, labels = make_stream_frame()

    effect = standardized_mean_difference(features, labels)
    assert effect["amount"] == pytest.approx(2.5, abs=0.3)
    assert abs(effect["noise"]) < 0.15
    assert total_variation_distance(features, labels)["device"] == pytest.approx(0.8, abs=0.05)
    assert total_variation_distance(features, labels)["country"] < 0.1
    assert class_separation(features, labels, ["amount"]) == pytest.approx(abs(effect["amount"]))


def test_diagnostics_return_nan_when_a_class_is_missing() -> None:
    features, labels = make_stream_frame()

    assert np.isnan(class_separation(features, np.zeros_like(labels)))
    assert all(
        np.isnan(v) for v in total_variation_distance(features, np.zeros_like(labels)).values()
    )


def test_camouflage_decreases_class_separation_phase_by_phase() -> None:
    _, _, _, control = stream(SCENARIO_NONE)
    _, _, _, camouflage = stream(SCENARIO_CAMOUFLAGE)

    control_by_phase = separation_by_phase(by_schedule(control), SIGNAL_COLUMNS)
    pooled = separation_by_phase(by_schedule(camouflage), SIGNAL_COLUMNS)

    pre, ramp, drifted = pooled["pre_drift"], pooled["ramp_drift"], pooled["drifted"]
    assert pre.separation == pytest.approx(control_by_phase["pre_drift"].separation)
    assert pre.separation > ramp.separation > drifted.separation
    assert drifted.separation < 0.5 * pre.separation
    assert drifted.categorical_separation < 0.5 * pre.categorical_separation


def test_full_strength_camouflage_closes_the_gap_at_full_intensity() -> None:
    features, labels = make_stream_frame()
    injector = DriftInjector(make_settings(SCENARIO_CAMOUFLAGE, max_intensity=1.0))

    result = injector.apply(features, labels, 1.0)

    assert class_separation(result.features, labels, SIGNAL_COLUMNS) < 0.02
    assert max(total_variation_distance(result.features, labels, ["device"]).values()) < 0.1


def test_camouflage_strength_scales_the_effect() -> None:
    features, labels = make_stream_frame()
    gaps = []
    for strength in (0.0, 0.5, 1.0):
        injector = DriftInjector(
            make_settings(SCENARIO_CAMOUFLAGE, max_intensity=1.0, camouflage_strength=strength)
        )
        result = injector.apply(features, labels, 1.0)
        gaps.append(class_separation(result.features, labels, SIGNAL_COLUMNS))

    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[0] == pytest.approx(class_separation(features, labels, SIGNAL_COLUMNS))


def test_separability_scenario_widens_the_gap_on_this_stream() -> None:
    _, _, _, control = stream(SCENARIO_NONE)
    _, _, _, shifted = stream(SCENARIO_SEPARABILITY)

    control_by_phase = separation_by_phase(by_schedule(control), SIGNAL_COLUMNS)
    drifted = separation_by_phase(by_schedule(shifted), SIGNAL_COLUMNS)

    assert drifted["drifted"].separation != pytest.approx(control_by_phase["drifted"].separation)


# ---------------------------------------------------------------------------
# Pressure on a frozen static baseline
# ---------------------------------------------------------------------------


def fit_frozen_baseline(seed: int = 11) -> StaticModelSuite:
    features, labels = make_stream_frame(seed=seed, n_rows=3000)
    matrix = pd.get_dummies(features, columns=["device", "country"]).to_numpy(dtype=np.float64)
    suite = StaticModelSuite(
        seed=543,
        positive_weight_cap=5.0,
        rf_enabled=False,
        rf_training_rows=1000,
        rf_n_estimators=1,
        rf_max_depth=None,
        rf_min_samples_leaf=1,
    )
    return suite.fit(matrix, labels)


def score_phase(
    suite: StaticModelSuite, scenario: str, phase: str
) -> tuple[np.ndarray, np.ndarray]:
    _, _, _, batches = stream(scenario)
    selected = [batch for batch in batches if batch.phase == phase]
    frames = pd.concat([batch.features for batch in selected], ignore_index=True)
    matrix = pd.get_dummies(frames, columns=["device", "country"]).reindex(
        columns=_baseline_columns(), fill_value=0
    )
    labels = np.concatenate([batch.labels for batch in selected])
    scores = suite.predict_probabilities(matrix.to_numpy(dtype=np.float64))["static_logistic"]
    return labels, scores


def _baseline_columns() -> list[str]:
    features, _ = make_stream_frame(seed=11, n_rows=3000)
    return list(pd.get_dummies(features, columns=["device", "country"]).columns)


def test_camouflage_degrades_the_frozen_static_baseline() -> None:
    suite = fit_frozen_baseline()

    control_labels, control_scores = score_phase(suite, SCENARIO_NONE, "pre_drift")
    drifted_labels, drifted_scores = score_phase(suite, SCENARIO_CAMOUFLAGE, "drifted")

    control_auc = roc_auc_score(control_labels, control_scores)
    drifted_auc = roc_auc_score(drifted_labels, drifted_scores)
    control_recall = (control_scores[control_labels == 1] >= 0.5).mean()
    drifted_recall = (drifted_scores[drifted_labels == 1] >= 0.5).mean()

    assert control_auc > 0.98
    assert drifted_auc < control_auc - 0.1
    assert drifted_recall < control_recall - 0.3


def test_no_drift_does_not_degrade_the_frozen_static_baseline() -> None:
    suite = fit_frozen_baseline()

    early_labels, early_scores = score_phase(suite, SCENARIO_NONE, "pre_drift")
    # The no-drift control never leaves pre_drift, so compare it to an equally sized later slice.
    _, _, _, batches = stream(SCENARIO_NONE)
    late = batches[-3:]
    frame = pd.concat([b.features for b in late], ignore_index=True)
    matrix = pd.get_dummies(frame, columns=["device", "country"]).reindex(
        columns=_baseline_columns(), fill_value=0
    )
    late_labels = np.concatenate([b.labels for b in late])
    late_scores = suite.predict_probabilities(matrix.to_numpy(dtype=np.float64))["static_logistic"]

    assert (
        abs(roc_auc_score(early_labels, early_scores) - roc_auc_score(late_labels, late_scores))
        < 0.02
    )


# ---------------------------------------------------------------------------
# Scenario report over the real loader pipeline (synthetic Parquet)
# ---------------------------------------------------------------------------


def make_report_config(synthetic_parquet: Path, tmp_path: Path) -> ProjectConfig:
    from test_experiment import make_config

    config = make_config(synthetic_parquet, tmp_path)
    raw = deepcopy(config.raw)
    raw["drift"]["start_fraction"] = 0.30
    raw["drift"]["end_fraction"] = 0.60
    raw["drift"]["max_intensity"] = 1.0
    raw["models"]["random_forest"]["enabled"] = False
    return ProjectConfig(raw=raw, root=config.root)


def test_scenario_settings_override_the_scenario_but_keep_the_schedule() -> None:
    base = make_settings(SCENARIO_SEPARABILITY)

    camouflage = scenario_settings(base, SCENARIO_CAMOUFLAGE)
    none = scenario_settings(base, SCENARIO_NONE)

    assert (camouflage.mode, camouflage.enabled) == (SCENARIO_CAMOUFLAGE, True)
    assert (none.mode, none.enabled) == (SCENARIO_NONE, False)
    assert camouflage.start_fraction == base.start_fraction
    with pytest.raises(ValueError, match="Unknown"):
        scenario_settings(base, "sideways")


def test_scenario_report_quantifies_pressure_on_the_frozen_baseline(
    synthetic_parquet: Path, tmp_path: Path
) -> None:
    config = make_report_config(synthetic_parquet, tmp_path)

    report = build_report(config, max_evaluation_rows=600)

    assert set(report["scenarios"]) == {SCENARIO_NONE, SCENARIO_SEPARABILITY, SCENARIO_CAMOUFLAGE}
    control = report["scenarios"][SCENARIO_NONE]
    camouflage = report["scenarios"][SCENARIO_CAMOUFLAGE]

    assert set(control["separation"]) == {"pre_drift", "ramp_drift", "drifted"}
    assert (
        camouflage["separation"]["pre_drift"]["mean_abs_d"]
        == control["separation"]["pre_drift"]["mean_abs_d"]
    )
    assert (
        camouflage["separation"]["drifted"]["mean_abs_d"]
        < control["separation"]["drifted"]["mean_abs_d"]
    )
    delta = camouflage["delta_vs_no_drift"]["models"]["static_logistic"]
    assert delta["pre_drift"]["roc_auc"] == 0.0
    assert delta["drifted"]["roc_auc"] < 0.0
    assert delta["drifted"]["false_positive_rate"] == pytest.approx(0.0, abs=0.05)
    assert "delta_vs_no_drift" not in control


def test_scenario_report_cli_writes_json(
    synthetic_parquet: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = make_report_config(synthetic_parquet, tmp_path).raw
    config_dir = tmp_path / "project" / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "report.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    output = tmp_path / "out" / "report.json"

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--max-evaluation-rows",
            "600",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert set(report["scenarios"]) == {SCENARIO_NONE, SCENARIO_SEPARABILITY, SCENARIO_CAMOUFLAGE}
    assert report["scenarios"][SCENARIO_CAMOUFLAGE]["batches"] == 6
    printed = capsys.readouterr().out
    assert "camouflage" in printed
    assert "Report written to" in printed


@pytest.mark.parametrize("scenario", [SCENARIO_NONE, SCENARIO_SEPARABILITY, SCENARIO_CAMOUFLAGE])
def test_end_to_end_experiment_runs_every_scenario(
    scenario: str, synthetic_parquet: Path, tmp_path: Path
) -> None:
    from sentinelflow.experiment import run_experiment

    config = make_report_config(synthetic_parquet, tmp_path)
    raw = deepcopy(config.raw)
    raw["drift"]["scenario"] = scenario
    config = ProjectConfig(raw=raw, root=config.root)

    result = run_experiment(config, max_evaluation_rows=600)

    assert result.summary.evaluation_rows == 600
    assert result.summary.batch_count == 6
    expected = (
        {"pre_drift"} if scenario == SCENARIO_NONE else {"pre_drift", "ramp_drift", "drifted"}
    )
    assert set(result.summary.phase_counts) == expected
