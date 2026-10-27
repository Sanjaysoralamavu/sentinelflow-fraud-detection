"""Tests for the multi-seed detector study (issue #10)."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from sentinelflow.config import ProjectConfig, load_config
from sentinelflow.experiment import run_experiment
from sentinelflow.models.common import (
    ALARM_CONFIRMED,
    ALARM_SUPPRESSED_COOLDOWN,
    ALARM_SUPPRESSED_GRACE,
    DriftEvent,
)
from sentinelflow.models.detector_study import (
    check_recovery,
    drift_start_batch,
    parse_scenario,
    run_detector_study,
    scenario_config,
)
from sentinelflow.streaming.contracts import PHASE_PRE_DRIFT

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WINDOW_ROWS = 300
DELAY = 2


def make_config(synthetic_parquet: Path, tmp_path: Path) -> ProjectConfig:
    """Fast synthetic configuration with no grace period so alarms can confirm."""

    base = load_config(PROJECT_ROOT / "configs" / "full.yaml")
    raw = deepcopy(base.raw)
    raw["paths"]["dataset"] = str(synthetic_parquet)
    raw["paths"]["results"] = str(tmp_path / "results")
    raw["data"].update({"batch_size": 50, "warmup_rows": 300})
    raw["stream"].update({"label_delay_batches": DELAY})
    raw["drift"].update(
        {"enabled": True, "start_fraction": 0.3, "end_fraction": 0.6, "max_intensity": 1.0}
    )
    raw["models"]["random_forest"].update(
        {"training_rows": 250, "n_estimators": 5, "max_depth": 4, "min_samples_leaf": 2}
    )
    raw["models"]["adaptive"].update(
        {
            "detector_sample_size": 50,
            "recent_window_rows": WINDOW_ROWS,
            "ddm_warm_start": 10,
            "detector_grace_batches": 0,
            "detector_cooldown_batches": 2,
        }
    )
    raw["models"]["online_ensemble"].update({"n_estimators": 2})
    return ProjectConfig(raw=raw, root=base.root)


def event(state: str, *, source: int, reset_count: int, **recovery) -> DriftEvent:
    return DriftEvent(
        model="adaptive_ddm",
        detector="ddm",
        source_batch=source,
        observed_batch=source + DELAY,
        phase="drifted",
        alarm_type="drift",
        alarm_state=state,
        reset_count=reset_count,
        **recovery,
    )


def confirmed(source: int, reset_count: int = 1, **overrides) -> DriftEvent:
    recovery = {
        "retrain_rows": 250,
        "retrain_first_batch": source - 2,
        "retrain_last_batch": source,
        "fraud_weight": 5.0,
    }
    recovery.update(overrides)
    return event(ALARM_CONFIRMED, source=source, reset_count=reset_count, **recovery)


# drift_start_batch -------------------------------------------------------------


def test_drift_start_batch_is_first_batch_past_start_fraction() -> None:
    drift = {"enabled": True, "start_fraction": 0.45, "end_fraction": 0.70}
    drift.update({"max_intensity": 0.9, "random_seed": 1})

    # Ten batches of 100: batch 4 ends at progress 0.5, the first value >= 0.45.
    assert drift_start_batch(drift, 1_000, 100) == 4
    # 1.5M rows in 50k batches (the 30-batch experiment): batch 13 ends at 0.4667.
    assert drift_start_batch(drift, 1_500_000, 50_000) == 13


def test_drift_start_batch_is_none_without_drift() -> None:
    drift = {"enabled": False, "start_fraction": 0.45, "end_fraction": 0.70}
    drift.update({"max_intensity": 0.9, "random_seed": 1})

    assert drift_start_batch(drift, 1_000, 100) is None


def test_drift_start_batch_matches_the_stream_phases(
    synthetic_parquet: Path, tmp_path: Path
) -> None:
    config = make_config(synthetic_parquet, tmp_path)
    phases: list[str] = []

    class PhaseRecorder:
        def record(self, batch, predictions) -> None:
            phases.append(batch.phase)

    run_experiment(config, recorder=PhaseRecorder(), max_evaluation_rows=600)

    first_drift = next(i for i, phase in enumerate(phases) if phase != PHASE_PRE_DRIFT)
    assert drift_start_batch(config.section("drift"), 600, config.batch_size) == first_drift


# scenario configuration --------------------------------------------------------


def test_scenario_config_overrides_seed_and_drift_without_mutating_base(
    synthetic_parquet: Path, tmp_path: Path
) -> None:
    base = make_config(synthetic_parquet, tmp_path)
    before = deepcopy(base.raw)

    scenario = scenario_config(base, seed=99, drift_overrides={"enabled": False})

    assert scenario.seed == 99
    assert scenario.section("drift")["enabled"] is False
    assert scenario.section("drift")["random_seed"] == base.section("drift")["random_seed"]
    assert base.raw == before


def test_parse_scenario_reads_yaml_values() -> None:
    assert parse_scenario("no_drift:enabled=false") == ("no_drift", {"enabled": False})
    assert parse_scenario("degrading:enabled=true, mode=degrading, max_intensity=0.5") == (
        "degrading",
        {"enabled": True, "mode": "degrading", "max_intensity": 0.5},
    )


@pytest.mark.parametrize("text", ["no_drift", ":enabled=false", "x:", "x:enabled"])
def test_parse_scenario_rejects_malformed_text(text: str) -> None:
    with pytest.raises(ValueError):
        parse_scenario(text)


# recovery contract --------------------------------------------------------------


def test_valid_recovery_has_no_violations() -> None:
    events = [
        event(ALARM_SUPPRESSED_GRACE, source=1, reset_count=0),
        confirmed(5, reset_count=1),
        event(ALARM_SUPPRESSED_COOLDOWN, source=6, reset_count=1),
        confirmed(9, reset_count=2),
    ]

    assert check_recovery(events, recent_window_rows=WINDOW_ROWS, label_delay_batches=DELAY) == []


@pytest.mark.parametrize(
    ("bad_event", "message"),
    [
        (confirmed(5, retrain_rows=WINDOW_ROWS + 1), "expected 1 to"),
        (confirmed(5, retrain_rows=0), "expected 1 to"),
        (confirmed(5, retrain_last_batch=6), "not yet released"),
        (confirmed(5, retrain_first_batch=None), "not recorded"),
        (confirmed(5, fraud_weight=None), "fraud weight"),
        (confirmed(5, reset_count=3), "reset count 3"),
        (event(ALARM_SUPPRESSED_GRACE, source=5, reset_count=0, retrain_rows=10), "retrained"),
        (event(ALARM_SUPPRESSED_COOLDOWN, source=5, reset_count=1), "changed the reset count"),
    ],
)
def test_recovery_violations_are_reported(bad_event: DriftEvent, message: str) -> None:
    violations = check_recovery(
        [bad_event], recent_window_rows=WINDOW_ROWS, label_delay_batches=DELAY
    )

    assert any(message in violation for violation in violations)


# end to end ---------------------------------------------------------------------


def test_study_runs_every_scenario_and_seed_and_writes_artifacts(
    synthetic_parquet: Path, tmp_path: Path
) -> None:
    config = make_config(synthetic_parquet, tmp_path)
    output = tmp_path / "study"

    study = run_detector_study(
        config, seeds=[1, 2], max_evaluation_rows=600, output_directory=output
    )

    assert [(run.scenario, run.seed) for run in study.runs] == [
        ("no_drift", 1),
        ("no_drift", 2),
        ("controlled_drift", 1),
        ("controlled_drift", 2),
    ]

    for run in study.runs:
        assert run.last_scored_batch == 11
        assert run.recovery_violations == ()
        assert set(run.alarm_counts()) == {"adaptive_adwin", "adaptive_ddm"}
        assert all(e.model in {"adaptive_adwin", "adaptive_ddm"} for e in run.events)
        if run.scenario == "no_drift":
            assert run.drift_start_batch is None
            assert set(run.summary.phase_counts) == {PHASE_PRE_DRIFT}
        else:
            assert run.drift_start_batch == 3

    saved = json.loads((output / "detector_study.json").read_text(encoding="utf-8"))
    assert saved["parameters"]["seeds"] == [1, 2]
    assert saved["parameters"]["label_delay_batches"] == DELAY
    assert saved["parameters"]["adaptive"]["recent_window_rows"] == WINDOW_ROWS
    assert len(saved["runs"]) == 4

    for scenario in ("no_drift", "controlled_drift"):
        for seed in (1, 2):
            run_directory = output / scenario / f"seed_{seed}"
            for name in ("run_manifest", "feature_manifest", "drift_events", "study_run"):
                assert (run_directory / f"{name}.json").is_file()


class PeriodicDetector:
    """Signals drift on every ``period``-th error it receives."""

    def __init__(self, period: int) -> None:
        self.period = period
        self.updates = 0
        self.drift_detected = False

    def update(self, _error: int) -> None:
        self.updates += 1
        self.drift_detected = self.updates % self.period == 0


def test_study_recovery_holds_on_real_pipeline_alarms(
    synthetic_parquet: Path, tmp_path: Path, monkeypatch
) -> None:
    """Force alarms through the full pipeline and check every recovery it logs."""

    import sentinelflow.models.adaptive as adaptive_module

    monkeypatch.setattr(adaptive_module, "build_detector", lambda *_: PeriodicDetector(120))
    config = make_config(synthetic_parquet, tmp_path)
    config.raw["models"]["adaptive"]["detector_cooldown_batches"] = 4

    study = run_detector_study(
        config,
        seeds=[1],
        scenarios={"controlled_drift": {"enabled": True}},
        max_evaluation_rows=600,
    )
    run = study.runs[0]
    states = {e.alarm_state for e in run.events}

    assert ALARM_CONFIRMED in states
    assert ALARM_SUPPRESSED_COOLDOWN in states
    assert run.recovery_violations == ()
    for alarm in run.events:
        if alarm.confirmed:
            assert 0 < alarm.retrain_rows <= WINDOW_ROWS
            assert alarm.retrain_last_batch <= alarm.observed_batch - DELAY


def test_study_is_reproducible(synthetic_parquet: Path, tmp_path: Path) -> None:
    config = make_config(synthetic_parquet, tmp_path)
    scenarios = {"controlled_drift": {"enabled": True}}

    first = run_detector_study(config, seeds=[3], scenarios=scenarios, max_evaluation_rows=500)
    second = run_detector_study(config, seeds=[3], scenarios=scenarios, max_evaluation_rows=500)

    assert first.runs[0].events == second.runs[0].events
    assert first.runs[0].drift_start_batch == second.runs[0].drift_start_batch


def test_study_rejects_missing_or_duplicate_seeds(synthetic_parquet: Path, tmp_path: Path) -> None:
    config = make_config(synthetic_parquet, tmp_path)

    with pytest.raises(ValueError, match="At least one seed"):
        run_detector_study(config, seeds=[])
    with pytest.raises(ValueError, match="unique"):
        run_detector_study(config, seeds=[1, 1])
