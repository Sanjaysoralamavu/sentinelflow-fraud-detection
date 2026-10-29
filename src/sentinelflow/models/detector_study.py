"""Multi-seed, multi-scenario study of the ADWIN and DDM detectors (issue #10).

Each study run is one complete ``run_experiment`` call for a (scenario, seed)
pair. A scenario is a set of overrides for the ``drift`` configuration section,
for example ``{"enabled": False}`` for the no-drift control. Only
``project.random_seed`` changes between seeds; ``drift.random_seed`` is kept
from the base configuration, so every seed sees the same drift directions and
the seeds measure model, shuffling and detector-sampling variability.

For every run the study keeps:

- every logged ``DriftEvent`` (confirmed and suppressed);
- ``drift_start_batch``: the first source batch outside ``pre_drift``, or
  ``None`` for a no-drift scenario. This is the reference point for false
  alarms, missed detections and detection delays;
- ``last_scored_batch``: the last batch scored in the stream. Labels still
  queued when the stream ends are learned afterwards, so an alarm with
  ``observed_batch > last_scored_batch`` can no longer change any prediction;
- recovery violations found by :func:`check_recovery`;
- detector quality from ``evaluation.detectors`` (false alarms, missed
  detection, source- and observed-batch delays), scored against
  ``drift_start_batch``;
- overall and per-phase metrics for every model from the evaluation
  recorder, so adaptive models can be compared with the frozen baselines.

Default scenarios use ``drift.scenario`` together with a matching
``drift.enabled``, so they mean the same thing whether or not the installed
injector supports named scenarios. ``camouflage`` is added automatically
when the injector provides it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from sentinelflow.config import ProjectConfig
from sentinelflow.evaluation.detectors import DetectorQuality, evaluate_all_detectors
from sentinelflow.evaluation.recorder import EvaluationRecorder
from sentinelflow.experiment import ExperimentSummary, run_experiment, write_experiment_artifacts
from sentinelflow.models.adaptive import DETECTOR_NAMES
from sentinelflow.models.common import (
    ALARM_CONFIRMED,
    ALARM_SUPPRESSED_COOLDOWN,
    ALARM_SUPPRESSED_GRACE,
    DriftEvent,
)
from sentinelflow.streaming import drift_injector
from sentinelflow.streaming.contracts import PHASE_PRE_DRIFT
from sentinelflow.streaming.drift_injector import DriftSettings, drift_phase

DEFAULT_SCENARIOS: dict[str, dict[str, Any]] = {
    "no_drift": {"enabled": False, "scenario": "none"},
    "separability": {"enabled": True, "scenario": "separability"},
}
if "camouflage" in getattr(drift_injector, "SCENARIOS", ()):
    DEFAULT_SCENARIOS["camouflage"] = {"enabled": True, "scenario": "camouflage"}

DETECTOR_MODELS = (("adaptive_adwin", "adwin"), ("adaptive_ddm", "ddm"))

ALARM_STATES = (ALARM_CONFIRMED, ALARM_SUPPRESSED_GRACE, ALARM_SUPPRESSED_COOLDOWN)


@dataclass(frozen=True)
class StudyRun:
    """Outcome of one (scenario, seed) experiment."""

    scenario: str
    seed: int
    drift_overrides: dict[str, Any]
    drift_start_batch: int | None
    last_scored_batch: int
    summary: ExperimentSummary
    events: tuple[DriftEvent, ...]
    recovery_violations: tuple[str, ...]
    detector_quality: tuple[DetectorQuality, ...] = ()
    overall_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    phase_metrics: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    def alarm_counts(self) -> dict[str, dict[str, int]]:
        """Count alarms per detector model and state.

        ``confirmed_after_stream_end`` counts confirmed alarms whose labels
        were released only after the last scored batch.
        """

        counts: dict[str, dict[str, int]] = {}
        for detector in DETECTOR_NAMES:
            if detector == "none":
                continue
            model = f"adaptive_{detector}"
            model_events = [event for event in self.events if event.model == model]
            row = {
                state: sum(e.alarm_state == state for e in model_events) for state in ALARM_STATES
            }
            row["confirmed_after_stream_end"] = sum(
                e.confirmed and e.observed_batch > self.last_scored_batch for e in model_events
            )
            counts[model] = row
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "seed": self.seed,
            "drift_overrides": self.drift_overrides,
            "drift_start_batch": self.drift_start_batch,
            "last_scored_batch": self.last_scored_batch,
            "summary": self.summary.to_dict(),
            "alarm_counts": self.alarm_counts(),
            "recovery_violations": list(self.recovery_violations),
            "detector_quality": [asdict(quality) for quality in self.detector_quality],
            "overall_metrics": self.overall_metrics,
            "phase_metrics": self.phase_metrics,
            "events": [asdict(event) for event in self.events],
        }


@dataclass(frozen=True)
class DetectorStudy:
    """All runs of one study plus the settings they used."""

    parameters: dict[str, Any]
    runs: tuple[StudyRun, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameters": self.parameters,
            "runs": [run.to_dict() for run in self.runs],
        }


def scenario_config(
    config: ProjectConfig,
    *,
    seed: int,
    drift_overrides: Mapping[str, Any],
) -> ProjectConfig:
    """Return a copy of ``config`` with a new project seed and drift overrides."""

    raw = deepcopy(config.raw)
    raw["project"]["random_seed"] = int(seed)
    raw["drift"].update(deepcopy(dict(drift_overrides)))
    return ProjectConfig(raw=raw, root=config.root)


def drift_start_batch(
    drift: Mapping[str, Any], evaluation_rows: int, batch_size: int
) -> int | None:
    """Return the first source batch outside ``pre_drift``, or ``None`` if there is none.

    Uses the same progress definition as the stream (rows emitted by the end
    of the batch divided by evaluation rows) and the injector's own phase
    function, so it agrees with ``Batch.phase``.
    """

    if evaluation_rows <= 0 or batch_size <= 0:
        raise ValueError("evaluation_rows and batch_size must be greater than zero.")

    settings = DriftSettings.from_config(dict(drift))
    for batch_id in range(math.ceil(evaluation_rows / batch_size)):
        progress = min((batch_id + 1) * batch_size, evaluation_rows) / evaluation_rows
        if drift_phase(progress, settings) != PHASE_PRE_DRIFT:
            return batch_id
    return None


def check_recovery(
    events: Iterable[DriftEvent],
    *,
    recent_window_rows: int,
    label_delay_batches: int,
) -> list[str]:
    """Return every way the logged alarms break the recovery contract.

    A confirmed alarm must retrain a fresh learner on between 1 and
    ``recent_window_rows`` rows, using only batches whose labels were already
    released, and must increase that model's reset count by exactly one.
    A suppressed alarm must not retrain.
    """

    violations: list[str] = []
    resets: dict[str, int] = {}

    for event in sorted(events, key=lambda e: (e.model, e.observed_batch, e.source_batch)):
        where = f"{event.model} source {event.source_batch} observed {event.observed_batch}"
        previous = resets.get(event.model, 0)

        if event.alarm_state == ALARM_CONFIRMED:
            if not 0 < event.retrain_rows <= recent_window_rows:
                violations.append(
                    f"{where}: retrained on {event.retrain_rows} rows, "
                    f"expected 1 to {recent_window_rows}."
                )
            if event.retrain_first_batch is None or event.retrain_last_batch is None:
                violations.append(f"{where}: retraining window is not recorded.")
            elif event.retrain_last_batch > event.observed_batch - label_delay_batches:
                violations.append(
                    f"{where}: retrained on batch {event.retrain_last_batch}, "
                    "whose labels were not yet released."
                )
            if event.fraud_weight is None or event.fraud_weight < 1.0:
                violations.append(f"{where}: invalid retraining fraud weight.")
            if event.reset_count != previous + 1:
                violations.append(
                    f"{where}: reset count {event.reset_count}, expected {previous + 1}."
                )
            resets[event.model] = event.reset_count
        else:
            if event.retrain_rows != 0 or event.retrain_first_batch is not None:
                violations.append(f"{where}: {event.alarm_state} alarm retrained the model.")
            if event.reset_count != previous:
                violations.append(f"{where}: suppressed alarm changed the reset count.")

    return violations


def run_detector_study(
    config: ProjectConfig,
    *,
    seeds: Sequence[int],
    scenarios: Mapping[str, Mapping[str, Any]] | None = None,
    max_evaluation_rows: int | None = None,
    output_directory: str | Path | None = None,
    evaluation_outputs: bool = False,
) -> DetectorStudy:
    """Run every scenario with every seed and optionally write the artifacts.

    Artifacts, when ``output_directory`` is given::

        detector_study.json                 parameters and every run
        detector_summary.csv                one row per run and detector
        model_phase_summary.csv             one row per run, model and phase
        <scenario>/seed_<seed>/             run_manifest.json,
                                            feature_manifest.json,
                                            drift_events.json, study_run.json
        <scenario>/seed_<seed>/evaluation/  CSVs, figures and dashboards,
                                            only with ``evaluation_outputs``
    """

    if not seeds:
        raise ValueError("At least one seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique.")

    scenarios = DEFAULT_SCENARIOS if scenarios is None else scenarios
    if not scenarios:
        raise ValueError("At least one scenario is required.")

    adaptive = config.section("models")["adaptive"]
    output = None if output_directory is None else Path(output_directory)
    runs: list[StudyRun] = []

    for scenario, overrides in scenarios.items():
        for seed in seeds:
            run_config = scenario_config(config, seed=seed, drift_overrides=overrides)
            recorder = EvaluationRecorder.from_config(run_config)
            result = run_experiment(
                run_config, recorder=recorder, max_evaluation_rows=max_evaluation_rows
            )
            summary = result.summary
            start = drift_start_batch(
                run_config.section("drift"),
                summary.evaluation_rows,
                run_config.batch_size,
            )

            run = StudyRun(
                scenario=scenario,
                seed=int(seed),
                drift_overrides=dict(overrides),
                drift_start_batch=start,
                last_scored_batch=summary.batch_count - 1,
                summary=summary,
                events=result.drift_events,
                recovery_violations=tuple(
                    check_recovery(
                        result.drift_events,
                        recent_window_rows=int(adaptive["recent_window_rows"]),
                        label_delay_batches=run_config.label_delay_batches,
                    )
                ),
                detector_quality=tuple(
                    evaluate_all_detectors(
                        result.drift_events,
                        drift_start_batch=start,
                        expected_detectors=DETECTOR_MODELS,
                    )
                ),
                overall_metrics={
                    model: _clean(asdict(metrics))
                    for model, metrics in recorder.overall_metrics().items()
                },
                phase_metrics=_phase_metrics(recorder),
            )
            runs.append(run)

            if output is not None:
                run_directory = output / scenario / f"seed_{seed}"
                write_experiment_artifacts(result, run_config, run_directory)
                _write_json(run_directory / "study_run.json", run.to_dict())
                if evaluation_outputs:
                    from sentinelflow.evaluation.pipeline import generate_evaluation_outputs

                    generate_evaluation_outputs(recorder, result, run_directory / "evaluation")

    study = DetectorStudy(
        parameters=study_parameters(
            config,
            seeds=seeds,
            scenarios=scenarios,
            max_evaluation_rows=max_evaluation_rows,
        ),
        runs=tuple(runs),
    )

    if output is not None:
        _write_json(output / "detector_study.json", study.to_dict())
        detector_summary_frame(study).to_csv(output / "detector_summary.csv", index=False)
        model_phase_frame(study).to_csv(output / "model_phase_summary.csv", index=False)

    return study


def detector_summary_frame(study: DetectorStudy) -> pd.DataFrame:
    """One row per (scenario, seed, detector) with quality and alarm counts."""

    rows = []
    for run in study.runs:
        counts = run.alarm_counts()
        for quality in run.detector_quality:
            row = {
                "scenario": run.scenario,
                "seed": run.seed,
                "drift_start_batch": run.drift_start_batch,
                "last_scored_batch": run.last_scored_batch,
                **asdict(quality),
            }
            row.update(counts.get(quality.model, {}))
            row["recovery_violations"] = len(run.recovery_violations)
            rows.append(row)
    return pd.DataFrame(rows)


def model_phase_frame(study: DetectorStudy) -> pd.DataFrame:
    """One row per (scenario, seed, model, phase), plus an ``overall`` phase."""

    rows = []
    for run in study.runs:
        for model, metrics in run.overall_metrics.items():
            rows.append(
                {"scenario": run.scenario, "seed": run.seed, "model": model, "phase": "overall"}
                | metrics
            )
        for model, phases in run.phase_metrics.items():
            for phase, metrics in phases.items():
                rows.append(
                    {"scenario": run.scenario, "seed": run.seed, "model": model, "phase": phase}
                    | metrics
                )
    return pd.DataFrame(rows)


def _phase_metrics(recorder: EvaluationRecorder) -> dict[str, dict[str, dict[str, Any]]]:
    phases: dict[str, dict[str, dict[str, Any]]] = {}
    for (model, phase), metrics in recorder.phase_metrics().items():
        phases.setdefault(model, {})[phase] = _clean(asdict(metrics))
    return phases


def _clean(values: dict[str, Any]) -> dict[str, Any]:
    """Replace NaN (for example AUC on a one-class phase) with ``None`` for JSON."""

    return {
        key: None if isinstance(value, float) and math.isnan(value) else value
        for key, value in values.items()
    }


def study_parameters(
    config: ProjectConfig,
    *,
    seeds: Sequence[int],
    scenarios: Mapping[str, Mapping[str, Any]],
    max_evaluation_rows: int | None,
) -> dict[str, Any]:
    """Every setting that affects detector behavior, reported with the results."""

    models = config.section("models")
    return {
        "seeds": [int(seed) for seed in seeds],
        "scenarios": {name: dict(overrides) for name, overrides in scenarios.items()},
        "max_evaluation_rows": max_evaluation_rows,
        "batch_size": config.batch_size,
        "warmup_rows": config.warmup_rows,
        "label_delay_batches": config.label_delay_batches,
        "threshold": models["threshold"],
        "positive_weight_cap": models["positive_weight_cap"],
        "adaptive": deepcopy(models["adaptive"]),
        "base_drift": deepcopy(config.section("drift")),
    }


def parse_scenario(text: str) -> tuple[str, dict[str, Any]]:
    """Parse ``name:key=value,key=value`` into a scenario name and drift overrides.

    Values are read as YAML scalars, so ``enabled=false`` gives ``False`` and
    ``max_intensity=0.5`` gives ``0.5``.
    """

    name, separator, assignments = text.partition(":")
    name = name.strip()
    if not name or not separator or not assignments.strip():
        raise ValueError(f"Scenario must look like 'name:key=value[,key=value]', got '{text}'.")

    overrides: dict[str, Any] = {}
    for assignment in assignments.split(","):
        key, equals, value = assignment.partition("=")
        if not key.strip() or not equals:
            raise ValueError(f"Invalid drift override '{assignment}' in scenario '{name}'.")
        overrides[key.strip()] = yaml.safe_load(value.strip())
    return name, overrides


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
