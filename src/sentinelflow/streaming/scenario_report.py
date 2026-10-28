"""Quantitative comparison of the three drift scenarios (no-drift, separability, camouflage).

For each scenario the same warm-up window fits one preprocessor and one set of
frozen static baselines. The identical evaluation stream is then replayed
through the scenario's injector and scored batch by batch, without any model
update. Per drift phase the report records:

- model-free class separation (mean absolute standardized mean difference over
  the numeric features, see ``streaming.diagnostics``), and
- frozen-baseline ROC-AUC, recall and false-positive rate at the configured
  threshold, each with its change relative to the same phase of the no-drift
  control.

Run it with::

    python -m sentinelflow.streaming.scenario_report --config configs/full.yaml \\
        --max-evaluation-rows 600000

``--max-evaluation-rows`` shrinks the stream but keeps the drift schedule,
because phases are defined by progress through the evaluated rows.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from sentinelflow.config import ProjectConfig, load_config
from sentinelflow.data.loader import DataSettings, load_warmup_and_stream
from sentinelflow.data.preprocessing import FraudPreprocessor
from sentinelflow.models.static import StaticModelSuite
from sentinelflow.streaming.contracts import PHASE_DRIFTED, PHASE_PRE_DRIFT, PHASE_RAMP_DRIFT
from sentinelflow.streaming.diagnostics import separation_by_phase
from sentinelflow.streaming.drift_injector import (
    SCENARIO_NONE,
    SCENARIO_SEPARABILITY,
    SCENARIOS,
    DriftInjector,
    DriftSettings,
    drift_phase,
)
from sentinelflow.streaming.label_delay import StreamSettings, batch_chunks

DEFAULT_MAX_EVALUATION_ROWS = 600_000
PHASES = (PHASE_PRE_DRIFT, PHASE_RAMP_DRIFT, PHASE_DRIFTED)


def scenario_settings(base: DriftSettings, scenario: str) -> DriftSettings:
    """Return ``base`` with its schedule and camouflage options but a different scenario."""

    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown drift scenario '{scenario}'. Expected one of {SCENARIOS}.")
    return replace(base, scenario=scenario, enabled=scenario != SCENARIO_NONE)


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def _threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold
    fraud = labels == 1
    fraud_rows = int(fraud.sum())
    legit_rows = int((~fraud).sum())
    return {
        "recall": float((predicted & fraud).sum() / fraud_rows) if fraud_rows else None,
        "false_positive_rate": float((predicted & ~fraud).sum() / legit_rows)
        if legit_rows
        else None,
    }


def evaluate_scenario(
    config: ProjectConfig,
    scenario: str,
    *,
    max_evaluation_rows: int,
    preprocessor: FraudPreprocessor,
    static_models: StaticModelSuite,
) -> dict[str, Any]:
    """Replay the evaluation stream under one scenario and score the frozen baselines."""

    data_settings = DataSettings.from_config(config)
    stream_settings = StreamSettings.from_config(
        data=config.section("data"), stream=config.section("stream"), seed=config.seed
    )
    drift_settings = scenario_settings(DriftSettings.from_config(config.section("drift")), scenario)
    injector = DriftInjector(drift_settings)
    # Group every scenario by the same schedule-based phases. The no-drift control labels all
    # of its batches "pre_drift", which would otherwise compare different rows across scenarios.
    schedule = scenario_settings(drift_settings, SCENARIO_SEPARABILITY)
    threshold = float(config.section("models")["threshold"])

    _, chunks = load_warmup_and_stream(
        data_settings, max_rows=config.warmup_rows + max_evaluation_rows
    )
    batches = batch_chunks(chunks, max_evaluation_rows, stream_settings, injector)

    scores: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    labels: dict[str, list[np.ndarray]] = defaultdict(list)
    kept: list = []
    batch_count = 0

    for batch in batches:
        phase = drift_phase(batch.progress, schedule)
        kept.append(SimpleNamespace(phase=phase, features=batch.features, labels=batch.labels))
        labels[phase].append(np.asarray(batch.labels))
        probabilities = static_models.predict_probabilities(preprocessor.transform(batch.features))
        for name, values in probabilities.items():
            scores[name][phase].append(np.asarray(values))
        batch_count += 1

    separation = separation_by_phase(kept)
    models: dict[str, dict[str, Any]] = {}
    for name, per_phase in scores.items():
        models[name] = {}
        for phase in PHASES:
            if phase not in labels:
                continue
            y = np.concatenate(labels[phase])
            s = np.concatenate(per_phase[phase])
            models[name][phase] = {
                "rows": int(len(y)),
                "roc_auc": _safe_auc(y, s),
                **_threshold_metrics(y, s, threshold),
            }

    return {
        "scenario": scenario,
        "batches": batch_count,
        "camouflage_columns": list(injector.camouflage_columns or ()),
        "camouflage_categorical_columns": list(injector.camouflage_categorical_columns),
        "separation": {
            phase: {
                "rows": value.rows,
                "fraud_rows": value.fraud_rows,
                "mean_abs_d": None if math.isnan(value.separation) else value.separation,
                "mean_tvd": None
                if math.isnan(value.categorical_separation)
                else value.categorical_separation,
            }
            for phase, value in separation.items()
        },
        "models": models,
    }


def add_deltas(report: dict[str, Any]) -> None:
    """Attach each scenario's change versus the no-drift control, phase by phase."""

    control = report["scenarios"].get(SCENARIO_NONE)
    if control is None:
        return

    for scenario, result in report["scenarios"].items():
        if scenario == SCENARIO_NONE:
            continue
        deltas: dict[str, Any] = {"separation": {}, "models": {}}
        for phase, value in result["separation"].items():
            base = control["separation"].get(phase, {}).get("mean_abs_d")
            current = value["mean_abs_d"]
            if base is not None and current is not None:
                deltas["separation"][phase] = current - base
            base_tvd = control["separation"].get(phase, {}).get("mean_tvd")
            if base_tvd is not None and value["mean_tvd"] is not None:
                deltas["separation"][f"{phase}_tvd"] = value["mean_tvd"] - base_tvd
        for name, per_phase in result["models"].items():
            deltas["models"][name] = {}
            for phase, metrics in per_phase.items():
                base = control["models"].get(name, {}).get(phase, {})
                deltas["models"][name][phase] = {
                    key: metrics[key] - base[key]
                    for key in ("roc_auc", "recall", "false_positive_rate")
                    if metrics.get(key) is not None and base.get(key) is not None
                }
        result["delta_vs_no_drift"] = deltas


def build_report(
    config: ProjectConfig,
    *,
    max_evaluation_rows: int = DEFAULT_MAX_EVALUATION_ROWS,
    scenarios: Sequence[str] = SCENARIOS,
) -> dict[str, Any]:
    """Fit the frozen baselines once, then evaluate every requested scenario."""

    from sentinelflow.suite import ModelSuite

    warmup, _ = load_warmup_and_stream(DataSettings.from_config(config), max_rows=None)
    preprocessor = FraudPreprocessor.from_config(config)
    warmup_features = preprocessor.fit_transform(warmup.features)

    static_models = ModelSuite.from_config(config).static_models
    static_models.fit(warmup_features, warmup.labels)

    drift = config.section("drift")
    report: dict[str, Any] = {
        "seed": config.seed,
        "warmup_rows": warmup.row_count,
        "evaluation_rows": max_evaluation_rows,
        "drift_schedule": {
            "start_fraction": drift["start_fraction"],
            "end_fraction": drift["end_fraction"],
            "max_intensity": drift["max_intensity"],
        },
        "scenarios": {},
    }
    for scenario in scenarios:
        report["scenarios"][scenario] = evaluate_scenario(
            config,
            scenario,
            max_evaluation_rows=max_evaluation_rows,
            preprocessor=preprocessor,
            static_models=static_models,
        )

    add_deltas(report)
    return report


def format_summary(report: dict[str, Any]) -> str:
    """Compact human-readable table of separation and frozen-baseline metrics."""

    def number(value: float | None) -> str:
        return "   n/a" if value is None else f"{value:6.3f}"

    lines = [
        f"{'scenario':<13}{'phase':<12}{'model':<22}{'|d|':>7}{'TVD':>7}{'AUC':>8}{'recall':>8}{'FPR':>8}"
    ]
    for scenario, result in report["scenarios"].items():
        for phase in PHASES:
            if phase not in result["separation"]:
                continue
            sep = result["separation"][phase]["mean_abs_d"]
            tvd = result["separation"][phase]["mean_tvd"]
            for name, per_phase in result["models"].items():
                metrics = per_phase.get(phase)
                if metrics is None:
                    continue
                lines.append(
                    f"{scenario:<13}{phase:<12}{name:<22}{number(sep):>7}"
                    f"{number(tvd):>7}{number(metrics['roc_auc']):>8}{number(metrics['recall']):>8}"
                    f"{number(metrics['false_positive_rate']):>8}"
                )
    return "\n".join(lines)


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare the three SentinelFlow drift scenarios.")
    parser.add_argument("--config", type=Path, default=Path("configs/full.yaml"))
    parser.add_argument(
        "--max-evaluation-rows", type=int, default=DEFAULT_MAX_EVALUATION_ROWS, help="Stream size."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Report path; defaults to results/metrics/drift_scenarios/scenario_report.json.",
    )
    parsed = parser.parse_args(arguments)

    config = load_config(parsed.config)
    report = build_report(config, max_evaluation_rows=parsed.max_evaluation_rows)

    output = parsed.output or (
        config.results_path / "metrics" / "drift_scenarios" / "scenario_report.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(format_summary(report))
    print(f"\nReport written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
