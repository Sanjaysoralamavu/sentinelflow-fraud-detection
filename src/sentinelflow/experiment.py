"""End-to-end SentinelFlow experiment orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Protocol

import numpy as np

from sentinelflow.config import ProjectConfig
from sentinelflow.data.loader import DataSettings, load_warmup_and_stream
from sentinelflow.data.preprocessing import FraudPreprocessor
from sentinelflow.models.common import DriftEvent
from sentinelflow.models.static import FeatureMatrix
from sentinelflow.streaming.contracts import Batch
from sentinelflow.streaming.drift_injector import DriftInjector, DriftSettings
from sentinelflow.streaming.label_delay import (
    DelayedLabelQueue,
    StreamSettings,
    batch_chunks,
    run_stream,
)
from sentinelflow.suite import ModelSuite


class PredictionRecorder(Protocol):
    """Interface consumed later by the evaluation work package."""

    def record(
        self,
        batch: Batch,
        predictions: dict[str, np.ndarray],
    ) -> None:
        """Record predictions produced before delayed learning."""


class SummaryRecorder:
    """Collect bounded run statistics without retaining row-level predictions."""

    def __init__(self) -> None:
        self.batch_count = 0
        self.row_count = 0
        self.fraud_rows = 0
        self.phase_counts: dict[str, int] = {}
        self.model_names: tuple[str, ...] = ()
        self.last_batch_id: int | None = None

    def record(
        self,
        batch: Batch,
        predictions: dict[str, np.ndarray],
    ) -> None:
        """Record one batch's bounded summary information."""

        names = tuple(predictions)

        if not self.model_names:
            self.model_names = names
        elif names != self.model_names:
            raise RuntimeError("The model set changed during the experiment stream.")

        for name, scores in predictions.items():
            values = np.asarray(scores).reshape(-1)

            if values.shape != (batch.row_count,):
                raise ValueError(
                    f"Model '{name}' produced {len(values)} predictions "
                    f"for a batch containing {batch.row_count} rows."
                )

        if self.last_batch_id is not None and batch.batch_id != self.last_batch_id + 1:
            raise RuntimeError("Experiment batches are not sequential.")

        self.batch_count += 1
        self.row_count += batch.row_count
        self.fraud_rows += int(np.count_nonzero(batch.labels))
        self.phase_counts[batch.phase] = self.phase_counts.get(batch.phase, 0) + batch.row_count
        self.last_batch_id = batch.batch_id


@dataclass(frozen=True)
class ExperimentSummary:
    """Bounded summary of one completed experiment."""

    seed: int
    warmup_rows: int
    evaluation_rows: int
    fraud_rows: int
    batch_count: int
    model_names: tuple[str, ...]
    feature_count: int
    label_delay_batches: int
    drift_event_count: int
    phase_counts: dict[str, int]
    runtime_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""

        return asdict(self)


@dataclass(frozen=True)
class ExperimentResult:
    """Experiment output shared with evaluation and reporting code."""

    summary: ExperimentSummary
    drift_events: tuple[DriftEvent, ...]
    feature_manifest: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return all bounded experiment metadata as a dictionary."""

        return {
            "summary": self.summary.to_dict(),
            "drift_events": [asdict(event) for event in self.drift_events],
            "feature_manifest": self.feature_manifest,
        }


def run_experiment(
    config: ProjectConfig,
    *,
    recorder: PredictionRecorder | None = None,
    max_evaluation_rows: int | None = None,
) -> ExperimentResult:
    """Run one complete predict-before-learn SentinelFlow experiment.

    The preprocessor and every model are fitted on the warm-up window.
    Evaluation batches are then scored before their labels enter the
    delayed-label queue. Only adaptive models learn from released labels.

    Args:
        config: Validated project configuration.
        recorder: Optional evaluation callback receiving every scored batch.
            The runner itself stores only bounded summary information.
        max_evaluation_rows: Optional number of evaluation rows for smoke
            runs. When omitted, the verified audit determines the full
            evaluation-stream size.
    """

    started = perf_counter()
    evaluation_rows = _resolve_evaluation_rows(
        config,
        max_evaluation_rows=max_evaluation_rows,
    )

    data_settings = DataSettings.from_config(config)
    stream_settings = StreamSettings.from_config(
        data=config.section("data"),
        stream=config.section("stream"),
        seed=config.seed,
    )
    drift_settings = DriftSettings.from_config(config.section("drift"))

    maximum_loaded_rows = config.warmup_rows + evaluation_rows
    warmup, evaluation_chunks = load_warmup_and_stream(
        data_settings,
        max_rows=maximum_loaded_rows,
    )

    if warmup.row_count != config.warmup_rows:
        raise RuntimeError(
            f"Expected {config.warmup_rows} warm-up rows, but loaded {warmup.row_count}."
        )

    preprocessor = FraudPreprocessor.from_config(config)
    warmup_features = preprocessor.fit_transform(warmup.features)

    if preprocessor.manifest is None:
        raise RuntimeError("The fitted preprocessor did not produce a feature manifest.")

    models = ModelSuite.from_config(config)
    models.warm_up(warmup_features, warmup.labels)

    drift = DriftInjector(drift_settings)
    queue = DelayedLabelQueue(stream_settings.label_delay_batches)
    summary_recorder = SummaryRecorder()

    batches = batch_chunks(
        chunks=evaluation_chunks,
        total_rows=evaluation_rows,
        settings=stream_settings,
        drift=drift,
    )

    def predict(
        batch: Batch,
    ) -> tuple[FeatureMatrix, dict[str, np.ndarray]]:
        transformed = preprocessor.transform(batch.features)
        predictions = models.predict_probabilities(transformed)
        return transformed, predictions

    def record(
        batch: Batch,
        predictions: dict[str, np.ndarray],
    ) -> None:
        summary_recorder.record(batch, predictions)

        if recorder is not None:
            recorder.record(batch, predictions)

    run_stream(
        batches=batches,
        queue=queue,
        predict=predict,
        record=record,
        learn=models.learn,
    )

    if summary_recorder.row_count != evaluation_rows:
        raise RuntimeError(
            f"Expected {evaluation_rows} evaluation rows, "
            f"but processed {summary_recorder.row_count}."
        )

    drift_events = tuple(models.drift_events)

    summary = ExperimentSummary(
        seed=config.seed,
        warmup_rows=warmup.row_count,
        evaluation_rows=summary_recorder.row_count,
        fraud_rows=summary_recorder.fraud_rows,
        batch_count=summary_recorder.batch_count,
        model_names=summary_recorder.model_names,
        feature_count=preprocessor.manifest.n_output_features,
        label_delay_batches=stream_settings.label_delay_batches,
        drift_event_count=len(drift_events),
        phase_counts=dict(summary_recorder.phase_counts),
        runtime_seconds=round(perf_counter() - started, 6),
    )

    return ExperimentResult(
        summary=summary,
        drift_events=drift_events,
        feature_manifest=preprocessor.manifest.to_dict(),
    )


def _resolve_evaluation_rows(
    config: ProjectConfig,
    *,
    max_evaluation_rows: int | None,
) -> int:
    """Resolve a smoke-run limit or the audited full evaluation size."""

    if max_evaluation_rows is not None:
        if max_evaluation_rows <= 0:
            raise ValueError("max_evaluation_rows must be greater than zero.")

        return int(max_evaluation_rows)

    return _load_audited_evaluation_rows(config)


def _load_audited_evaluation_rows(config: ProjectConfig) -> int:
    """Read the verified strict-CNP evaluation count from the audit."""

    audit_path = config.results_path / "metrics" / "audit" / "dataset_audit.json"

    if not audit_path.is_file():
        raise FileNotFoundError(
            f"Dataset audit is required for a full experiment run: {audit_path}"
        )

    with audit_path.open(encoding="utf-8") as handle:
        audit = json.load(handle)

    try:
        counts = audit["counts"]
        audited_warmup_rows = int(counts["warmup_rows"])
        evaluation_rows = int(counts["evaluation_rows"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Dataset audit has invalid row counts: {audit_path}") from error

    if audited_warmup_rows != config.warmup_rows:
        raise ValueError(
            "Configured warm-up rows do not match the verified audit: "
            f"{config.warmup_rows} configured versus "
            f"{audited_warmup_rows} audited."
        )

    if evaluation_rows <= 0:
        raise ValueError("The dataset audit must contain a positive evaluation row count.")

    return evaluation_rows
