"""Stream contracts shared across SentinelFlow packages (work package 3).

``Batch`` is the unit the stream engine produces for every source chunk,
before preprocessing: drift has already been injected, but features are
still the raw (post-scope, pre-transform) frame. ``DelayedBatch`` is the
unit released to the models once its configured delay elapses; its
``features`` are whatever the preprocessing + prediction step produced
(typically a preprocessed numpy or sparse matrix), not ``Batch.features``.

``DelayedBatch`` is field-for-field compatible with the consumer-side copy
in ``sentinelflow.models.common`` (work package 4). The two packages define
it independently so each can be built and tested without depending on the
other's unmerged branch; once both branches land on ``main``, one should
import the other's definition instead of duplicating it (see the
contract-change procedure in the implementation plan).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

PHASE_PRE_DRIFT = "pre_drift"
PHASE_RAMP_DRIFT = "ramp_drift"
PHASE_DRIFTED = "drifted"

WARMUP_BATCH = -1


def _row_count(features: Any) -> int:
    """Row count for DataFrames, dense arrays, and SciPy sparse matrices alike."""

    return int(features.shape[0])


@dataclass(frozen=True)
class Batch:
    """One deterministic chunk of the evaluation stream, before preprocessing.

    Attributes:
        batch_id: Zero-based, monotonically increasing index of this batch.
        source_rows: Original evaluation-row positions included in this
            batch, in the exact order used for ``features``/``labels``.
        progress: Fraction of the evaluation stream completed by the end of
            this batch, in the interval (0, 1].
        phase: One of ``PHASE_PRE_DRIFT``, ``PHASE_RAMP_DRIFT``, ``PHASE_DRIFTED``.
        features: Feature frame for this batch (drift already applied, if any).
        labels: Binary ``is_fraud`` labels aligned with ``features``.
    """

    batch_id: int
    source_rows: np.ndarray
    progress: float
    phase: str
    features: pd.DataFrame
    labels: np.ndarray

    def __post_init__(self) -> None:
        if self.batch_id < 0:
            raise ValueError("batch_id must be nonnegative.")
        if not 0.0 < self.progress <= 1.0:
            raise ValueError("progress must be in the interval (0, 1].")
        if len(self.source_rows) != _row_count(self.features):
            raise ValueError("source_rows and features must have the same number of rows.")
        if _row_count(self.features) != len(self.labels):
            raise ValueError("features and labels must have the same number of rows.")

    @property
    def row_count(self) -> int:
        return _row_count(self.features)


@dataclass(frozen=True)
class DelayedBatch:
    """A labelled batch released to the learners after the configured delay.

    Attributes:
        source_batch: Batch index at which the rows were scored.
        observed_batch: Batch index at which the labels became available
            (``source_batch`` plus the configured label delay).
        features: Preprocessed feature matrix used for prediction, one row
            per transaction.
        labels: Binary ``is_fraud`` labels aligned with ``features``.
        predictions: Fraud probabilities each model produced at scoring
            time, keyed by model name.
        phase: Stream phase of the source batch, if tracked.
    """

    source_batch: int
    observed_batch: int
    features: Any
    labels: np.ndarray
    predictions: dict[str, np.ndarray] = field(default_factory=dict)
    phase: str | None = None

    def __post_init__(self) -> None:
        if self.observed_batch < self.source_batch:
            raise ValueError("observed_batch cannot precede source_batch.")
        if _row_count(self.features) != len(self.labels):
            raise ValueError("features and labels must have the same number of rows.")
