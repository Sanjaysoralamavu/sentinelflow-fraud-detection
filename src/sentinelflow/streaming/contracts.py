"""Stream contracts shared across SentinelFlow packages (work package 3).

``Batch`` is the unit the stream engine produces for every source chunk,
before preprocessing: drift has already been injected, but features are
still the raw (post-scope, pre-transform) frame. ``DelayedBatch`` is the
unit released to the models once its configured delay elapses; its
``features`` are whatever the preprocessing + prediction step produced
(typically a preprocessed numpy or sparse matrix), not ``Batch.features``.

``DelayedBatch`` and ``WARMUP_BATCH`` are re-exported from
``sentinelflow.models.common`` (work package 4), the single definition the
adaptive models consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sentinelflow.models.common import WARMUP_BATCH, DelayedBatch

PHASE_PRE_DRIFT = "pre_drift"
PHASE_RAMP_DRIFT = "ramp_drift"
PHASE_DRIFTED = "drifted"


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


__all__ = [
    "PHASE_DRIFTED",
    "PHASE_PRE_DRIFT",
    "PHASE_RAMP_DRIFT",
    "WARMUP_BATCH",
    "Batch",
    "DelayedBatch",
]
