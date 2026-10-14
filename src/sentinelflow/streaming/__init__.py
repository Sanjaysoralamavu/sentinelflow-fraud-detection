"""Pseudo-stream, delayed labels, and controlled drift (work package 3)."""

from sentinelflow.streaming.contracts import (
    PHASE_DRIFTED,
    PHASE_PRE_DRIFT,
    PHASE_RAMP_DRIFT,
    WARMUP_BATCH,
    Batch,
    DelayedBatch,
)
from sentinelflow.streaming.drift_injector import DriftInjector, DriftResult, DriftSettings
from sentinelflow.streaming.label_delay import (
    DelayedLabelQueue,
    StreamSettings,
    batch_stream,
    deterministic_order,
    run_stream,
)

__all__ = [
    "PHASE_DRIFTED",
    "PHASE_PRE_DRIFT",
    "PHASE_RAMP_DRIFT",
    "WARMUP_BATCH",
    "Batch",
    "DelayedBatch",
    "DriftInjector",
    "DriftResult",
    "DriftSettings",
    "DelayedLabelQueue",
    "StreamSettings",
    "batch_stream",
    "deterministic_order",
    "run_stream",
]
