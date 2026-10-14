"""Deterministic batching, delayed-label queue, and prequential driver
for the SentinelFlow evaluation stream (work package 3).

``batch_stream`` turns the warm-up-excluded evaluation rows into
deterministic, drift-injected ``Batch`` objects. ``DelayedLabelQueue`` holds
each batch's labels back until its configured delay elapses, releasing them
in source-batch order once the stream clock (the current batch id) reaches
``observed_batch``. ``run_stream`` wires the two together with caller-
supplied predict/record/learn steps, enforcing predict-before-learn on every
batch -- including a zero-batch delay.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sentinelflow.streaming.contracts import Batch, DelayedBatch
from sentinelflow.streaming.drift_injector import DriftInjector


@dataclass(frozen=True)
class StreamSettings:
    """Validated parameters for batch ordering and label delay."""

    batch_size: int
    label_delay_batches: int
    seed: int
    row_group_shuffle: bool = True
    within_batch_shuffle: bool = True

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if self.label_delay_batches < 0:
            raise ValueError("label_delay_batches must be nonnegative.")

    @classmethod
    def from_config(cls, data: dict, stream: dict, seed: int) -> StreamSettings:
        return cls(
            batch_size=int(data["batch_size"]),
            label_delay_batches=int(stream["label_delay_batches"]),
            seed=seed,
            row_group_shuffle=bool(stream.get("row_group_shuffle", True)),
            within_batch_shuffle=bool(stream.get("within_batch_shuffle", True)),
        )


def deterministic_order(n_rows: int, settings: StreamSettings) -> np.ndarray:
    """Return a deterministic row order for the evaluation stream.

    When ``row_group_shuffle`` is enabled, the full row order is permuted
    once (seeded by ``settings.seed``); otherwise rows keep their source
    order. The same seed always returns the same permutation.
    """

    if not settings.row_group_shuffle:
        return np.arange(n_rows)
    rng = np.random.default_rng(settings.seed)
    return rng.permutation(n_rows)


def batch_stream(
    features: pd.DataFrame,
    labels: np.ndarray,
    settings: StreamSettings,
    drift: DriftInjector,
) -> Iterator[Batch]:
    """Yield deterministic, drift-injected batches of the evaluation stream.

    ``features``/``labels`` are the warm-up-excluded, already-scoped
    evaluation rows (``is_fraud`` separated) in their original source order:
    row 0 is the first evaluation row, not the first dataset row. Batch 0
    starts at offset 0; the last batch may be shorter than
    ``settings.batch_size``.
    """

    n_rows = len(labels)
    if n_rows == 0:
        return

    order = deterministic_order(n_rows, settings)
    n_batches = math.ceil(n_rows / settings.batch_size)
    labels = np.asarray(labels)

    for batch_id in range(n_batches):
        start = batch_id * settings.batch_size
        stop = min(start + settings.batch_size, n_rows)
        positions = order[start:stop]

        if settings.within_batch_shuffle:
            batch_rng = np.random.default_rng((settings.seed, batch_id))
            positions = positions[batch_rng.permutation(len(positions))]

        batch_features = features.iloc[positions].reset_index(drop=True)
        batch_labels = labels[positions]
        progress = stop / n_rows

        result = drift.apply(batch_features, batch_labels, progress)

        yield Batch(
            batch_id=batch_id,
            source_rows=positions,
            progress=progress,
            phase=result.phase,
            features=result.features,
            labels=batch_labels,
        )


class DelayedLabelQueue:
    """Holds a batch's labels back until its configured release batch arrives.

    Release is driven by the stream's own batch clock, not wall time: a
    pending item becomes due the first time ``release_due`` is called with
    ``current_batch_id >= observed_batch``. ``flush`` releases everything
    still pending, for use once the stream itself has ended.
    """

    def __init__(self, delay_batches: int) -> None:
        if delay_batches < 0:
            raise ValueError("delay_batches must be nonnegative.")
        self.delay_batches = delay_batches
        self._pending: deque[DelayedBatch] = deque()

    def __len__(self) -> int:
        return len(self._pending)

    def enqueue(self, batch: Batch, features: Any, predictions: dict[str, np.ndarray]) -> None:
        """Queue ``batch`` for release at ``batch.batch_id + delay_batches``.

        ``features`` is the preprocessed matrix used to produce
        ``predictions`` (not necessarily ``batch.features``, which is the
        pre-preprocessing frame).
        """

        self._pending.append(
            DelayedBatch(
                source_batch=batch.batch_id,
                observed_batch=batch.batch_id + self.delay_batches,
                features=features,
                labels=batch.labels,
                predictions=dict(predictions),
                phase=batch.phase,
            )
        )

    def release_due(self, current_batch_id: int) -> list[DelayedBatch]:
        """Pop and return, in source-batch order, every item due by ``current_batch_id``."""

        due: list[DelayedBatch] = []
        remaining: deque[DelayedBatch] = deque()

        for item in self._pending:
            if item.observed_batch <= current_batch_id:
                due.append(item)
            else:
                remaining.append(item)

        self._pending = remaining
        return due

    def flush(self) -> list[DelayedBatch]:
        """Release every remaining pending item, in source-batch order."""

        due = list(self._pending)
        self._pending.clear()
        return due


PredictFn = Callable[[Batch], tuple[Any, dict[str, np.ndarray]]]
RecordFn = Callable[[Batch, dict[str, np.ndarray]], None]
LearnFn = Callable[[DelayedBatch], None]


def run_stream(
    batches: Iterable[Batch],
    queue: DelayedLabelQueue,
    predict: PredictFn,
    record: RecordFn,
    learn: LearnFn,
) -> None:
    """Drive one full prequential pass: predict before learn, every batch.

    For every batch, in order: ``predict`` scores it and returns the
    preprocessed features alongside each model's probabilities (scoring
    must never change model state); ``record`` logs batch metrics from
    those scores; the batch is enqueued into ``queue``; and finally every
    item the queue now reports as due is handed to ``learn``. This keeps
    even a zero-batch delay correct, since release is checked only after
    the current batch has already been scored and recorded. Any items still
    queued after the last batch are flushed and learned from last.
    """

    for batch in batches:
        preprocessed, predictions = predict(batch)
        record(batch, predictions)
        queue.enqueue(batch, preprocessed, predictions)

        for due in queue.release_due(batch.batch_id):
            learn(due)

    for due in queue.flush():
        learn(due)
