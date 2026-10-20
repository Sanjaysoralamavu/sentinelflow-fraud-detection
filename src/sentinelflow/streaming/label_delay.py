"""Deterministic batching, delayed-label queue, and prequential driver
for the SentinelFlow evaluation stream (work package 3).

``batch_stream`` (in-memory frame) and ``batch_chunks`` (the data loader's
``CNPChunk`` iterator) turn the warm-up-excluded evaluation rows into
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

from sentinelflow.data.loader import CNPChunk
from sentinelflow.streaming.contracts import Batch, DelayedBatch
from sentinelflow.streaming.drift_injector import DriftInjector


@dataclass(frozen=True)
class StreamSettings:
    """Validated parameters for batch ordering and label delay."""

    batch_size: int
    label_delay_batches: int
    seed: int
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
            within_batch_shuffle=bool(stream.get("within_batch_shuffle", True)),
        )


def _make_batch(
    batch_id: int,
    features: pd.DataFrame,
    labels: np.ndarray,
    source_rows: np.ndarray,
    progress: float,
    settings: StreamSettings,
    drift: DriftInjector,
) -> Batch:
    """Apply the seeded within-batch shuffle, then drift, to one batch of rows."""

    if settings.within_batch_shuffle:
        batch_rng = np.random.default_rng((settings.seed, batch_id))
        positions = batch_rng.permutation(len(labels))
        features = features.iloc[positions]
        labels = labels[positions]
        source_rows = source_rows[positions]

    features = features.reset_index(drop=True)
    result = drift.apply(features, labels, progress)

    return Batch(
        batch_id=batch_id,
        source_rows=source_rows,
        progress=progress,
        phase=result.phase,
        features=result.features,
        labels=labels,
    )


def batch_stream(
    features: pd.DataFrame,
    labels: np.ndarray,
    settings: StreamSettings,
    drift: DriftInjector,
) -> Iterator[Batch]:
    """Yield deterministic, drift-injected batches of an in-memory evaluation stream.

    ``features``/``labels`` are the warm-up-excluded, already-scoped
    evaluation rows (``is_fraud`` separated) in stream order. Row-group
    shuffling belongs to the data loader (``stream.row_group_shuffle``), so
    rows are batched in the order given. Batch 0 starts at offset 0; the
    last batch may be shorter than ``settings.batch_size``.
    """

    n_rows = len(labels)
    labels = np.asarray(labels)

    for batch_id in range(math.ceil(n_rows / settings.batch_size)):
        start = batch_id * settings.batch_size
        stop = min(start + settings.batch_size, n_rows)
        yield _make_batch(
            batch_id,
            features.iloc[start:stop],
            labels[start:stop],
            np.arange(start, stop),
            stop / n_rows,
            settings,
            drift,
        )


def batch_chunks(
    chunks: Iterable[CNPChunk],
    total_rows: int,
    settings: StreamSettings,
    drift: DriftInjector,
) -> Iterator[Batch]:
    """Re-batch the loader's ``CNPChunk`` stream into fixed-size drifted batches.

    ``chunks`` is the evaluation iterator from ``load_warmup_and_stream`` /
    ``split_warmup``. Chunks follow Parquet row groups and are not batch
    sized, so rows are buffered and cut every ``settings.batch_size`` rows;
    memory stays bounded to about one chunk plus one batch. ``total_rows``
    (the evaluation row count) is needed up front because drift phases are a
    function of stream progress. ``Batch.source_rows`` holds the loader's
    global strict-CNP row positions.
    """

    if total_rows <= 0:
        return

    buffered: list[CNPChunk] = []
    buffered_rows = 0
    batch_id = 0
    emitted = 0

    def cut(count: int) -> Batch:
        nonlocal buffered, buffered_rows, batch_id, emitted
        frame = pd.concat([c.features for c in buffered], ignore_index=True)
        labels = np.concatenate([c.labels for c in buffered])
        rows = np.concatenate([c.cnp_rows for c in buffered])

        rest = len(labels) - count
        buffered = (
            [CNPChunk(0, rows[count:], frame.iloc[count:].reset_index(drop=True), labels[count:])]
            if rest
            else []
        )
        buffered_rows = rest
        emitted += count

        batch = _make_batch(
            batch_id,
            frame.iloc[:count],
            labels[:count],
            rows[:count],
            min(emitted / total_rows, 1.0),
            settings,
            drift,
        )
        batch_id += 1
        return batch

    for chunk in chunks:
        buffered.append(chunk)
        buffered_rows += chunk.row_count
        while buffered_rows >= settings.batch_size:
            yield cut(settings.batch_size)

    if buffered_rows:
        yield cut(buffered_rows)


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
