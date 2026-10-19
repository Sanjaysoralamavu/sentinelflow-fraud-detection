"""Shared records and helpers for SentinelFlow stream models.

``DelayedBatch`` mirrors the delayed-label contract in the implementation plan
(source batch, observed batch, features, labels, predictions). ``DriftEvent``
mirrors the drift-event contract (detector, source batch, observed batch,
phase, alarm type) and adds the reset and recovery metadata required by
work package 4.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy import sparse
from sklearn.linear_model import SGDClassifier

CLASSES = np.array([0, 1])
WARMUP_BATCH = -1


def n_rows(features) -> int:
    """Row count for dense arrays and SciPy sparse matrices alike."""

    return int(features.shape[0])


def stack_rows(parts: list):
    """Stack feature blocks, keeping sparse input sparse."""

    if any(sparse.issparse(part) for part in parts):
        return sparse.vstack(parts, format="csr")
    return np.concatenate(parts)


ALARM_CONFIRMED = "confirmed"
ALARM_SUPPRESSED_GRACE = "suppressed_grace"
ALARM_SUPPRESSED_COOLDOWN = "suppressed_cooldown"


@dataclass(frozen=True)
class DelayedBatch:
    """A labelled batch released to the learners after the configured delay.

    Attributes:
        source_batch: Batch index at which the rows were scored.
        observed_batch: Batch index at which the labels became available.
        features: Preprocessed feature matrix, one row per transaction.
        labels: Binary fraud labels (``1`` = fraud).
        predictions: Fraud probabilities each model produced at scoring time,
            keyed by model name. Detectors read their own model's entry.
        phase: Stream phase of the source batch (for example ``pre_drift``).
    """

    source_batch: int
    observed_batch: int
    features: np.ndarray
    labels: np.ndarray
    predictions: dict[str, np.ndarray] = field(default_factory=dict)
    phase: str | None = None

    def __post_init__(self) -> None:
        if self.observed_batch < self.source_batch:
            raise ValueError("observed_batch cannot precede source_batch.")
        if n_rows(self.features) != len(self.labels):
            raise ValueError("features and labels must have the same number of rows.")


@dataclass(frozen=True)
class DriftEvent:
    """One detector signal and, when confirmed, the recovery it triggered.

    Attributes:
        model: Model name, for example ``adaptive_adwin``.
        detector: ``adwin`` or ``ddm``.
        source_batch: Batch whose errors produced the signal.
        observed_batch: Batch at which those labels were released.
        phase: Stream phase of the source batch, when provided.
        alarm_type: Always ``drift`` for the detectors used here.
        alarm_state: ``confirmed``, ``suppressed_grace`` or ``suppressed_cooldown``.
        reset_count: Number of confirmed resets so far, including this one.
        retrain_rows: Rows used to retrain the fresh learner (0 if suppressed).
        retrain_first_batch: Oldest source batch in the retraining window
            (``-1`` means warm-up rows), or ``None`` if suppressed.
        retrain_last_batch: Newest source batch in the retraining window.
        fraud_weight: Fraud-class weight used for retraining.
    """

    model: str
    detector: str
    source_batch: int
    observed_batch: int
    phase: str | None
    alarm_type: str
    alarm_state: str
    reset_count: int
    retrain_rows: int = 0
    retrain_first_batch: int | None = None
    retrain_last_batch: int | None = None
    fraud_weight: float | None = None

    @property
    def confirmed(self) -> bool:
        return self.alarm_state == ALARM_CONFIRMED


class StreamModel(Protocol):
    """Interface every adaptive model exposes to the experiment runner."""

    name: str

    def warm_up(self, features: np.ndarray, labels: np.ndarray) -> None: ...

    def predict_proba(self, features: np.ndarray) -> np.ndarray: ...

    def learn(self, batch: DelayedBatch) -> DriftEvent | None: ...


def positive_class_weight(labels: np.ndarray, cap: float) -> float:
    """Return the fraud-class weight ``min(negatives / positives, cap)``, at least 1."""

    labels = np.asarray(labels)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)

    if positives == 0:
        return 1.0

    return float(min(max(negatives / positives, 1.0), cap))


def sample_weights(labels: np.ndarray, fraud_weight: float) -> np.ndarray:
    """Weight fraud rows by ``fraud_weight`` and legitimate rows by one."""

    return np.where(np.asarray(labels) == 1, fraud_weight, 1.0)


def new_learner(alpha: float, seed: int) -> SGDClassifier:
    """Create an averaged-SGD incremental logistic-regression learner.

    Averaging the SGD weights (ASGD) keeps single-pass updates stable on
    heavily weighted, imbalanced fraud batches.
    """

    return SGDClassifier(
        loss="log_loss",
        alpha=alpha,
        learning_rate="optimal",
        average=True,
        random_state=seed,
    )


def fit_in_chunks(
    learner: SGDClassifier,
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    chunk_size: int = 50_000,
) -> None:
    """Call ``partial_fit`` over bounded chunks to keep memory predictable."""

    for start in range(0, len(labels), chunk_size):
        stop = start + chunk_size
        learner.partial_fit(
            features[start:stop],
            labels[start:stop],
            classes=CLASSES,
            sample_weight=weights[start:stop],
        )


def fraud_probability(learner: SGDClassifier, features: np.ndarray) -> np.ndarray:
    """Return one fraud probability per row from a fitted learner."""

    return learner.predict_proba(features)[:, 1]


class RecentWindow:
    """Bounded memory of the most recently released labelled rows.

    Each stored part remembers its source batch so recovery metadata can
    state exactly which released batches were used for retraining.
    """

    def __init__(self, max_rows: int) -> None:
        if max_rows <= 0:
            raise ValueError("max_rows must be greater than zero.")
        self.max_rows = max_rows
        self._parts: deque[tuple[int, np.ndarray, np.ndarray]] = deque()
        self._rows = 0

    def __len__(self) -> int:
        return self._rows

    def add(self, source_batch: int, features: np.ndarray, labels: np.ndarray) -> None:
        if len(labels) >= self.max_rows:
            self._parts = deque(
                [(source_batch, features[-self.max_rows :], labels[-self.max_rows :])]
            )
            self._rows = self.max_rows
            return

        self._parts.append((source_batch, features, labels))
        self._rows += len(labels)

        while self._rows > self.max_rows:
            batch_id, old_features, old_labels = self._parts.popleft()
            overflow = self._rows - self.max_rows
            if overflow < len(old_labels):
                self._parts.appendleft((batch_id, old_features[overflow:], old_labels[overflow:]))
                self._rows -= overflow
            else:
                self._rows -= len(old_labels)

    def source_batches(self) -> list[int]:
        return [part[0] for part in self._parts]

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._parts:
            raise ValueError("The recent window is empty.")
        features = stack_rows([part[1] for part in self._parts])
        labels = np.concatenate([part[2] for part in self._parts])
        return features, labels
