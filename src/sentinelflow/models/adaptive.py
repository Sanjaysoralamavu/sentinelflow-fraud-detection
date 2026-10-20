"""Adaptive incremental fraud models with optional ADWIN or DDM recovery.

Update order for every stream batch:

1. ``predict_proba`` scores the batch. Scoring never changes model state.
2. The runner stores each model's probabilities in ``DelayedBatch.predictions``.
3. When the labels are released, ``learn`` adds the rows to a bounded recent
   window, feeds a sample of the model's own past 0/1 errors to its detector,
   and then either updates the learner with ``partial_fit`` or, after a
   confirmed alarm, retrains a fresh learner on the recent window only.

No unreleased or future label is ever visible to a model or a detector,
because both only see a ``DelayedBatch`` after the runner releases it.

Detector and reset parameters (``models.adaptive`` in ``configs/full.yaml``):

========================== ========= ===============================================
Parameter                  Default   Meaning
========================== ========= ===============================================
detectors                  none,     Adaptive variants to build.
                           adwin,
                           ddm
alpha                      0.0001    L2 regularisation of the incremental learner.
detector_sample_size       2000      Errors sampled per released batch for detection.
recent_window_rows         200000    Maximum released rows kept for retraining.
adwin_delta                0.002     ADWIN confidence; smaller is more conservative.
ddm_warm_start             30        Observations DDM needs before it may signal.
detector_grace_batches     12        Released batches after warm-up whose alarms
                                     are logged but suppressed.
detector_cooldown_batches  20        Minimum observed batches between two
                                     confirmed alarms. A signal inside the
                                     cooldown is logged and held, then acted on
                                     at the first released batch after it.
========================== ========= ===============================================

Shared ``models`` settings: ``threshold`` (0.50) turns probabilities into the
0/1 predictions used for detector errors; ``positive_weight_cap`` (25.0) caps
the fraud-class weight ``min(negatives / positives, cap)``. The weight is
computed on the warm-up rows and recomputed on the retraining window after
every confirmed reset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from river import drift

from sentinelflow.models.common import (
    ALARM_CONFIRMED,
    ALARM_SUPPRESSED_COOLDOWN,
    ALARM_SUPPRESSED_GRACE,
    WARMUP_BATCH,
    DelayedBatch,
    DriftEvent,
    RecentWindow,
    fit_in_chunks,
    fraud_probability,
    new_learner,
    positive_class_weight,
    sample_weights,
)

DETECTOR_NAMES = ("none", "adwin", "ddm")


@dataclass(frozen=True)
class AdaptiveSettings:
    """Validated parameters for the adaptive models."""

    alpha: float = 0.0001
    threshold: float = 0.5
    positive_weight_cap: float = 25.0
    detector_sample_size: int = 2000
    recent_window_rows: int = 200_000
    adwin_delta: float = 0.002
    ddm_warm_start: int = 30
    detector_grace_batches: int = 12
    detector_cooldown_batches: int = 20

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError("alpha must be greater than zero.")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be between zero and one.")
        if self.positive_weight_cap < 1.0:
            raise ValueError("positive_weight_cap must be at least one.")
        if self.detector_sample_size <= 0:
            raise ValueError("detector_sample_size must be greater than zero.")
        if self.recent_window_rows <= 0:
            raise ValueError("recent_window_rows must be greater than zero.")
        if not 0.0 < self.adwin_delta < 1.0:
            raise ValueError("adwin_delta must be between zero and one.")
        if self.ddm_warm_start <= 0:
            raise ValueError("ddm_warm_start must be greater than zero.")
        if self.detector_grace_batches < 0 or self.detector_cooldown_batches < 0:
            raise ValueError("grace and cooldown batches must be nonnegative.")

    @classmethod
    def from_config(cls, models: dict) -> AdaptiveSettings:
        adaptive = models["adaptive"]
        return cls(
            alpha=float(adaptive["alpha"]),
            threshold=float(models["threshold"]),
            positive_weight_cap=float(models["positive_weight_cap"]),
            detector_sample_size=int(adaptive["detector_sample_size"]),
            recent_window_rows=int(adaptive["recent_window_rows"]),
            adwin_delta=float(adaptive["adwin_delta"]),
            ddm_warm_start=int(adaptive["ddm_warm_start"]),
            detector_grace_batches=int(adaptive["detector_grace_batches"]),
            detector_cooldown_batches=int(adaptive["detector_cooldown_batches"]),
        )


def build_detector(name: str, settings: AdaptiveSettings):
    """Return a fresh River drift detector, or ``None`` for the detector-free model."""

    if name == "none":
        return None
    if name == "adwin":
        return drift.ADWIN(delta=settings.adwin_delta)
    if name == "ddm":
        return drift.binary.DDM(warm_start=settings.ddm_warm_start)
    raise ValueError(f"Unknown detector '{name}'. Expected one of {DETECTOR_NAMES}.")


class AdaptiveModel:
    """Incremental logistic regression with delayed learning and drift recovery."""

    def __init__(self, detector: str, settings: AdaptiveSettings, seed: int) -> None:
        if detector not in DETECTOR_NAMES:
            raise ValueError(f"Unknown detector '{detector}'. Expected one of {DETECTOR_NAMES}.")

        self.detector_name = detector
        self.name = f"adaptive_{detector}"
        self.settings = settings
        self.seed = seed

        self._rng = np.random.default_rng(seed)
        self._learner = new_learner(settings.alpha, seed)
        self._detector = build_detector(detector, settings)
        self._window = RecentWindow(settings.recent_window_rows)
        self._fraud_weight = 1.0
        self._learned_batches = 0
        self._last_confirmed_batch: int | None = None
        self._pending_alarm = False
        self._is_fitted = False

        self.reset_count = 0
        self.events: list[DriftEvent] = []

    @property
    def fraud_weight(self) -> float:
        return self._fraud_weight

    def warm_up(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fit the initial learner on the warm-up rows and seed the recent window."""

        labels = np.asarray(labels).astype(int)
        self._fraud_weight = positive_class_weight(labels, self.settings.positive_weight_cap)
        fit_in_chunks(self._learner, features, labels, sample_weights(labels, self._fraud_weight))
        self._window.add(WARMUP_BATCH, features, labels)
        self._is_fitted = True

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return one fraud probability per row without changing model state."""

        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be warmed up before scoring.")
        return fraud_probability(self._learner, features)

    def learn(self, batch: DelayedBatch) -> DriftEvent | None:
        """Learn from released labels.

        Returns the ``DriftEvent`` logged for this batch, if the detector
        signalled; ``event.confirmed`` tells whether a reset happened.
        """

        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be warmed up before learning.")

        labels = np.asarray(batch.labels).astype(int)
        self._window.add(batch.source_batch, batch.features, labels)
        self._learned_batches += 1

        event = self._check_detector(batch, labels)

        if event is not None and event.confirmed:
            return event

        fit_in_chunks(
            self._learner,
            batch.features,
            labels,
            sample_weights(labels, self._fraud_weight),
        )
        return event

    def _check_detector(self, batch: DelayedBatch, labels: np.ndarray) -> DriftEvent | None:
        if self._detector is None:
            return None

        scores = batch.predictions.get(self.name)
        if scores is None:
            raise KeyError(
                f"DelayedBatch {batch.source_batch} has no stored predictions for {self.name}."
            )

        predicted = (np.asarray(scores) >= self.settings.threshold).astype(int)
        errors = (predicted != labels).astype(int)
        sample_size = min(self.settings.detector_sample_size, len(errors))
        sampled = np.sort(self._rng.choice(len(errors), size=sample_size, replace=False))

        signalled = False
        for error in errors[sampled]:
            self._detector.update(int(error))
            if self._detector.drift_detected:
                signalled = True

        in_cooldown = (
            self._last_confirmed_batch is not None
            and batch.observed_batch - self._last_confirmed_batch
            < self.settings.detector_cooldown_batches
        )

        if not signalled:
            # River detectors clear their state when they signal, so a signal
            # suppressed by the cooldown would otherwise be lost. It is held
            # and acted on at the first released batch after the cooldown.
            if self._pending_alarm and not in_cooldown:
                return self._reset(batch)
            return None

        if self._learned_batches <= self.settings.detector_grace_batches:
            return self._log(batch, ALARM_SUPPRESSED_GRACE)

        if in_cooldown:
            self._pending_alarm = True
            return self._log(batch, ALARM_SUPPRESSED_COOLDOWN)

        return self._reset(batch)

    def _log(self, batch: DelayedBatch, state: str, **recovery) -> DriftEvent:
        event = DriftEvent(
            model=self.name,
            detector=self.detector_name,
            source_batch=batch.source_batch,
            observed_batch=batch.observed_batch,
            phase=batch.phase,
            alarm_type="drift",
            alarm_state=state,
            reset_count=self.reset_count,
            **recovery,
        )
        self.events.append(event)
        return event

    def _reset(self, batch: DelayedBatch) -> DriftEvent:
        features, labels = self._window.arrays()
        window_batches = self._window.source_batches()

        self.reset_count += 1
        self._last_confirmed_batch = batch.observed_batch
        self._pending_alarm = False
        self._fraud_weight = positive_class_weight(labels, self.settings.positive_weight_cap)
        self._learner = new_learner(self.settings.alpha, self.seed + self.reset_count)
        fit_in_chunks(self._learner, features, labels, sample_weights(labels, self._fraud_weight))
        self._detector = build_detector(self.detector_name, self.settings)

        return self._log(
            batch,
            ALARM_CONFIRMED,
            retrain_rows=len(labels),
            retrain_first_batch=min(window_batches),
            retrain_last_batch=max(window_batches),
            fraud_weight=self._fraud_weight,
        )


def build_adaptive_models(models: dict, seed: int) -> dict:
    """Build the adaptive variants and the online ensemble from ``config.section("models")``."""

    from sentinelflow.models.online_ensemble import OnlineBaggingEnsemble

    settings = AdaptiveSettings.from_config(models)
    detectors = models["adaptive"]["detectors"]
    if not detectors:
        raise ValueError("models.adaptive.detectors must not be empty.")

    built: dict = {}
    for detector in detectors:
        model = AdaptiveModel(str(detector), settings, seed=seed)
        built[model.name] = model

    if models.get("online_ensemble", {}).get("enabled", False):
        ensemble = OnlineBaggingEnsemble.from_config(models, seed=seed)
        built[ensemble.name] = ensemble

    return built
