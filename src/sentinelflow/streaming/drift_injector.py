"""Controlled, label-preserving concept-drift injection (work package 3).

Drift is injected once per batch, upstream of preprocessing, so every model
scores and learns from the identical drifted values (required end-to-end
behavior item 8 in the implementation plan).

Drift schedule (``drift.start_fraction`` / ``drift.end_fraction`` in
``configs/full.yaml``), as a function of stream progress:

- ``progress < start_fraction``             -> phase "pre_drift",  intensity 0.0
- ``start_fraction <= progress < end``      -> phase "ramp_drift", intensity ramps
                                                linearly from 0 to ``max_intensity``
- ``progress >= end_fraction``              -> phase "drifted",    intensity held at
                                                ``max_intensity``

Only rows with ``is_fraud == 1`` are perturbed, and only numeric feature
columns are touched; ``is_fraud`` itself is never part of the feature frame
this injector sees. Each perturbed column is shifted along a fixed
direction (``+1``/``-1``, drawn once per column from ``drift.random_seed``)
by ``intensity * shift_scale * column_std`` -- so the *direction* of drift is
stable for the whole stream and only its magnitude ramps up. Two runs with
the same seed therefore reproduce byte-identical drifted values.

The no-drift control (``drift.enabled: false``) forces intensity to 0.0 for
every batch, so ``apply`` always returns an unmodified copy of the input
features; phase is still computed for logging, but it never gates a feature
change while disabled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sentinelflow.streaming.contracts import PHASE_DRIFTED, PHASE_PRE_DRIFT, PHASE_RAMP_DRIFT

DEFAULT_SHIFT_SCALE = 2.0


@dataclass(frozen=True)
class DriftSettings:
    """Validated parameters for the controlled drift injector."""

    enabled: bool
    start_fraction: float
    end_fraction: float
    max_intensity: float
    random_seed: int
    shift_scale: float = DEFAULT_SHIFT_SCALE

    def __post_init__(self) -> None:
        if not 0.0 <= self.start_fraction < self.end_fraction <= 1.0:
            raise ValueError("start_fraction and end_fraction must satisfy 0 <= start < end <= 1.")
        if not 0.0 <= self.max_intensity <= 1.0:
            raise ValueError("max_intensity must be between zero and one.")
        if self.shift_scale < 0:
            raise ValueError("shift_scale must be nonnegative.")

    @classmethod
    def from_config(cls, drift: dict) -> DriftSettings:
        return cls(
            enabled=bool(drift["enabled"]),
            start_fraction=float(drift["start_fraction"]),
            end_fraction=float(drift["end_fraction"]),
            max_intensity=float(drift["max_intensity"]),
            random_seed=int(drift["random_seed"]),
        )


@dataclass(frozen=True)
class DriftResult:
    """Outcome of injecting drift into one batch."""

    features: pd.DataFrame
    phase: str
    intensity: float
    affected_rows: int
    affected_columns: tuple[str, ...]


def drift_phase(progress: float, settings: DriftSettings) -> str:
    """Return the stream phase for ``progress`` under ``settings``'s schedule."""

    if progress < settings.start_fraction:
        return PHASE_PRE_DRIFT
    if progress < settings.end_fraction:
        return PHASE_RAMP_DRIFT
    return PHASE_DRIFTED


def drift_intensity(progress: float, settings: DriftSettings) -> float:
    """Return the configured drift intensity at ``progress``; 0.0 when disabled."""

    if not settings.enabled:
        return 0.0
    if progress < settings.start_fraction:
        return 0.0
    if progress >= settings.end_fraction:
        return settings.max_intensity

    span = settings.end_fraction - settings.start_fraction
    ramp = (progress - settings.start_fraction) / span
    return settings.max_intensity * ramp


class DriftInjector:
    """Applies the configured controlled drift to fraud rows of one batch at a time.

    One injector instance is meant to serve an entire stream run: it caches
    each column's drift direction the first time that column is seen, so
    repeated calls across batches (and repeated runs with the same
    ``settings.random_seed``) stay consistent and reproducible.
    """

    def __init__(self, settings: DriftSettings, feature_columns: list[str] | None = None) -> None:
        self.settings = settings
        self._explicit_columns = tuple(feature_columns) if feature_columns is not None else None
        self._directions: dict[str, float] = {}
        self._direction_rng = np.random.default_rng(settings.random_seed)

    def _columns_for(self, features: pd.DataFrame) -> tuple[str, ...]:
        if self._explicit_columns is not None:
            return self._explicit_columns
        return tuple(features.select_dtypes(include="number").columns)

    def _direction(self, column: str) -> float:
        if column not in self._directions:
            self._directions[column] = float(self._direction_rng.choice([-1.0, 1.0]))
        return self._directions[column]

    def apply(self, features: pd.DataFrame, labels: np.ndarray, progress: float) -> DriftResult:
        """Return a drifted copy of ``features`` for a batch at ``progress`` through the stream."""

        if not 0.0 < progress <= 1.0:
            raise ValueError("progress must be in the interval (0, 1].")

        phase = drift_phase(progress, self.settings)
        intensity = drift_intensity(progress, self.settings)

        fraud_mask = np.asarray(labels) == 1

        if intensity <= 0.0 or not fraud_mask.any():
            return DriftResult(
                features=features.copy(deep=True),
                phase=phase,
                intensity=intensity,
                affected_rows=0,
                affected_columns=(),
            )

        drifted = features.copy(deep=True)
        shifted_columns: list[str] = []

        for column in self._columns_for(features):
            std = float(drifted[column].std(ddof=0))
            if not np.isfinite(std) or std == 0.0:
                continue
            shift = self._direction(column) * intensity * self.settings.shift_scale * std
            if drifted[column].dtype.kind != "f":
                # Integer columns (e.g. cards_on_device_30d, account_age_days in the real
                # schema) cannot hold a fractional shift in place without pandas raising a
                # dtype-incompatibility error; promote the whole column to float64 first so
                # every row keeps a consistent dtype.
                drifted[column] = drifted[column].astype("float64")
            drifted.loc[fraud_mask, column] = drifted.loc[fraud_mask, column] + shift
            shifted_columns.append(column)

        affected_rows = int(fraud_mask.sum()) if shifted_columns else 0

        return DriftResult(
            features=drifted,
            phase=phase,
            intensity=intensity,
            affected_rows=affected_rows,
            affected_columns=tuple(shifted_columns),
        )
