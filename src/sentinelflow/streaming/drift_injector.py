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

Scenarios (``drift.scenario`` in ``configs/full.yaml``)
-------------------------------------------------------

``none`` (no-drift control)
    Intensity is 0.0 and every batch stays in phase "pre_drift", so ``apply``
    returns an unmodified copy of the input. Use it as the false-alarm and
    stability baseline: any model degradation or detector alarm here is not
    caused by injected drift.

``separability`` (the original scenario, unchanged)
    Fraud rows are shifted along a fixed random direction per numeric column,
    by ``intensity * shift_scale * column_std``. It is a covariate shift
    confined to the fraud class. Because the direction is random per column it
    can widen *or* narrow the gap to legitimate traffic, so its effect should
    be read from the separation diagnostics (``streaming.diagnostics``), not
    assumed. Interpretation: fraud behavior changes in a way a frozen model
    never saw; it tests whether stale decision boundaries still catch it.

``camouflage`` (degrading scenario)
    Fraud behavior mimics legitimate behavior. For a fixed set of numeric
    columns, each fraud value moves toward a version of itself rescaled to the
    legitimate batch's mean and standard deviation::

        matched = legit_mean + (x - fraud_mean) * legit_std / fraud_std
        x'      = (1 - a) * x + a * matched,      a = intensity * strength

    ``a = 0`` is the identity and ``a = 1`` gives fraud the legitimate mean
    and spread, so the fraud-versus-legitimate gap closes as the stream
    progresses. Legitimate rows and labels are never touched. The columns are
    either listed in ``drift.camouflage.columns`` or are the ``top_k`` numeric
    columns with the largest ``|d|`` in the first batch containing both
    classes; the choice is frozen for the whole stream. No random numbers are
    used, so output depends only on the input stream and the settings.
    Interpretation: attackers adapt to look normal. A frozen baseline loses
    ranking power (ROC-AUC) and recall, which is a stress test and not a
    guarantee that adaptive models will recover.

    Limitations: only numeric columns change. Categorical bin columns derived
    from numeric ones (for example ``geo_velocity_bin``) keep their original
    values, so a model that leans on those bins is under less pressure than
    one that leans on the raw numerics. Labels are used only to pick fraud
    rows and legitimate reference statistics inside the simulator; they are
    never added to the feature frame.

Compatibility: configs without ``drift.scenario`` behave as before
(``enabled: true`` is ``separability``, ``enabled: false`` is ``none``). When
``drift.scenario`` is present it wins over ``drift.enabled``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sentinelflow.streaming.contracts import PHASE_DRIFTED, PHASE_PRE_DRIFT, PHASE_RAMP_DRIFT
from sentinelflow.streaming.diagnostics import (
    MIN_CLASS_ROWS,
    categorical_columns,
    standardized_mean_difference,
    total_variation_distance,
)

DEFAULT_SHIFT_SCALE = 2.0
DEFAULT_CAMOUFLAGE_STRENGTH = 1.0
DEFAULT_CAMOUFLAGE_TOP_K = 4
DEFAULT_CAMOUFLAGE_CATEGORICAL_TOP_K = 8

SCENARIO_NONE = "none"
SCENARIO_SEPARABILITY = "separability"
SCENARIO_CAMOUFLAGE = "camouflage"
SCENARIOS = (SCENARIO_NONE, SCENARIO_SEPARABILITY, SCENARIO_CAMOUFLAGE)


@dataclass(frozen=True)
class DriftSettings:
    """Validated parameters for the controlled drift injector."""

    enabled: bool
    start_fraction: float
    end_fraction: float
    max_intensity: float
    random_seed: int
    shift_scale: float = DEFAULT_SHIFT_SCALE
    scenario: str | None = None
    camouflage_strength: float = DEFAULT_CAMOUFLAGE_STRENGTH
    camouflage_columns: tuple[str, ...] | None = None
    camouflage_top_k: int = DEFAULT_CAMOUFLAGE_TOP_K
    camouflage_categorical_top_k: int = DEFAULT_CAMOUFLAGE_CATEGORICAL_TOP_K

    def __post_init__(self) -> None:
        if not 0.0 <= self.start_fraction < self.end_fraction <= 1.0:
            raise ValueError("start_fraction and end_fraction must satisfy 0 <= start < end <= 1.")
        if not 0.0 <= self.max_intensity <= 1.0:
            raise ValueError("max_intensity must be between zero and one.")
        if self.shift_scale < 0:
            raise ValueError("shift_scale must be nonnegative.")
        if self.scenario is not None:
            if self.scenario not in SCENARIOS:
                raise ValueError(
                    f"Unknown drift scenario '{self.scenario}'. Expected one of {SCENARIOS}."
                )
            if (self.scenario == SCENARIO_NONE) == self.enabled:
                raise ValueError(
                    f"drift.enabled={self.enabled} contradicts drift.scenario='{self.scenario}'."
                )
        if not 0.0 <= self.camouflage_strength <= 1.0:
            raise ValueError("camouflage strength must be between zero and one.")
        if self.camouflage_top_k <= 0:
            raise ValueError("camouflage top_k must be greater than zero.")
        if self.camouflage_categorical_top_k < 0:
            raise ValueError("camouflage categorical_top_k must be nonnegative.")
        if self.camouflage_columns is not None and not self.camouflage_columns:
            raise ValueError("camouflage columns must not be empty when given.")

    @property
    def mode(self) -> str:
        """Effective scenario: explicit ``scenario``, else derived from ``enabled``."""

        if self.scenario is not None:
            return self.scenario
        return SCENARIO_SEPARABILITY if self.enabled else SCENARIO_NONE

    @classmethod
    def from_config(cls, drift: dict) -> DriftSettings:
        scenario = drift.get("scenario")
        if scenario is not None:
            scenario = str(scenario)
        if scenario is not None:
            # An explicit scenario wins over a (possibly inherited) "enabled" flag, so
            # {"scenario": "none"} or {"scenario": "camouflage"} can override any base config.
            enabled = scenario != SCENARIO_NONE
        elif "enabled" in drift:
            enabled = bool(drift["enabled"])
        else:
            raise KeyError("drift must define 'enabled' or 'scenario'.")

        camouflage = drift.get("camouflage") or {}
        columns = camouflage.get("columns")
        return cls(
            enabled=enabled,
            start_fraction=float(drift["start_fraction"]),
            end_fraction=float(drift["end_fraction"]),
            max_intensity=float(drift["max_intensity"]),
            random_seed=int(drift["random_seed"]),
            scenario=scenario,
            camouflage_strength=float(camouflage.get("strength", DEFAULT_CAMOUFLAGE_STRENGTH)),
            camouflage_columns=None if columns is None else tuple(str(c) for c in columns),
            camouflage_top_k=int(camouflage.get("top_k", DEFAULT_CAMOUFLAGE_TOP_K)),
            camouflage_categorical_top_k=int(
                camouflage.get("categorical_top_k", DEFAULT_CAMOUFLAGE_CATEGORICAL_TOP_K)
            ),
        )


@dataclass(frozen=True)
class DriftResult:
    """Outcome of injecting drift into one batch."""

    features: pd.DataFrame
    phase: str
    intensity: float
    affected_rows: int
    affected_columns: tuple[str, ...]
    scenario: str = SCENARIO_NONE


def drift_phase(progress: float, settings: DriftSettings) -> str:
    """Return the stream phase for ``progress`` under ``settings``'s schedule.

    With drift disabled (the no-drift control) the whole stream stays in
    ``PHASE_PRE_DRIFT``, so alarms are never labelled as occurring during drift.
    """

    if not settings.enabled:
        return PHASE_PRE_DRIFT
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
        self._camouflage_selected: tuple[str, ...] | None = None
        self._camouflage_categorical: tuple[str, ...] = ()

    @property
    def camouflage_columns(self) -> tuple[str, ...] | None:
        """Numeric columns camouflage acts on, once chosen (``None`` before a usable batch)."""

        return self._camouflage_selected

    @property
    def camouflage_categorical_columns(self) -> tuple[str, ...]:
        """Categorical columns camouflage resamples, once chosen."""

        return self._camouflage_categorical

    def _columns_for(self, features: pd.DataFrame) -> tuple[str, ...]:
        if self._explicit_columns is not None:
            return self._explicit_columns
        return tuple(features.select_dtypes(include="number").columns)

    def _direction(self, column: str) -> float:
        if column not in self._directions:
            self._directions[column] = float(self._direction_rng.choice([-1.0, 1.0]))
        return self._directions[column]

    def _select_camouflage_columns(self, features: pd.DataFrame, labels: np.ndarray) -> None:
        """Freeze the camouflaged columns using the first batch with both classes."""

        if self._camouflage_selected is not None:
            return
        fraud = labels == 1
        if fraud.sum() < MIN_CLASS_ROWS or (~fraud).sum() < MIN_CLASS_ROWS:
            return

        numeric = [str(c) for c in self._columns_for(features)]
        categorical = categorical_columns(features)
        wanted = self.settings.camouflage_columns
        if wanted is not None:
            missing = [c for c in wanted if c not in numeric and c not in categorical]
            if missing:
                raise ValueError(
                    f"camouflage columns are missing or not numeric/categorical: "
                    f"{', '.join(missing)}"
                )
            self._camouflage_selected = tuple(c for c in wanted if c in numeric)
            self._camouflage_categorical = tuple(c for c in wanted if c in categorical)
            return

        effect = standardized_mean_difference(features, labels, numeric)
        ranked = sorted(
            (c for c in numeric if np.isfinite(effect[c])),
            key=lambda c: -abs(effect[c]),  # stable sort keeps frame order on ties
        )
        self._camouflage_selected = tuple(ranked[: self.settings.camouflage_top_k])

        top_categorical = self.settings.camouflage_categorical_top_k
        if top_categorical and categorical:
            distance = total_variation_distance(features, labels, categorical)
            ranked_categorical = sorted(
                (c for c in categorical if np.isfinite(distance[c]) and distance[c] > 0.0),
                key=lambda c: -distance[c],
            )
            self._camouflage_categorical = tuple(ranked_categorical[:top_categorical])

    def apply(self, features: pd.DataFrame, labels: np.ndarray, progress: float) -> DriftResult:
        """Return a drifted copy of ``features`` for a batch at ``progress`` through the stream."""

        if not 0.0 < progress <= 1.0:
            raise ValueError("progress must be in the interval (0, 1].")

        mode = self.settings.mode
        phase = drift_phase(progress, self.settings)
        intensity = drift_intensity(progress, self.settings)
        labels = np.asarray(labels)
        fraud_mask = labels == 1

        if mode == SCENARIO_CAMOUFLAGE:
            self._select_camouflage_columns(features, labels)

        if intensity <= 0.0 or not fraud_mask.any():
            return DriftResult(
                features=features.copy(deep=True),
                phase=phase,
                intensity=intensity,
                affected_rows=0,
                affected_columns=(),
                scenario=mode,
            )

        drifted = features.copy(deep=True)
        if mode == SCENARIO_CAMOUFLAGE:
            shifted_columns = self._apply_camouflage(drifted, labels, intensity, progress)
        else:
            shifted_columns = self._apply_separability(drifted, fraud_mask, intensity)

        affected_rows = int(fraud_mask.sum()) if shifted_columns else 0

        return DriftResult(
            features=drifted,
            phase=phase,
            intensity=intensity,
            affected_rows=affected_rows,
            affected_columns=tuple(shifted_columns),
            scenario=mode,
        )

    def _apply_separability(
        self, drifted: pd.DataFrame, fraud_mask: np.ndarray, intensity: float
    ) -> list[str]:
        shifted_columns: list[str] = []

        for column in self._columns_for(drifted):
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

        return shifted_columns

    def _apply_camouflage(
        self, drifted: pd.DataFrame, labels: np.ndarray, intensity: float, progress: float
    ) -> list[str]:
        columns = self._camouflage_selected or ()
        fraud_mask = labels == 1
        legit_mask = ~fraud_mask
        if not (columns or self._camouflage_categorical) or not legit_mask.any():
            return []

        strength = min(intensity * self.settings.camouflage_strength, 1.0)
        shifted_columns: list[str] = []

        for column in columns:
            values = drifted[column].to_numpy(dtype=np.float64, na_value=np.nan)
            fraud_values = values[fraud_mask]
            legit_values = values[legit_mask]
            fraud_mean, fraud_std = np.nanmean(fraud_values), np.nanstd(fraud_values)
            legit_mean, legit_std = np.nanmean(legit_values), np.nanstd(legit_values)
            if not (np.isfinite(fraud_std) and fraud_std > 0.0):
                continue
            if not (np.isfinite(legit_mean) and np.isfinite(legit_std)):
                continue

            matched = legit_mean + (fraud_values - fraud_mean) * (legit_std / fraud_std)
            moved = (1.0 - strength) * fraud_values + strength * matched

            original_dtype = drifted[column].dtype
            if original_dtype.kind != "f":
                drifted[column] = drifted[column].astype("float64")
                original_dtype = drifted[column].dtype
            drifted.loc[fraud_mask, column] = moved.astype(original_dtype)
            shifted_columns.append(column)

        shifted_columns.extend(
            self._camouflage_categories(drifted, fraud_mask, legit_mask, strength, progress)
        )
        return shifted_columns

    def _camouflage_categories(
        self,
        drifted: pd.DataFrame,
        fraud_mask: np.ndarray,
        legit_mask: np.ndarray,
        strength: float,
        progress: float,
    ) -> list[str]:
        """Replace a ``strength`` share of fraud categories with legitimate-row draws.

        The generator is keyed by the configured seed and the batch's stream
        progress, so a batch is reproduced exactly regardless of which other
        batches were processed.
        """

        if not self._camouflage_categorical or strength <= 0.0:
            return []

        rng = np.random.default_rng((self.settings.random_seed, round(progress * 1_000_000_000)))
        fraud_positions = np.flatnonzero(fraud_mask)
        legit_positions = np.flatnonzero(legit_mask)
        changed: list[str] = []

        for column in self._camouflage_categorical:
            replace = rng.random(len(fraud_positions)) < strength
            donors = rng.integers(0, len(legit_positions), size=len(fraud_positions))
            targets = fraud_positions[replace]
            if not len(targets):
                continue
            values = drifted[column].to_numpy(copy=True)
            donor_values = drifted[column].to_numpy()[legit_positions[donors[replace]]]
            if isinstance(drifted[column].dtype, pd.CategoricalDtype):
                drifted.loc[drifted.index[targets], column] = donor_values
            else:
                values[targets] = donor_values
                drifted[column] = values
            changed.append(column)

        return changed
