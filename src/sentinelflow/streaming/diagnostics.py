"""Class-separation diagnostics for the controlled-drift scenarios.

The separation measure is the absolute standardized mean difference between
fraud and legitimate rows (Cohen's ``d`` with a pooled standard deviation) per
numeric feature. It is model-free, so it shows whether a drift scenario made
fraud easier (larger ``|d|``) or harder (smaller ``|d|``) to tell apart from
legitimate traffic before any classifier is involved.

These functions read labels only to group rows for measurement. They never
add the label to a feature frame.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_CLASS_ROWS = 2


def numeric_columns(features: pd.DataFrame) -> list[str]:
    """Numeric (non-boolean) feature columns, in frame order."""

    return [str(column) for column in features.select_dtypes(include="number").columns]


def categorical_columns(features: pd.DataFrame) -> list[str]:
    """Non-numeric, non-datetime feature columns (strings, categories, booleans)."""

    selected = features.select_dtypes(exclude=["number", "datetime", "datetimetz", "timedelta"])
    return [str(column) for column in selected.columns]


def total_variation_distance(
    features: pd.DataFrame,
    labels: np.ndarray,
    columns: Sequence[str] | None = None,
) -> dict[str, float]:
    """Total variation distance between fraud and legitimate category frequencies.

    ``0`` means identical category mix, ``1`` means disjoint. Missing values
    count as their own category. Classes with fewer than two rows give ``nan``.
    """

    labels = np.asarray(labels)
    names = list(columns) if columns is not None else categorical_columns(features)
    fraud = labels == 1
    legit = ~fraud

    if fraud.sum() < MIN_CLASS_ROWS or legit.sum() < MIN_CLASS_ROWS:
        return {name: float("nan") for name in names}

    result: dict[str, float] = {}
    for name in names:
        column = features[name]
        fraud_freq = column[fraud].value_counts(normalize=True, dropna=False)
        legit_freq = column[legit].value_counts(normalize=True, dropna=False)
        aligned = pd.concat([fraud_freq, legit_freq], axis=1, keys=["f", "l"]).fillna(0.0)
        result[name] = float(0.5 * (aligned["f"] - aligned["l"]).abs().sum())
    return result


def standardized_mean_difference(
    features: pd.DataFrame,
    labels: np.ndarray,
    columns: Sequence[str] | None = None,
) -> dict[str, float]:
    """Signed pooled-standard-deviation mean difference (fraud minus legitimate).

    Columns with no variation, or classes with fewer than two rows, yield
    ``nan`` so callers can tell "not measurable" from "zero separation".
    """

    labels = np.asarray(labels)
    names = list(columns) if columns is not None else numeric_columns(features)
    fraud = labels == 1
    legit = ~fraud

    if fraud.sum() < MIN_CLASS_ROWS or legit.sum() < MIN_CLASS_ROWS:
        return {name: float("nan") for name in names}

    result: dict[str, float] = {}
    for name in names:
        values = features[name].to_numpy(dtype=np.float64, na_value=np.nan)
        fraud_values = values[fraud]
        legit_values = values[legit]
        pooled = np.sqrt((np.nanvar(fraud_values) + np.nanvar(legit_values)) / 2.0)
        if not np.isfinite(pooled) or pooled == 0.0:
            result[name] = float("nan")
        else:
            result[name] = float((np.nanmean(fraud_values) - np.nanmean(legit_values)) / pooled)
    return result


def class_separation(
    features: pd.DataFrame,
    labels: np.ndarray,
    columns: Sequence[str] | None = None,
) -> float:
    """Mean absolute standardized mean difference over ``columns``.

    Returns ``nan`` when no column is measurable.
    """

    values = np.array(list(standardized_mean_difference(features, labels, columns).values()))
    values = np.abs(values[np.isfinite(values)])
    return float(values.mean()) if values.size else float("nan")


@dataclass(frozen=True)
class PhaseSeparation:
    """Separation of the rows a stream produced in one drift phase."""

    phase: str
    rows: int
    fraud_rows: int
    separation: float
    per_column: dict[str, float]
    categorical_separation: float = float("nan")
    per_categorical: dict[str, float] | None = None


def separation_by_phase(
    batches: Iterable,
    columns: Sequence[str] | None = None,
) -> dict[str, PhaseSeparation]:
    """Pool each phase's rows across batches and measure class separation.

    ``batches`` yields objects with ``phase``, ``features`` and ``labels``
    (``Batch``). Pooling matches how phase-level model metrics are computed.
    """

    frames: dict[str, list[pd.DataFrame]] = defaultdict(list)
    labels: dict[str, list[np.ndarray]] = defaultdict(list)
    order: list[str] = []

    for batch in batches:
        if batch.phase not in frames:
            order.append(batch.phase)
        frames[batch.phase].append(batch.features)
        labels[batch.phase].append(np.asarray(batch.labels))

    report: dict[str, PhaseSeparation] = {}
    for phase in order:
        pooled = pd.concat(frames[phase], ignore_index=True)
        pooled_labels = np.concatenate(labels[phase])
        per_column = standardized_mean_difference(pooled, pooled_labels, columns)
        finite = np.abs([v for v in per_column.values() if np.isfinite(v)])
        per_categorical = total_variation_distance(pooled, pooled_labels)
        finite_tvd = [v for v in per_categorical.values() if np.isfinite(v)]
        report[phase] = PhaseSeparation(
            phase=phase,
            rows=len(pooled_labels),
            fraud_rows=int(np.count_nonzero(pooled_labels)),
            separation=float(finite.mean()) if len(finite) else float("nan"),
            per_column=per_column,
            categorical_separation=float(np.mean(finite_tvd)) if finite_tvd else float("nan"),
            per_categorical=per_categorical,
        )
    return report
