"""Leakage-safe preprocessing fitted once on the warm-up window (work package 2).

``FraudPreprocessor`` decides each column's role from the warm-up frame, fits a
scikit-learn ``ColumnTransformer`` on those rows only, and then freezes. Every
later batch, drifted or not, goes through ``transform`` with the warm-up
statistics; nothing is ever refitted.

Column roles:

- numeric (int, float, bool): median imputation, then standard scaling
- low-cardinality categorical (string/category): constant imputation, then
  one-hot encoding that ignores categories unseen during warm-up
- high-cardinality categorical, datetime, and anything else: dropped, with the
  reason recorded in the :class:`FeatureManifest`

The output is a dense ``float32`` matrix by default, because every downstream
model (SGD, Random Forest, the adaptive learners) accepts it and the feature
width is small. Set ``sparse_output=True`` if one-hot width grows large.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from sentinelflow.data.loader import DEFAULT_IDENTIFIER_COLUMNS, configured_drop_columns

DEFAULT_MAX_CATEGORIES = 100
MISSING_CATEGORY = "__missing__"


class LeakageError(ValueError):
    """Raised when a target or identifier column reaches the preprocessor."""


class NotFittedError(RuntimeError):
    """Raised when transform is called before the warm-up fit."""


@dataclass
class FeatureManifest:
    """Which columns became model features, and why the others did not."""

    input_columns: list[str]
    numeric: list[str]
    categorical: list[str]
    dropped: dict[str, str] = field(default_factory=dict)
    category_counts: dict[str, int] = field(default_factory=dict)
    output_features: list[str] = field(default_factory=list)
    fit_rows: int = 0

    @property
    def included(self) -> list[str]:
        return [*self.numeric, *self.categorical]

    @property
    def n_output_features(self) -> int:
        return len(self.output_features)

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["included"] = self.included
        record["n_output_features"] = self.n_output_features
        return record

    def write_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return output


def infer_column_roles(
    frame: pd.DataFrame,
    *,
    max_categories: int = DEFAULT_MAX_CATEGORIES,
    exclude: tuple[str, ...] = (),
) -> FeatureManifest:
    """Classify each column of the warm-up frame as numeric, categorical, or dropped."""

    numeric: list[str] = []
    categorical: list[str] = []
    dropped: dict[str, str] = {}
    category_counts: dict[str, int] = {}

    for column in frame.columns:
        series = frame[column]
        if column in exclude:
            dropped[column] = "excluded by configuration"
        elif pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            numeric.append(column)
        elif pd.api.types.is_datetime64_any_dtype(series) or isinstance(
            series.dtype, pd.PeriodDtype
        ):
            dropped[column] = "datetime: not used as a raw feature"
        elif (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            distinct = int(series.nunique(dropna=True))
            category_counts[column] = distinct
            if distinct > max_categories:
                dropped[column] = f"high cardinality: {distinct} > {max_categories} categories"
            elif distinct <= 1:
                dropped[column] = "constant in warm-up window"
            else:
                categorical.append(column)
        else:
            dropped[column] = f"unsupported dtype {series.dtype}"

    for column in list(numeric):
        if frame[column].nunique(dropna=True) <= 1:
            numeric.remove(column)
            dropped[column] = "constant in warm-up window"

    return FeatureManifest(
        input_columns=list(frame.columns),
        numeric=numeric,
        categorical=categorical,
        dropped=dropped,
        category_counts=category_counts,
    )


def _to_text(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast categoricals to plain strings with an explicit missing marker."""

    return frame.astype("object").where(frame.notna(), MISSING_CATEGORY).astype(str)


def _to_float(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.astype("float64")


def _passthrough_names(function: Any) -> FunctionTransformer:
    """A stateless step that keeps column names so feature names stay readable."""

    return FunctionTransformer(function, feature_names_out="one-to-one", validate=False)


class FraudPreprocessor:
    """Warm-up-fitted, then frozen, feature transformer."""

    def __init__(
        self,
        *,
        target_column: str = "is_fraud",
        identifier_columns: tuple[str, ...] = DEFAULT_IDENTIFIER_COLUMNS,
        exclude_columns: tuple[str, ...] = (),
        max_categories: int = DEFAULT_MAX_CATEGORIES,
        sparse_output: bool = False,
    ) -> None:
        if max_categories < 2:
            raise ValueError("max_categories must be at least 2.")
        self.target_column = target_column
        self.identifier_columns = tuple(identifier_columns)
        self.exclude_columns = tuple(exclude_columns)
        self.max_categories = int(max_categories)
        self.sparse_output = bool(sparse_output)
        self.manifest: FeatureManifest | None = None
        self._transformer: ColumnTransformer | None = None

    @classmethod
    def from_config(cls, config: Any) -> FraudPreprocessor:
        """Build from ``ProjectConfig``; reads the optional ``data.preprocessing`` block."""

        data = config.section("data")
        options = data.get("preprocessing") or {}
        return cls(
            target_column=config.target_column,
            identifier_columns=configured_drop_columns(data),
            exclude_columns=tuple(options.get("exclude_columns", ())),
            max_categories=int(options.get("max_categories", DEFAULT_MAX_CATEGORIES)),
            sparse_output=bool(options.get("sparse_output", False)),
        )

    @property
    def is_fitted(self) -> bool:
        return self._transformer is not None

    def fit(self, warmup_features: pd.DataFrame) -> Self:
        """Fit on the warm-up frame once. A second call raises."""

        if self.is_fitted:
            raise RuntimeError("The preprocessor is frozen after warm-up and cannot be refitted.")
        self._check_leakage(warmup_features)
        if warmup_features.empty:
            raise ValueError("Cannot fit the preprocessor on an empty warm-up frame.")

        manifest = infer_column_roles(
            warmup_features,
            max_categories=self.max_categories,
            exclude=self.exclude_columns,
        )
        if not manifest.included:
            raise ValueError("No usable feature columns remain after type inference.")

        transformers = []
        if manifest.numeric:
            transformers.append(
                (
                    "numeric",
                    Pipeline(
                        [
                            ("as_float", _passthrough_names(_to_float)),
                            ("impute", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    manifest.numeric,
                )
            )
        if manifest.categorical:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            ("as_text", _passthrough_names(_to_text)),
                            (
                                "one_hot",
                                OneHotEncoder(
                                    handle_unknown="ignore",
                                    sparse_output=True,
                                    dtype=np.float32,
                                ),
                            ),
                        ]
                    ),
                    manifest.categorical,
                )
            )

        transformer = ColumnTransformer(
            transformers,
            remainder="drop",
            sparse_threshold=1.0 if self.sparse_output else 0.0,
            verbose_feature_names_out=True,
        )
        transformer.fit(warmup_features)

        manifest.output_features = [str(name) for name in transformer.get_feature_names_out()]
        manifest.fit_rows = len(warmup_features)
        self.manifest = manifest
        self._transformer = transformer
        return self

    def transform(self, features: pd.DataFrame) -> np.ndarray | sparse.csr_matrix:
        """Transform any later batch with the frozen warm-up statistics."""

        if self._transformer is None or self.manifest is None:
            raise NotFittedError("Fit the preprocessor on the warm-up window first.")
        self._check_leakage(features)

        missing = [column for column in self.manifest.included if column not in features.columns]
        if missing:
            raise KeyError(f"Batch is missing fitted feature columns: {', '.join(missing)}")

        output = self._transformer.transform(features)
        if self.sparse_output:
            return sparse.csr_matrix(output, dtype=np.float32)
        if sparse.issparse(output):
            output = output.toarray()
        return np.asarray(output, dtype=np.float32)

    def fit_transform(self, warmup_features: pd.DataFrame) -> np.ndarray | sparse.csr_matrix:
        return self.fit(warmup_features).transform(warmup_features)

    def _check_leakage(self, features: pd.DataFrame) -> None:
        forbidden = [self.target_column, *self.identifier_columns]
        present = [column for column in forbidden if column in features.columns]
        if present:
            raise LeakageError(
                "Target or identifier columns reached the preprocessor: " + ", ".join(present)
            )
