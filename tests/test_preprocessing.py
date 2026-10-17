from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from sentinelflow.data.loader import load_warmup_and_stream
from sentinelflow.data.preprocessing import (
    FraudPreprocessor,
    LeakageError,
    NotFittedError,
    infer_column_roles,
)


@pytest.fixture
def fitted(data_settings):
    warmup, evaluation = load_warmup_and_stream(data_settings)
    preprocessor = FraudPreprocessor(max_categories=20).fit(warmup.features)
    return preprocessor, warmup, list(evaluation)


def test_column_roles(fitted) -> None:
    manifest = fitted[0].manifest
    assert set(manifest.numeric) == {"amount", "hour", "is_international"}
    assert set(manifest.categorical) == {"merchant_category", "device_type", "card_present_cnp"}
    assert "merchant_name" in manifest.dropped
    assert "high cardinality" in manifest.dropped["merchant_name"]


def test_output_is_dense_float32_and_finite(fitted) -> None:
    preprocessor, warmup, evaluation = fitted
    matrix = preprocessor.transform(evaluation[0].features)
    assert isinstance(matrix, np.ndarray)
    assert matrix.dtype == np.float32
    assert matrix.shape == (evaluation[0].row_count, preprocessor.manifest.n_output_features)
    assert np.isfinite(matrix).all(), "imputation must remove every NaN"


def test_width_is_stable_across_batches(fitted) -> None:
    preprocessor, _, evaluation = fitted
    widths = {preprocessor.transform(chunk.features).shape[1] for chunk in evaluation}
    assert widths == {preprocessor.manifest.n_output_features}


def test_scaling_statistics_come_from_warmup_only(fitted) -> None:
    preprocessor, warmup, _ = fitted
    scaler = preprocessor._transformer.named_transformers_["numeric"].named_steps["scale"]
    expected = warmup.features["amount"].astype(float)
    expected = expected.fillna(expected.median()).mean()
    index = preprocessor.manifest.numeric.index("amount")
    assert scaler.mean_[index] == pytest.approx(expected)


def test_frozen_after_fit(fitted) -> None:
    preprocessor, warmup, _ = fitted
    with pytest.raises(RuntimeError, match="frozen"):
        preprocessor.fit(warmup.features)


def test_drifted_batch_does_not_change_statistics(fitted) -> None:
    preprocessor, _, evaluation = fitted
    scaler = preprocessor._transformer.named_transformers_["numeric"].named_steps["scale"]
    before = scaler.mean_.copy()
    drifted = evaluation[0].features.copy()
    drifted["amount"] = drifted["amount"] * 50
    preprocessor.transform(drifted)
    np.testing.assert_array_equal(scaler.mean_, before)


def test_unseen_category_is_ignored(fitted) -> None:
    preprocessor, _, evaluation = fitted
    batch = evaluation[0].features.copy()
    batch["device_type"] = "smart_fridge"
    matrix = preprocessor.transform(batch)
    names = preprocessor.manifest.output_features
    device = [i for i, name in enumerate(names) if name.startswith("categorical__device_type_")]
    assert not matrix[:, device].any()


@pytest.mark.parametrize("column", ["is_fraud", "customer_id", "transaction_id"])
def test_leakage_columns_rejected(fitted, column) -> None:
    preprocessor, _, evaluation = fitted
    batch = evaluation[0].features.copy()
    batch[column] = 1
    with pytest.raises(LeakageError, match=column):
        preprocessor.transform(batch)


def test_leakage_rejected_at_fit() -> None:
    frame = pd.DataFrame({"amount": [1.0, 2.0, 3.0], "is_fraud": [0, 1, 0]})
    with pytest.raises(LeakageError):
        FraudPreprocessor().fit(frame)


def test_missing_feature_column_rejected(fitted) -> None:
    preprocessor, _, evaluation = fitted
    with pytest.raises(KeyError, match="amount"):
        preprocessor.transform(evaluation[0].features.drop(columns=["amount"]))


def test_transform_before_fit() -> None:
    with pytest.raises(NotFittedError):
        FraudPreprocessor().transform(pd.DataFrame({"amount": [1.0]}))


def test_sparse_output_option(data_settings) -> None:
    warmup, _ = load_warmup_and_stream(data_settings)
    matrix = FraudPreprocessor(max_categories=20, sparse_output=True).fit_transform(warmup.features)
    assert sparse.isspmatrix_csr(matrix)
    assert matrix.dtype == np.float32


def test_excluded_and_constant_columns_dropped() -> None:
    frame = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0], "same": [5, 5, 5], "risk_signal": [0.1, 0.9, 0.2], "c": list("xyx")}
    )
    manifest = infer_column_roles(frame, exclude=("risk_signal",))
    assert manifest.numeric == ["a"]
    assert manifest.categorical == ["c"]
    assert manifest.dropped["risk_signal"] == "excluded by configuration"
    assert "constant" in manifest.dropped["same"]


def test_manifest_written_as_json(fitted, tmp_path) -> None:
    path = fitted[0].manifest.write_json(tmp_path / "feature_manifest.json")
    record = json.loads(path.read_text())
    assert record["n_output_features"] == len(record["output_features"])
    assert record["fit_rows"] == 300
    assert "is_fraud" not in record["input_columns"]
