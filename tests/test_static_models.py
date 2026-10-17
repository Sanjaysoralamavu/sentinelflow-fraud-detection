from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from sentinelflow.models.static import StaticModelSuite, positive_class_weights


def synthetic_data(
    rows: int = 240,
    columns: int = 8,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Create reproducible, imbalanced binary classification data."""

    generator = np.random.default_rng(543)
    features = generator.normal(size=(rows, columns)).astype(np.float32)

    fraud_score = (
        1.5 * features[:, 0]
        - 0.8 * features[:, 1]
        + 0.5 * features[:, 2]
        + generator.normal(scale=0.5, size=rows)
    )

    threshold = np.quantile(fraud_score, 0.85)
    labels = (fraud_score >= threshold).astype(np.int8)

    return sparse.csr_matrix(features), labels


def make_suite(*, random_forest_enabled: bool = True) -> StaticModelSuite:
    return StaticModelSuite(
        seed=543,
        positive_weight_cap=10,
        rf_enabled=random_forest_enabled,
        rf_training_rows=160,
        rf_n_estimators=12,
        rf_max_depth=6,
        rf_min_samples_leaf=2,
    )


def test_static_models_train_and_predict_probabilities() -> None:
    features, labels = synthetic_data()
    suite = make_suite()

    suite.fit(features[:180], labels[:180])
    probabilities = suite.predict_probabilities(features[180:])

    assert suite.is_fitted
    assert suite.names == ["static_logistic", "static_random_forest"]
    assert set(probabilities) == {"static_logistic", "static_random_forest"}

    for values in probabilities.values():
        assert values.shape == (60,)
        assert values.dtype == np.float32
        assert np.all(values >= 0)
        assert np.all(values <= 1)


def test_static_models_cannot_predict_before_training() -> None:
    features, _ = synthetic_data()
    suite = make_suite()

    with pytest.raises(RuntimeError, match="must be fitted"):
        suite.predict_probabilities(features)


def test_static_models_are_frozen_after_warmup() -> None:
    features, labels = synthetic_data()
    suite = make_suite()

    suite.fit(features, labels)

    with pytest.raises(RuntimeError, match="frozen"):
        suite.fit(features, labels)


def test_random_forest_can_be_disabled() -> None:
    features, labels = synthetic_data()
    suite = make_suite(random_forest_enabled=False)

    suite.fit(features[:180], labels[:180])
    probabilities = suite.predict_probabilities(features[180:])

    assert suite.names == ["static_logistic"]
    assert set(probabilities) == {"static_logistic"}


def test_static_models_are_reproducible() -> None:
    features, labels = synthetic_data()
    first_suite = make_suite()
    second_suite = make_suite()

    first_suite.fit(features[:180], labels[:180])
    second_suite.fit(features[:180], labels[:180])

    first_probabilities = first_suite.predict_probabilities(features[180:])
    second_probabilities = second_suite.predict_probabilities(features[180:])

    for model_name in first_probabilities:
        np.testing.assert_allclose(
            first_probabilities[model_name],
            second_probabilities[model_name],
        )


def test_positive_class_weight_is_capped() -> None:
    labels = np.array([0] * 90 + [1] * 10, dtype=np.int8)

    weights = positive_class_weights(labels, weight_cap=5)

    assert np.all(weights[labels == 0] == 1)
    assert np.all(weights[labels == 1] == 5)


def test_static_models_are_saved(tmp_path: Path) -> None:
    features, labels = synthetic_data()
    suite = make_suite()

    suite.fit(features, labels)
    suite.save(tmp_path)

    assert (tmp_path / "static_logistic.joblib").is_file()
    assert (tmp_path / "static_random_forest.joblib").is_file()
