"""Tests for the unified static and adaptive model coordinator."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from sentinelflow.config import ProjectConfig, load_config
from sentinelflow.models.common import DelayedBatch
from sentinelflow.suite import ModelSuite

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_config() -> ProjectConfig:
    """Create a fast test configuration from the shared full configuration."""

    base_config = load_config(PROJECT_ROOT / "configs" / "full.yaml")
    raw = deepcopy(base_config.raw)

    raw["models"]["random_forest"].update(
        {
            "enabled": True,
            "training_rows": 600,
            "n_estimators": 8,
            "max_depth": 5,
            "min_samples_leaf": 2,
        }
    )
    raw["models"]["adaptive"].update(
        {
            "detector_sample_size": 100,
            "recent_window_rows": 500,
            "ddm_warm_start": 10,
            "detector_grace_batches": 50,
            "detector_cooldown_batches": 3,
        }
    )
    raw["models"]["online_ensemble"].update(
        {
            "enabled": True,
            "n_estimators": 3,
        }
    )

    return ProjectConfig(raw=raw, root=base_config.root)


def make_data(
    seed: int = 1,
    rows: int = 800,
) -> tuple[np.ndarray, np.ndarray]:
    """Create reproducible binary fraud-classification data."""

    generator = np.random.default_rng(seed)
    features = generator.normal(size=(rows, 6)).astype(np.float32)
    signal = features[:, 0] + (0.5 * features[:, 1])
    labels = (signal > 1.0).astype(np.int8)

    assert set(np.unique(labels)) == {0, 1}
    return features, labels


def test_from_config_builds_all_enabled_models() -> None:
    suite = ModelSuite.from_config(make_config())

    assert suite.names == [
        "static_logistic",
        "static_random_forest",
        "adaptive_none",
        "adaptive_adwin",
        "adaptive_ddm",
        "online_ensemble",
    ]
    assert not suite.is_warmed_up


def test_suite_refuses_prediction_before_warm_up() -> None:
    suite = ModelSuite.from_config(make_config())
    features, _ = make_data(rows=20)

    with pytest.raises(RuntimeError, match="warm-up"):
        suite.predict_probabilities(features)


def test_warm_up_and_prediction_cover_every_model() -> None:
    suite = ModelSuite.from_config(make_config())
    features, labels = make_data()

    suite.warm_up(features, labels)
    probabilities = suite.predict_probabilities(features[:100])

    assert suite.is_warmed_up
    assert list(probabilities) == suite.names

    for scores in probabilities.values():
        assert scores.shape == (100,)
        assert scores.dtype == np.float32
        assert np.all(np.isfinite(scores))
        assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_suite_cannot_repeat_warm_up() -> None:
    suite = ModelSuite.from_config(make_config())
    features, labels = make_data()

    suite.warm_up(features, labels)

    with pytest.raises(RuntimeError, match="already completed"):
        suite.warm_up(features, labels)


def test_delayed_learning_does_not_update_static_models() -> None:
    suite = ModelSuite.from_config(make_config())
    warmup_features, warmup_labels = make_data(seed=2)
    stream_features, stream_labels = make_data(seed=3, rows=200)

    suite.warm_up(warmup_features, warmup_labels)

    predictions = suite.predict_probabilities(stream_features)
    static_before = {name: predictions[name].copy() for name in suite.static_models.names}

    delayed_batch = DelayedBatch(
        source_batch=0,
        observed_batch=2,
        features=stream_features,
        labels=1 - stream_labels,
        predictions=predictions,
        phase="pre_drift",
    )

    events = suite.learn(delayed_batch)
    static_after = suite.predict_probabilities(stream_features)

    assert isinstance(events, list)

    for name, expected in static_before.items():
        np.testing.assert_array_equal(static_after[name], expected)
