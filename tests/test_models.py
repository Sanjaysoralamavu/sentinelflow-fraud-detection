"""Adaptive-model tests (work package 4).

The synthetic stream makes fraud depend on the first feature and flips that
relationship at ``flip_at`` to create a known, abrupt concept drift.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from sentinelflow.models.adaptive import (
    AdaptiveModel,
    AdaptiveSettings,
    build_adaptive_models,
)
from sentinelflow.models.common import (
    ALARM_CONFIRMED,
    ALARM_SUPPRESSED_GRACE,
    WARMUP_BATCH,
    DelayedBatch,
    RecentWindow,
    positive_class_weight,
)
from sentinelflow.models.online_ensemble import OnlineBaggingEnsemble

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_FEATURES = 6


def make_batch(rng: np.random.Generator, rows: int, flipped: bool = False):
    features = rng.normal(size=(rows, N_FEATURES))
    signal = -features[:, 0] if flipped else features[:, 0]
    labels = (signal > 1.0).astype(int)
    return features, labels


def small_settings(**overrides) -> AdaptiveSettings:
    values = {
        "alpha": 0.001,
        "threshold": 0.5,
        "positive_weight_cap": 10.0,
        "detector_sample_size": 300,
        "recent_window_rows": 1500,
        "adwin_delta": 0.002,
        "ddm_warm_start": 30,
        "detector_grace_batches": 2,
        "detector_cooldown_batches": 3,
    }
    values.update(overrides)
    return AdaptiveSettings(**values)


def run_stream(models: dict, batches: int, flip_at: int, delay: int = 0, rows: int = 500):
    """Score every batch with every model first, release labels ``delay`` batches later.

    Returns the delayed batches each model learned from and the events it logged.
    """

    rng = np.random.default_rng(7)
    warm_x, warm_y = make_batch(rng, 2000)
    for model in models.values():
        model.warm_up(warm_x, warm_y)

    pending: list[DelayedBatch] = []
    seen = {name: [] for name in models}
    events = {name: [] for name in models}

    for batch_id in range(batches):
        x, y = make_batch(rng, rows, flipped=batch_id >= flip_at)
        scores = {name: model.predict_proba(x) for name, model in models.items()}
        pending.append(
            DelayedBatch(
                source_batch=batch_id,
                observed_batch=batch_id + delay,
                features=x,
                labels=y,
                predictions=scores,
                phase="drifted" if batch_id >= flip_at else "pre_drift",
            )
        )
        due = [item for item in pending if item.observed_batch <= batch_id]
        pending = [item for item in pending if item.observed_batch > batch_id]
        for item in due:
            for name, model in models.items():
                seen[name].append(item)
                event = model.learn(item)
                if event is not None:
                    events[name].append(event)

    return seen, events


def adaptive(detector: str, **overrides) -> dict:
    model = AdaptiveModel(detector, small_settings(**overrides), seed=543)
    return {model.name: model}


def confirmed(events) -> list:
    return [event for event in events if event.confirmed]


# ----------------------------------------------------------------- shared helpers


def test_positive_class_weight_is_bounded() -> None:
    assert positive_class_weight(np.array([0] * 99 + [1]), cap=25.0) == 25.0
    assert positive_class_weight(np.array([0, 0, 1, 1]), cap=25.0) == 1.0
    assert positive_class_weight(np.zeros(5), cap=25.0) == 1.0


def test_recent_window_keeps_latest_rows_and_their_batches() -> None:
    window = RecentWindow(max_rows=5)
    for batch_id, start in enumerate((0, 3, 6)):
        rows = np.arange(start, start + 3)
        window.add(batch_id, rows.reshape(-1, 1), rows)

    features, labels = window.arrays()
    assert len(window) == 5
    assert labels.tolist() == [4, 5, 6, 7, 8]
    assert features.ravel().tolist() == labels.tolist()
    assert window.source_batches() == [1, 2]


def test_delayed_batch_rejects_release_before_scoring() -> None:
    with pytest.raises(ValueError):
        DelayedBatch(5, 4, np.zeros((2, 1)), np.zeros(2))


# ------------------------------------------------- identical batches, probabilities


def test_all_variants_receive_identical_due_batches() -> None:
    models = {
        **adaptive("none"),
        **adaptive("adwin"),
        **adaptive("ddm"),
        "online_ensemble": OnlineBaggingEnsemble(5, 0.001, 10.0, seed=543),
    }
    seen, _ = run_stream(models, batches=12, flip_at=6, delay=2)

    reference = seen["adaptive_none"]
    for name in models:
        assert seen[name] == reference, name


@pytest.mark.parametrize("detector", ["none", "adwin", "ddm"])
def test_adaptive_models_emit_one_probability_per_row(detector: str) -> None:
    rng = np.random.default_rng(0)
    model = AdaptiveModel(detector, small_settings(), seed=543)
    x, y = make_batch(rng, 1000)
    model.warm_up(x, y)

    scores = model.predict_proba(x[:250])

    assert scores.shape == (250,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_scoring_does_not_change_model_state() -> None:
    rng = np.random.default_rng(1)
    model = AdaptiveModel("adwin", small_settings(), seed=543)
    x, y = make_batch(rng, 1000)
    model.warm_up(x, y)

    np.testing.assert_array_equal(model.predict_proba(x), model.predict_proba(x))


def test_models_refuse_to_score_before_warm_up() -> None:
    with pytest.raises(RuntimeError):
        AdaptiveModel("none", small_settings(), seed=1).predict_proba(np.zeros((1, 2)))


# ----------------------------------------------------------------- delayed learning


def test_labels_are_learned_only_after_release() -> None:
    models = adaptive("none")
    seen, _ = run_stream(models, batches=10, flip_at=99, delay=3)

    batches = seen["adaptive_none"]
    assert [item.source_batch for item in batches] == list(range(7))
    assert all(item.observed_batch == item.source_batch + 3 for item in batches)


def test_learning_changes_the_model() -> None:
    rng = np.random.default_rng(2)
    model = AdaptiveModel("none", small_settings(), seed=543)
    x, y = make_batch(rng, 1000)
    model.warm_up(x, y)
    before = model.predict_proba(x)

    bx, by = make_batch(rng, 500, flipped=True)
    model.learn(DelayedBatch(0, 0, bx, by))

    assert not np.allclose(before, model.predict_proba(x))


def test_detectors_need_the_stored_scoring_predictions() -> None:
    rng = np.random.default_rng(3)
    model = AdaptiveModel("adwin", small_settings(), seed=543)
    x, y = make_batch(rng, 500)
    model.warm_up(x, y)

    with pytest.raises(KeyError):
        model.learn(DelayedBatch(0, 0, x, y, predictions={}))


# ----------------------------------------------------------------- detectors, resets


def test_detector_free_model_never_resets() -> None:
    models = adaptive("none")
    _, events = run_stream(models, batches=20, flip_at=8)

    assert events["adaptive_none"] == []
    assert models["adaptive_none"].reset_count == 0


def test_adwin_and_ddm_receive_the_same_error_stream_before_any_reset() -> None:
    adwin = AdaptiveModel("adwin", small_settings(detector_grace_batches=100), seed=543)
    ddm = AdaptiveModel("ddm", small_settings(detector_grace_batches=100), seed=543)
    seen, _ = run_stream({adwin.name: adwin, ddm.name: ddm}, batches=10, flip_at=5)

    for item in seen[adwin.name]:
        np.testing.assert_array_equal(item.predictions[adwin.name], item.predictions[ddm.name])


@pytest.mark.parametrize("detector", ["adwin", "ddm"])
def test_detectors_reset_after_concept_flip(detector: str) -> None:
    models = adaptive(detector)
    _, events = run_stream(models, batches=20, flip_at=8)
    resets = confirmed(events[f"adaptive_{detector}"])

    assert any(event.source_batch >= 8 for event in resets), f"{detector} missed the flip"
    assert models[f"adaptive_{detector}"].reset_count == len(resets)
    if detector == "adwin":
        # ADWIN is conservative. DDM may raise early alarms, which the
        # evaluation package classifies as false alarms.
        assert all(event.source_batch >= 8 for event in resets)


def test_alarm_log_records_detector_batches_state_and_recovery() -> None:
    models = adaptive("adwin")
    _, events = run_stream(models, batches=20, flip_at=8)
    event = confirmed(events["adaptive_adwin"])[0]

    assert event.model == "adaptive_adwin"
    assert event.detector == "adwin"
    assert event.alarm_type == "drift"
    assert event.alarm_state == ALARM_CONFIRMED
    assert event.phase == "drifted"
    assert event.reset_count == 1
    assert event.retrain_rows > 0
    assert event.fraud_weight is not None
    assert models["adaptive_adwin"].events == events["adaptive_adwin"]


def test_alarms_reproduce_with_fixed_seed() -> None:
    _, first = run_stream(adaptive("adwin"), 20, 8)
    _, second = run_stream(adaptive("adwin"), 20, 8)

    assert first == second


def test_grace_period_logs_but_suppresses_alarms() -> None:
    models = adaptive("ddm", detector_grace_batches=50)
    _, events = run_stream(models, batches=20, flip_at=0)
    logged = events["adaptive_ddm"]

    assert models["adaptive_ddm"].reset_count == 0
    assert all(event.alarm_state == ALARM_SUPPRESSED_GRACE for event in logged)


def test_cooldown_spaces_confirmed_resets() -> None:
    models = adaptive("ddm", detector_cooldown_batches=5)
    _, events = run_stream(models, batches=30, flip_at=8)
    resets = confirmed(events["adaptive_ddm"])

    gaps = np.diff([event.observed_batch for event in resets])
    assert np.all(gaps >= 5)


@pytest.mark.parametrize("delay", [0, 3])
def test_reset_retrains_only_on_released_bounded_window(delay: int) -> None:
    models = adaptive("adwin", recent_window_rows=800)
    _, events = run_stream(models, batches=24, flip_at=8, delay=delay)
    resets = confirmed(events["adaptive_adwin"])

    assert resets
    for event in resets:
        assert event.retrain_rows == 800
        assert event.retrain_last_batch == event.source_batch
        assert event.retrain_last_batch <= event.observed_batch - delay
        assert WARMUP_BATCH <= event.retrain_first_batch <= event.retrain_last_batch


def test_unknown_detector_is_rejected() -> None:
    with pytest.raises(ValueError):
        AdaptiveModel("page_hinkley", small_settings(), seed=543)


# ----------------------------------------------------------------- online ensemble


def test_online_ensemble_averages_five_independent_learners() -> None:
    rng = np.random.default_rng(4)
    ensemble = OnlineBaggingEnsemble(5, alpha=0.001, positive_weight_cap=10.0, seed=543)
    x, y = make_batch(rng, 2000)
    ensemble.warm_up(x, y)

    per_learner = np.array([learner.predict_proba(x[:100])[:, 1] for learner in ensemble.learners])

    assert len(ensemble.learners) == 5
    np.testing.assert_allclose(ensemble.predict_proba(x[:100]), per_learner.mean(axis=0))
    assert not np.allclose(per_learner[0], per_learner[1])


def test_online_ensemble_poisson_weights_are_reproducible() -> None:
    def scores() -> np.ndarray:
        rng = np.random.default_rng(5)
        ensemble = OnlineBaggingEnsemble(5, alpha=0.001, positive_weight_cap=10.0, seed=543)
        x, y = make_batch(rng, 2000)
        ensemble.warm_up(x, y)
        bx, by = make_batch(rng, 500)
        ensemble.learn(DelayedBatch(0, 0, bx, by))
        return ensemble.predict_proba(bx)

    np.testing.assert_array_equal(scores(), scores())


def test_online_ensemble_updates_on_released_labels() -> None:
    rng = np.random.default_rng(6)
    ensemble = OnlineBaggingEnsemble(5, alpha=0.001, positive_weight_cap=10.0, seed=543)
    x, y = make_batch(rng, 2000)
    ensemble.warm_up(x, y)
    before = ensemble.predict_proba(x)

    bx, by = make_batch(rng, 500, flipped=True)
    ensemble.learn(DelayedBatch(0, 0, bx, by))

    assert not np.allclose(before, ensemble.predict_proba(x))


# ----------------------------------------------------------------- configuration


def full_models_section() -> dict:
    with (PROJECT_ROOT / "configs" / "full.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)["models"]


def test_build_adaptive_models_from_full_config() -> None:
    models = build_adaptive_models(full_models_section(), seed=543)

    assert set(models) == {"adaptive_none", "adaptive_adwin", "adaptive_ddm", "online_ensemble"}
    assert models["online_ensemble"].n_estimators == 5
    assert models["adaptive_adwin"].settings.recent_window_rows == 200_000


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("threshold",), 1.5),
        (("positive_weight_cap",), 0.5),
        (("adaptive", "alpha"), 0),
        (("adaptive", "detectors"), ["none", "adwn"]),
        (("adaptive", "adwin_delta"), 2.0),
        (("adaptive", "detector_cooldown_batches"), -1),
        (("online_ensemble", "n_estimators"), 0),
    ],
)
def test_invalid_model_settings_are_rejected(path: tuple, value) -> None:
    section = full_models_section()
    target = section
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError):
        build_adaptive_models(section, seed=543)


# ----------------------------------------------------------------- sparse input


def test_models_accept_sparse_preprocessed_features() -> None:
    from scipy import sparse

    rng = np.random.default_rng(8)
    dense_x, y = make_batch(rng, 3000)
    x = sparse.csr_matrix(dense_x)
    models = {
        **adaptive("adwin", recent_window_rows=1000, detector_grace_batches=0),
        "online_ensemble": OnlineBaggingEnsemble(5, 0.001, 10.0, seed=543),
    }

    for model in models.values():
        model.warm_up(x[:2000], y[:2000])
        scores = model.predict_proba(x[2000:])
        assert scores.shape == (1000,)
        model.learn(DelayedBatch(0, 0, x[2000:], y[2000:], {model.name: scores}))

    features, labels = models["adaptive_adwin"]._window.arrays()
    assert sparse.issparse(features)
    assert features.shape[0] == len(labels) == 1000


# ----------------------------------------------------------------- held alarms


class ScriptedDetector:
    """Signals drift on the first update of the scripted source batches."""

    def __init__(self, schedule: set, clock: dict) -> None:
        self.schedule = schedule
        self.clock = clock
        self.drift_detected = False

    def update(self, _error: int) -> None:
        batch = self.clock["batch"]
        self.drift_detected = batch in self.schedule
        self.schedule.discard(batch)


def test_alarm_inside_cooldown_is_held_and_acted_on_after(monkeypatch) -> None:
    import sentinelflow.models.adaptive as adaptive_module

    schedule, clock = {3, 5}, {"batch": -1}
    monkeypatch.setattr(
        adaptive_module, "build_detector", lambda *_: ScriptedDetector(schedule, clock)
    )
    model = AdaptiveModel(
        "adwin", small_settings(detector_grace_batches=0, detector_cooldown_batches=4), 1
    )
    rng = np.random.default_rng(9)
    model.warm_up(*make_batch(rng, 1000))

    states = []
    for batch_id in range(12):
        x, y = make_batch(rng, 200)
        clock["batch"] = batch_id
        event = model.learn(
            DelayedBatch(batch_id, batch_id, x, y, {model.name: model.predict_proba(x)})
        )
        if event is not None:
            states.append((batch_id, event.alarm_state))

    # 3: first reset; 5: real signal inside cooldown, held; 7: cooldown over, reset.
    assert states == [(3, "confirmed"), (5, "suppressed_cooldown"), (7, "confirmed")]
    assert model.reset_count == 2
