"""Stream, delay, and drift tests (work package 3).

Covers the required pull-request evidence from the implementation plan:
label-delay boundary tests, a label-preservation assertion, a no-drift test,
a drift-phase log, and reproducibility across repeated seeds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinelflow.data.loader import CNPChunk
from sentinelflow.models import common
from sentinelflow.models.adaptive import AdaptiveModel, AdaptiveSettings
from sentinelflow.streaming import contracts
from sentinelflow.streaming.contracts import PHASE_PRE_DRIFT, Batch, DelayedBatch
from sentinelflow.streaming.drift_injector import (
    DriftInjector,
    DriftSettings,
    drift_intensity,
    drift_phase,
)
from sentinelflow.streaming.label_delay import (
    DelayedLabelQueue,
    StreamSettings,
    batch_chunks,
    batch_stream,
    run_stream,
)

N_ROWS = 2000
BATCH_SIZE = 200
FEATURE_COLUMNS = ["amount", "velocity", "merchant_risk"]


def make_frame(
    seed: int = 11, n_rows: int = N_ROWS, fraud_rate: float = 0.2
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(
        {col: rng.normal(loc=10.0, scale=3.0, size=n_rows) for col in FEATURE_COLUMNS}
    )
    labels = (rng.random(n_rows) < fraud_rate).astype(int)
    return features, labels


def make_drift_settings(enabled: bool = True, **overrides) -> DriftSettings:
    values = {
        "enabled": enabled,
        "start_fraction": 0.45,
        "end_fraction": 0.70,
        "max_intensity": 0.90,
        "random_seed": 1543,
    }
    values.update(overrides)
    return DriftSettings(**values)


def make_stream_settings(seed: int = 543, **overrides) -> StreamSettings:
    values = {
        "batch_size": BATCH_SIZE,
        "label_delay_batches": 7,
        "seed": seed,
        "within_batch_shuffle": True,
    }
    values.update(overrides)
    return StreamSettings(**values)


def run_full_stream(
    stream_settings: StreamSettings, drift_settings: DriftSettings, frame_seed: int = 11
) -> list[Batch]:
    features, labels = make_frame(seed=frame_seed)
    injector = DriftInjector(drift_settings)
    return list(batch_stream(features, labels, stream_settings, injector))


# ---------------------------------------------------------------------------
# Drift-phase log
# ---------------------------------------------------------------------------


def test_drift_phase_boundaries() -> None:
    settings = make_drift_settings()

    assert drift_phase(0.10, settings) == "pre_drift"
    assert drift_phase(0.44, settings) == "pre_drift"
    assert drift_phase(0.45, settings) == "ramp_drift"
    assert drift_phase(0.60, settings) == "ramp_drift"
    assert drift_phase(0.69, settings) == "ramp_drift"
    assert drift_phase(0.70, settings) == "drifted"
    assert drift_phase(1.00, settings) == "drifted"


def test_drift_intensity_ramps_linearly_then_holds() -> None:
    settings = make_drift_settings()

    assert drift_intensity(0.10, settings) == 0.0
    midpoint = drift_intensity(0.575, settings)  # halfway through the 0.45-0.70 ramp
    assert midpoint == pytest.approx(settings.max_intensity / 2, rel=1e-6)
    assert drift_intensity(0.70, settings) == settings.max_intensity
    assert drift_intensity(1.00, settings) == settings.max_intensity


def test_batch_stream_logs_expected_phase_sequence() -> None:
    settings = make_stream_settings()
    batches = run_full_stream(settings, make_drift_settings())

    phases = [batch.phase for batch in batches]
    assert phases == [
        "pre_drift",
        "pre_drift",
        "pre_drift",
        "pre_drift",
        "ramp_drift",
        "ramp_drift",
        "drifted",
        "drifted",
        "drifted",
        "drifted",
    ]


# ---------------------------------------------------------------------------
# Label preservation
# ---------------------------------------------------------------------------


def test_drift_never_changes_labels() -> None:
    features, labels = make_frame()
    injector = DriftInjector(make_drift_settings())
    original_labels = labels.copy()

    result = injector.apply(features, labels, progress=0.80)

    np.testing.assert_array_equal(labels, original_labels)
    assert result.affected_rows == int((labels == 1).sum())


def test_drift_handles_integer_feature_columns_without_warning(recwarn) -> None:
    # The real dataset (schema confirmed against configs/full.yaml and a live audit) has
    # genuine int8/int16 numeric columns, e.g. cards_on_device_30d, account_age_days. A
    # fractional shift must not trigger a pandas dtype-incompatibility warning or error.
    rng = np.random.default_rng(3)
    n_rows = 500
    features = pd.DataFrame(
        {
            "transaction_amount": rng.normal(50.0, 10.0, size=n_rows),
            "cards_on_device_30d": rng.integers(1, 10, size=n_rows).astype("int16"),
            "chargeback_history_count": rng.integers(0, 5, size=n_rows).astype("int8"),
        }
    )
    labels = (rng.random(n_rows) < 0.2).astype(int)
    injector = DriftInjector(make_drift_settings())

    result = injector.apply(features, labels, progress=0.80)

    assert not any(issubclass(w.category, FutureWarning) for w in recwarn.list)
    assert result.features["cards_on_device_30d"].dtype.kind == "f"
    assert result.features["chargeback_history_count"].dtype.kind == "f"


def test_drift_only_touches_fraud_rows() -> None:
    features, labels = make_frame()
    injector = DriftInjector(make_drift_settings())

    result = injector.apply(features, labels, progress=0.80)

    fraud_mask = labels == 1
    pd.testing.assert_frame_equal(
        result.features.loc[~fraud_mask].reset_index(drop=True),
        features.loc[~fraud_mask].reset_index(drop=True),
    )
    assert not result.features.loc[fraud_mask].equals(features.loc[fraud_mask])


# ---------------------------------------------------------------------------
# No-drift control
# ---------------------------------------------------------------------------


def test_no_drift_control_changes_zero_feature_values() -> None:
    settings = make_stream_settings()
    disabled = make_drift_settings(enabled=False)
    features, labels = make_frame()

    batches = run_full_stream(settings, disabled)

    for batch in batches:
        original = features.iloc[batch.source_rows].reset_index(drop=True)
        pd.testing.assert_frame_equal(batch.features, original)


def test_no_drift_control_intensity_always_zero() -> None:
    disabled = make_drift_settings(enabled=False)

    for progress in (0.1, 0.5, 0.6, 0.9, 1.0):
        assert drift_intensity(progress, disabled) == 0.0


def test_no_drift_control_phase_stays_stable() -> None:
    disabled = make_drift_settings(enabled=False)

    for progress in (0.1, 0.45, 0.5, 0.69, 0.7, 0.9, 1.0):
        assert drift_phase(progress, disabled) == PHASE_PRE_DRIFT

    batches = run_full_stream(make_stream_settings(), disabled)
    assert {batch.phase for batch in batches} == {PHASE_PRE_DRIFT}


def test_no_drift_control_phase_propagates_to_delayed_batches() -> None:
    batches = run_full_stream(make_stream_settings(), make_drift_settings(enabled=False))
    queue = DelayedLabelQueue(2)
    for batch in batches:
        queue.enqueue(batch, batch.features.to_numpy(), {})

    assert {item.phase for item in queue.flush()} == {PHASE_PRE_DRIFT}


def test_no_drift_control_reports_no_affected_rows() -> None:
    features, labels = make_frame()
    injector = DriftInjector(make_drift_settings(enabled=False))

    for progress in (0.2, 0.6, 1.0):
        result = injector.apply(features, labels, progress)
        assert result.intensity == 0.0
        assert result.phase == PHASE_PRE_DRIFT
        assert result.affected_rows == 0
        assert result.affected_columns == ()
        pd.testing.assert_frame_equal(result.features, features)


# ---------------------------------------------------------------------------
# Reproducibility across repeated seeds
# ---------------------------------------------------------------------------


def test_same_seed_reproduces_batches_exactly() -> None:
    settings = make_stream_settings(seed=543)
    drift_settings = make_drift_settings()

    first = run_full_stream(settings, drift_settings)
    second = run_full_stream(settings, drift_settings)

    assert len(first) == len(second)
    for left, right in zip(first, second, strict=True):
        assert left.phase == right.phase
        np.testing.assert_array_equal(left.source_rows, right.source_rows)
        np.testing.assert_array_equal(left.labels, right.labels)
        pd.testing.assert_frame_equal(left.features, right.features)


def test_different_seed_changes_row_order() -> None:
    drift_settings = make_drift_settings()

    first = run_full_stream(make_stream_settings(seed=543), drift_settings)
    second = run_full_stream(make_stream_settings(seed=2543), drift_settings)

    assert not np.array_equal(first[0].source_rows, second[0].source_rows)


def test_batch_ids_and_row_counts_are_deterministic() -> None:
    settings = make_stream_settings()
    batches = run_full_stream(settings, make_drift_settings())

    assert [batch.batch_id for batch in batches] == list(range(10))
    assert all(batch.row_count == BATCH_SIZE for batch in batches)

    all_rows = np.concatenate([batch.source_rows for batch in batches])
    assert sorted(all_rows.tolist()) == list(range(N_ROWS))


# ---------------------------------------------------------------------------
# Label-delay boundary tests
# ---------------------------------------------------------------------------


def make_simple_batch(batch_id: int, rows: int = 4) -> Batch:
    features = pd.DataFrame({"x": np.arange(rows, dtype=float)})
    labels = np.zeros(rows, dtype=int)
    return Batch(
        batch_id=batch_id,
        source_rows=np.arange(rows),
        progress=(batch_id + 1) / 100,
        phase="pre_drift",
        features=features,
        labels=labels,
    )


@pytest.mark.parametrize("delay", [0, 3, 7, 14])
def test_delayed_label_queue_releases_at_observed_batch(delay: int) -> None:
    queue = DelayedLabelQueue(delay_batches=delay)
    batch = make_simple_batch(batch_id=5)
    queue.enqueue(batch, features=batch.features.to_numpy(), predictions={"model": np.zeros(4)})

    observed_batch = 5 + delay

    if observed_batch > 5:
        assert queue.release_due(5) == []
    if observed_batch > 0:
        assert queue.release_due(observed_batch - 1) == []

    due = queue.release_due(observed_batch)
    assert len(due) == 1
    assert isinstance(due[0], DelayedBatch)
    assert due[0].source_batch == 5
    assert due[0].observed_batch == observed_batch
    assert len(queue) == 0


def test_delayed_label_queue_releases_in_source_batch_order() -> None:
    queue = DelayedLabelQueue(delay_batches=0)
    for batch_id in (2, 0, 1):
        batch = make_simple_batch(batch_id)
        queue.enqueue(batch, features=batch.features.to_numpy(), predictions={})

    due = queue.release_due(2)
    assert [item.source_batch for item in due] == [2, 0, 1]


def test_delayed_label_queue_flushes_remaining_at_stream_end() -> None:
    queue = DelayedLabelQueue(delay_batches=14)
    for batch_id in range(3):
        batch = make_simple_batch(batch_id)
        queue.enqueue(batch, features=batch.features.to_numpy(), predictions={})

    assert queue.release_due(5) == []
    flushed = queue.flush()

    assert len(flushed) == 3
    assert len(queue) == 0
    assert queue.flush() == []


def test_delayed_label_queue_rejects_negative_delay() -> None:
    with pytest.raises(ValueError):
        DelayedLabelQueue(delay_batches=-1)


# ---------------------------------------------------------------------------
# Prequential order: predict before learn, even at zero delay
# ---------------------------------------------------------------------------


def test_run_stream_predicts_before_learning_with_zero_delay() -> None:
    calls: list[tuple[str, int]] = []

    def predict(batch: Batch):
        calls.append(("predict", batch.batch_id))
        return batch.features.to_numpy(), {"model": np.full(batch.row_count, 0.1)}

    def record(batch: Batch, predictions) -> None:
        calls.append(("record", batch.batch_id))

    def learn(delayed: DelayedBatch) -> None:
        calls.append(("learn", delayed.source_batch))

    queue = DelayedLabelQueue(delay_batches=0)
    batches = [make_simple_batch(batch_id) for batch_id in range(3)]

    run_stream(batches, queue, predict, record, learn)

    assert calls == [
        ("predict", 0),
        ("record", 0),
        ("learn", 0),
        ("predict", 1),
        ("record", 1),
        ("learn", 1),
        ("predict", 2),
        ("record", 2),
        ("learn", 2),
    ]


def test_run_stream_flushes_pending_after_delay_exceeds_stream_length() -> None:
    calls: list[tuple[str, int]] = []

    def predict(batch: Batch):
        return batch.features.to_numpy(), {}

    def record(batch: Batch, predictions) -> None:
        return None

    def learn(delayed: DelayedBatch) -> None:
        calls.append(("learn", delayed.source_batch))

    queue = DelayedLabelQueue(delay_batches=7)
    batches = [make_simple_batch(batch_id) for batch_id in range(3)]

    run_stream(batches, queue, predict, record, learn)

    assert calls == [("learn", 0), ("learn", 1), ("learn", 2)]
    assert len(queue) == 0


# ---------------------------------------------------------------------------
# Integration with the data loader and adaptive models
# ---------------------------------------------------------------------------


def test_delayed_batch_is_the_models_definition() -> None:
    assert contracts.DelayedBatch is common.DelayedBatch
    assert contracts.WARMUP_BATCH == common.WARMUP_BATCH


def test_batch_chunks_matches_batch_stream_for_uneven_chunks() -> None:
    features, labels = make_frame()
    settings = make_stream_settings()
    drift_settings = make_drift_settings()

    expected = list(batch_stream(features, labels, settings, DriftInjector(drift_settings)))

    chunks = []
    start = 0
    for size in [137, 500, 61, 1000, 302]:
        stop = start + size
        chunks.append(
            CNPChunk(
                row_group=len(chunks),
                cnp_rows=np.arange(start, stop),
                features=features.iloc[start:stop].reset_index(drop=True),
                labels=labels[start:stop],
            )
        )
        start = stop
    assert start == N_ROWS

    actual = list(batch_chunks(chunks, N_ROWS, settings, DriftInjector(drift_settings)))

    assert [b.batch_id for b in actual] == [b.batch_id for b in expected]
    for left, right in zip(actual, expected, strict=True):
        assert left.phase == right.phase
        assert left.progress == pytest.approx(right.progress)
        np.testing.assert_array_equal(left.source_rows, right.source_rows)
        np.testing.assert_array_equal(left.labels, right.labels)
        pd.testing.assert_frame_equal(left.features, right.features)


def test_run_stream_feeds_adaptive_model_with_released_labels() -> None:
    features, labels = make_frame()
    settings = make_stream_settings(label_delay_batches=3)
    batches = batch_stream(features, labels, settings, DriftInjector(make_drift_settings()))

    model = AdaptiveModel("adwin", AdaptiveSettings(), seed=543)
    model.warm_up(features.to_numpy(), labels)

    released: list[DelayedBatch] = []

    def predict(batch: Batch):
        x = batch.features.to_numpy()
        return x, {model.name: model.predict_proba(x)}

    def learn(delayed: DelayedBatch) -> None:
        released.append(delayed)
        model.learn(delayed)

    run_stream(batches, DelayedLabelQueue(3), predict, lambda b, p: None, learn)

    assert [d.source_batch for d in released] == list(range(10))
    assert all(d.observed_batch - d.source_batch == 3 for d in released[:7])
