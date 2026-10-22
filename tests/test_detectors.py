"""Tests for detector-quality evaluation."""

from sentinelflow.evaluation.detectors import evaluate_detector
from sentinelflow.models.common import (
    ALARM_CONFIRMED,
    ALARM_SUPPRESSED_COOLDOWN,
    ALARM_SUPPRESSED_GRACE,
    DriftEvent,
)


def make_event(
    *,
    source_batch: int,
    observed_batch: int,
    phase: str,
) -> DriftEvent:
    """Create a confirmed ADWIN event for detector tests."""

    return DriftEvent(
        model="adaptive_adwin",
        detector="adwin",
        source_batch=source_batch,
        observed_batch=observed_batch,
        phase=phase,
        alarm_type="drift",
        alarm_state=ALARM_CONFIRMED,
        reset_count=1,
    )


def test_detector_quality_counts_false_alarm_and_detection_delay() -> None:
    """Confirmed alarms should be classified relative to true drift start."""

    events = [
        make_event(
            source_batch=20,
            observed_batch=27,
            phase="pre_drift",
        ),
        make_event(
            source_batch=37,
            observed_batch=44,
            phase="ramp_drift",
        ),
        make_event(
            source_batch=55,
            observed_batch=62,
            phase="drifted",
        ),
    ]

    result = evaluate_detector(
        events,
        model="adaptive_adwin",
        detector="adwin",
        drift_start_batch=30,
    )

    assert result.total_alarms == 3
    assert result.false_alarms == 1
    assert result.post_drift_alarms == 2
    assert result.missed_detection is False

    assert result.source_detection_delay == 7
    assert result.observed_detection_delay == 14


def test_detector_quality_marks_missed_detection() -> None:
    """A detector with no post-drift alarm should be marked as missed."""

    events = [
        make_event(
            source_batch=20,
            observed_batch=27,
            phase="pre_drift",
        ),
    ]

    result = evaluate_detector(
        events,
        model="adaptive_adwin",
        detector="adwin",
        drift_start_batch=30,
    )

    assert result.total_alarms == 1
    assert result.false_alarms == 1
    assert result.post_drift_alarms == 0
    assert result.missed_detection is True
    assert result.source_detection_delay is None
    assert result.observed_detection_delay is None


def test_no_drift_treats_every_confirmed_alarm_as_false() -> None:
    """Every confirmed alarm is false when drift injection is disabled."""

    events = [
        make_event(
            source_batch=20,
            observed_batch=27,
            phase="pre_drift",
        ),
        make_event(
            source_batch=40,
            observed_batch=47,
            phase="pre_drift",
        ),
    ]

    result = evaluate_detector(
        events,
        model="adaptive_adwin",
        detector="adwin",
        drift_start_batch=None,
    )

    assert result.total_alarms == 2
    assert result.false_alarms == 2
    assert result.post_drift_alarms == 0
    assert result.missed_detection is False
    assert result.source_detection_delay is None
    assert result.observed_detection_delay is None

def test_suppressed_signals_are_not_counted_as_confirmed_alarms() -> None:
    """Grace and cooldown signals should not count as confirmed alarms."""

    events = [
        DriftEvent(
            model="adaptive_adwin",
            detector="adwin",
            source_batch=10,
            observed_batch=17,
            phase="pre_drift",
            alarm_type="drift",
            alarm_state=ALARM_SUPPRESSED_GRACE,
            reset_count=0,
        ),
        DriftEvent(
            model="adaptive_adwin",
            detector="adwin",
            source_batch=35,
            observed_batch=42,
            phase="ramp_drift",
            alarm_type="drift",
            alarm_state=ALARM_SUPPRESSED_COOLDOWN,
            reset_count=1,
        ),
        make_event(
            source_batch=40,
            observed_batch=47,
            phase="ramp_drift",
        ),
    ]

    result = evaluate_detector(
        events,
        model="adaptive_adwin",
        detector="adwin",
        drift_start_batch=30,
    )

    assert result.total_alarms == 1
    assert result.false_alarms == 0
    assert result.post_drift_alarms == 1
    assert result.missed_detection is False
    assert result.source_detection_delay == 10
    assert result.observed_detection_delay == 17