"""Detector-quality evaluation for controlled concept drift."""

from __future__ import annotations

from dataclasses import dataclass

from sentinelflow.models.common import (
    ALARM_CONFIRMED,
    DriftEvent,
)


@dataclass(frozen=True)
class DetectorQuality:
    """Summary of one detector's behavior around controlled drift."""

    model: str
    detector: str
    total_alarms: int
    false_alarms: int
    post_drift_alarms: int
    missed_detection: bool
    source_detection_delay: int | None
    observed_detection_delay: int | None


def evaluate_detector(
    events: list[DriftEvent] | tuple[DriftEvent, ...],
    *,
    model: str,
    detector: str,
    drift_start_batch: int | None,
) -> DetectorQuality:
    """Evaluate confirmed detector alarms relative to true drift start."""

    confirmed = [
        event
        for event in events
        if event.model == model
        and event.detector == detector
        and event.alarm_state == ALARM_CONFIRMED
    ]

    if drift_start_batch is None:
        return DetectorQuality(
            model=model,
            detector=detector,
            total_alarms=len(confirmed),
            false_alarms=len(confirmed),
            post_drift_alarms=0,
            missed_detection=False,
            source_detection_delay=None,
            observed_detection_delay=None,
        )

    false_alarms = [
        event
        for event in confirmed
        if event.source_batch < drift_start_batch
    ]

    post_drift = [
        event
        for event in confirmed
        if event.source_batch >= drift_start_batch
    ]

    if not post_drift:
        return DetectorQuality(
            model=model,
            detector=detector,
            total_alarms=len(confirmed),
            false_alarms=len(false_alarms),
            post_drift_alarms=0,
            missed_detection=True,
            source_detection_delay=None,
            observed_detection_delay=None,
        )

    first_detection = min(
        post_drift,
        key=lambda event: event.source_batch,
    )

    return DetectorQuality(
        model=model,
        detector=detector,
        total_alarms=len(confirmed),
        false_alarms=len(false_alarms),
        post_drift_alarms=len(post_drift),
        missed_detection=False,
        source_detection_delay=(
            first_detection.source_batch - drift_start_batch
        ),
        observed_detection_delay=(
            first_detection.observed_batch - drift_start_batch
        ),
    )


def evaluate_all_detectors(
    events: list[DriftEvent] | tuple[DriftEvent, ...],
    *,
    drift_start_batch: int | None,
    expected_detectors: tuple[tuple[str, str], ...] = (),
) -> list[DetectorQuality]:
    """Evaluate all expected detectors and any detectors found in events."""

    detector_pairs = {
        (event.model, event.detector)
        for event in events
    }

    detector_pairs.update(expected_detectors)

    return [
        evaluate_detector(
            events,
            model=model,
            detector=detector,
            drift_start_batch=drift_start_batch,
        )
        for model, detector in sorted(detector_pairs)
    ]