"""Tests for end-to-end evaluation output generation."""

import numpy as np
import pandas as pd

from sentinelflow.evaluation.pipeline import generate_evaluation_outputs
from sentinelflow.evaluation.recorder import EvaluationRecorder
from sentinelflow.experiment import (
    ExperimentResult,
    ExperimentSummary,
)
from sentinelflow.models.common import (
    ALARM_CONFIRMED,
    DriftEvent,
)
from sentinelflow.streaming.contracts import Batch


def test_generate_evaluation_outputs(tmp_path) -> None:
    """Completed experiments should generate all core evaluation CSVs."""

    recorder = EvaluationRecorder(threshold=0.5)

    batch = Batch(
        batch_id=0,
        source_rows=np.array([0, 1, 2, 3]),
        progress=0.50,
        phase="ramp_drift",
        features=pd.DataFrame(
            {
                "feature_a": [1.0, 2.0, 3.0, 4.0],
            }
        ),
        labels=np.array([1, 1, 0, 0]),
    )

    recorder.record(
        batch,
        {
            "adaptive_adwin": np.array(
                [0.9, 0.8, 0.2, 0.1]
            ),
        },
    )

    recorder.drift_start_batch = 0

    event = DriftEvent(
        model="adaptive_adwin",
        detector="adwin",
        source_batch=0,
        observed_batch=7,
        phase="ramp_drift",
        alarm_type="drift",
        alarm_state=ALARM_CONFIRMED,
        reset_count=1,
    )

    summary = ExperimentSummary(
        seed=543,
        warmup_rows=500_000,
        evaluation_rows=4,
        fraud_rows=2,
        batch_count=1,
        model_names=("adaptive_adwin",),
        feature_count=1,
        label_delay_batches=7,
        drift_event_count=1,
        phase_counts={"ramp_drift": 4},
        runtime_seconds=1.0,
    )

    result = ExperimentResult(
        summary=summary,
        drift_events=(event,),
        feature_manifest={},
    )

    paths = generate_evaluation_outputs(
        recorder,
        result,
        tmp_path,
    )

    assert set(paths) == {
        "batch_metrics",
        "phase_summary",
        "model_summary",
        "detector_quality",
        "model_comparison_figure",
        "drift_timeline_figure",
    }

    for path in paths.values():
        assert path.exists()

    detector_frame = pd.read_csv(
        paths["detector_quality"]
    )

    assert len(detector_frame) == 1
    assert detector_frame.iloc[0]["detector"] == "adwin"
    assert detector_frame.iloc[0]["source_detection_delay"] == 0
    assert detector_frame.iloc[0]["observed_detection_delay"] == 7