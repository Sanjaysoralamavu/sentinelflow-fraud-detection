"""Tests for evaluation CSV exports."""

import numpy as np
import pandas as pd

from sentinelflow.evaluation.export import (
    export_batch_metrics,
    export_model_summary,
    export_phase_summary,
)
from sentinelflow.evaluation.recorder import EvaluationRecorder
from sentinelflow.streaming.contracts import Batch
from sentinelflow.evaluation.detectors import DetectorQuality
from sentinelflow.evaluation.export import (
    export_batch_metrics,
    export_detector_quality,
    export_model_summary,
    export_phase_summary,
)


def test_evaluation_csv_exports(tmp_path) -> None:
    """Evaluation exports should contain batch and overall model metrics."""

    batch = Batch(
        batch_id=0,
        source_rows=np.array([0, 1, 2, 3]),
        progress=0.25,
        phase="pre_drift",
        features=pd.DataFrame(
            {
                "feature_a": [1.0, 2.0, 3.0, 4.0],
            }
        ),
        labels=np.array([1, 1, 0, 0]),
    )

    recorder = EvaluationRecorder(threshold=0.5)

    recorder.record(
        batch,
        {
            "model_a": np.array([0.9, 0.4, 0.8, 0.1]),
            "model_b": np.array([0.9, 0.8, 0.2, 0.1]),
        },
    )

    batch_path = export_batch_metrics(
        recorder,
        tmp_path / "batch_metrics.csv",
    )
    summary_path = export_model_summary(
        recorder,
        tmp_path / "model_summary.csv",
    )
    phase_path = export_phase_summary(
    recorder,
    tmp_path / "phase_summary.csv",
    )

    assert batch_path.exists()
    assert summary_path.exists()
    assert phase_path.exists()

    batch_frame = pd.read_csv(batch_path)
    summary_frame = pd.read_csv(summary_path)
    phase_frame = pd.read_csv(phase_path)

    assert len(batch_frame) == 2
    assert set(batch_frame["model"]) == {
        "model_a",
        "model_b",
    }

    assert len(summary_frame) == 2
    assert set(summary_frame["model"]) == {
        "model_a",
        "model_b",
    }

    assert len(phase_frame) == 2

    assert set(phase_frame["model"]) == {
        "model_a",
        "model_b",
    }

    assert set(phase_frame["phase"]) == {
        "pre_drift",
    }

    model_b = summary_frame.loc[
        summary_frame["model"] == "model_b"
    ].iloc[0]

    assert model_b["tp"] == 2
    assert model_b["tn"] == 2
    assert model_b["fp"] == 0
    assert model_b["fn"] == 0

def test_detector_quality_export(tmp_path) -> None:
    """Detector-quality summaries should be exported to CSV."""

    results = [
        DetectorQuality(
            model="adaptive_adwin",
            detector="adwin",
            total_alarms=3,
            false_alarms=1,
            post_drift_alarms=2,
            missed_detection=False,
            source_detection_delay=7,
            observed_detection_delay=14,
        ),
        DetectorQuality(
            model="adaptive_ddm",
            detector="ddm",
            total_alarms=2,
            false_alarms=0,
            post_drift_alarms=2,
            missed_detection=False,
            source_detection_delay=9,
            observed_detection_delay=16,
        ),
    ]

    output_path = export_detector_quality(
        results,
        tmp_path / "detector_quality.csv",
    )

    assert output_path.exists()

    frame = pd.read_csv(output_path)

    assert len(frame) == 2
    assert set(frame["detector"]) == {
        "adwin",
        "ddm",
    }

    adwin = frame.loc[
        frame["detector"] == "adwin"
    ].iloc[0]

    assert adwin["total_alarms"] == 3
    assert adwin["false_alarms"] == 1
    assert adwin["post_drift_alarms"] == 2
    assert adwin["source_detection_delay"] == 7
    assert adwin["observed_detection_delay"] == 14