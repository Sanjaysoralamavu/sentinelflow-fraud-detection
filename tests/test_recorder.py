"""Tests for the evaluation prediction recorder."""

import numpy as np
import pandas as pd
import pytest

from sentinelflow.evaluation.recorder import EvaluationRecorder
from sentinelflow.streaming.contracts import Batch


def test_recorder_collects_metrics_for_each_model() -> None:
    """Recorder should create one metric record per model for each batch."""

    batch = Batch(
        batch_id=3,
        source_rows=np.array([100, 101, 102, 103]),
        progress=0.25,
        phase="pre_drift",
        features=pd.DataFrame(
            {
                "feature_a": [1.0, 2.0, 3.0, 4.0],
            }
        ),
        labels=np.array([1, 1, 0, 0]),
    )

    predictions = {
        "model_a": np.array([0.9, 0.4, 0.8, 0.1]),
        "model_b": np.array([0.8, 0.7, 0.2, 0.1]),
    }

    recorder = EvaluationRecorder(threshold=0.5)

    recorder.record(batch, predictions)

    assert len(recorder.records) == 2

    first = recorder.records[0]

    assert first["model"] == "model_a"
    assert first["batch"] == 3
    assert first["progress"] == pytest.approx(0.25)
    assert first["phase"] == "pre_drift"

    assert first["tp"] == 1
    assert first["tn"] == 1
    assert first["fp"] == 1
    assert first["fn"] == 1

    assert first["precision"] == pytest.approx(0.5)
    assert first["recall"] == pytest.approx(0.5)
    assert first["f1"] == pytest.approx(0.5)
    assert first["fpr"] == pytest.approx(0.5)

    second = recorder.records[1]

    assert second["model"] == "model_b"
    assert second["tp"] == 2
    assert second["tn"] == 2
    assert second["fp"] == 0
    assert second["fn"] == 0
    assert second["precision"] == pytest.approx(1.0)
    assert second["recall"] == pytest.approx(1.0)
    assert second["f1"] == pytest.approx(1.0)
    assert second["fpr"] == pytest.approx(0.0)