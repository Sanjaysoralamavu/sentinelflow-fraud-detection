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