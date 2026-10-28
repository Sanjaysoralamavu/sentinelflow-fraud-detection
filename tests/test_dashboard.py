"""Tests for the SentinelFlow dashboard."""

import pandas as pd

from sentinelflow.dashboard import generate_dashboard
from sentinelflow.models.common import (
    ALARM_CONFIRMED,
    DriftEvent,
)


def test_generate_dashboard_creates_html(tmp_path) -> None:
    """Dashboard should create a standalone Plotly HTML file."""

    frame = pd.DataFrame(
        {
            "model": [
                "static_logistic",
                "static_logistic",
                "adaptive_adwin",
                "adaptive_adwin",
            ],
            "batch": [0, 1, 0, 1],
            "phase": [
                "pre_drift",
                "ramp_drift",
                "pre_drift",
                "ramp_drift",
            ],
            "f1": [0.90, 0.65, 0.89, 0.85],
        }
    )

    output_path = generate_dashboard(
        frame,
        tmp_path / "index.html",
        drift_start_batch=1,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    html = output_path.read_text(encoding="utf-8")

    assert "<html" in html.lower()
    assert "plotly" in html.lower()
    assert "static_logistic" in html
    assert "adaptive_adwin" in html


def test_dashboard_accepts_confirmed_drift_events(tmp_path) -> None:
    """Dashboard should render confirmed detector alarms."""

    frame = pd.DataFrame(
        {
            "model": [
                "adaptive_adwin",
                "adaptive_adwin",
                "adaptive_adwin",
            ],
            "batch": [0, 1, 2],
            "phase": [
                "pre_drift",
                "ramp_drift",
                "drifted",
            ],
            "f1": [0.90, 0.82, 0.87],
        }
    )

    event = DriftEvent(
        model="adaptive_adwin",
        detector="adwin",
        source_batch=2,
        observed_batch=9,
        phase="drifted",
        alarm_type="drift",
        alarm_state=ALARM_CONFIRMED,
        reset_count=1,
    )

    output_path = generate_dashboard(
        frame,
        tmp_path / "index.html",
        metric="f1",
        drift_start_batch=1,
        drift_events=(event,),
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
