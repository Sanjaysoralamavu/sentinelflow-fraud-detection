"""End-to-end tests for SentinelFlow experiment orchestration."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from sentinelflow.config import ProjectConfig, load_config
from sentinelflow.experiment import run_experiment, write_experiment_artifacts
from sentinelflow.streaming.contracts import Batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CapturingRecorder:
    """Small in-memory recorder used only by synthetic tests."""

    def __init__(self) -> None:
        self.batch_ids: list[int] = []
        self.source_rows: list[int] = []
        self.phases: list[str] = []
        self.predictions: list[dict[str, np.ndarray]] = []

    def record(
        self,
        batch: Batch,
        predictions: dict[str, np.ndarray],
    ) -> None:
        self.batch_ids.append(batch.batch_id)
        self.source_rows.extend(int(row) for row in batch.source_rows)
        self.phases.append(batch.phase)
        self.predictions.append(
            {name: np.asarray(scores).copy() for name, scores in predictions.items()}
        )


def make_config(
    synthetic_parquet: Path,
    tmp_path: Path,
) -> ProjectConfig:
    """Create a fast experiment configuration using synthetic Parquet."""

    base_config = load_config(PROJECT_ROOT / "configs" / "full.yaml")
    raw = deepcopy(base_config.raw)

    raw["paths"]["dataset"] = str(synthetic_parquet)
    raw["paths"]["results"] = str(tmp_path / "results")

    raw["data"].update(
        {
            "batch_size": 100,
            "warmup_rows": 300,
        }
    )

    raw["stream"].update(
        {
            "label_delay_batches": 2,
            "row_group_shuffle": True,
            "within_batch_shuffle": True,
        }
    )

    raw["drift"].update(
        {
            "enabled": True,
            "start_fraction": 0.25,
            "end_fraction": 0.75,
            "max_intensity": 0.60,
            "random_seed": 1543,
        }
    )

    raw["models"]["random_forest"].update(
        {
            "enabled": True,
            "training_rows": 250,
            "n_estimators": 8,
            "max_depth": 5,
            "min_samples_leaf": 2,
        }
    )

    raw["models"]["adaptive"].update(
        {
            "detector_sample_size": 50,
            "recent_window_rows": 300,
            "ddm_warm_start": 10,
            "detector_grace_batches": 100,
            "detector_cooldown_batches": 3,
        }
    )

    raw["models"]["online_ensemble"].update(
        {
            "enabled": True,
            "n_estimators": 3,
        }
    )

    return ProjectConfig(
        raw=raw,
        root=base_config.root,
    )


def test_end_to_end_experiment_processes_every_evaluation_row(
    synthetic_parquet: Path,
    tmp_path: Path,
) -> None:
    config = make_config(synthetic_parquet, tmp_path)
    recorder = CapturingRecorder()

    result = run_experiment(
        config,
        recorder=recorder,
        max_evaluation_rows=400,
    )

    assert result.summary.warmup_rows == 300
    assert result.summary.evaluation_rows == 400
    assert result.summary.batch_count == 4
    assert result.summary.fraud_rows > 0
    assert result.summary.feature_count > 0
    assert result.summary.label_delay_batches == 2
    assert result.summary.drift_event_count == len(result.drift_events)

    assert result.summary.model_names == (
        "static_logistic",
        "static_random_forest",
        "adaptive_none",
        "adaptive_adwin",
        "adaptive_ddm",
        "online_ensemble",
    )

    assert recorder.batch_ids == [0, 1, 2, 3]
    assert sorted(recorder.source_rows) == list(range(300, 700))
    assert len(set(recorder.source_rows)) == 400
    assert sum(result.summary.phase_counts.values()) == 400

    forbidden = {
        "is_fraud",
        "customer_id",
        "transaction_id",
    }
    included = set(result.feature_manifest["included"])

    assert forbidden.isdisjoint(included)


def test_every_model_produces_one_probability_per_row(
    synthetic_parquet: Path,
    tmp_path: Path,
) -> None:
    config = make_config(synthetic_parquet, tmp_path)
    recorder = CapturingRecorder()

    result = run_experiment(
        config,
        recorder=recorder,
        max_evaluation_rows=250,
    )

    expected_names = set(result.summary.model_names)

    for batch_predictions in recorder.predictions:
        assert set(batch_predictions) == expected_names

        row_count = len(next(iter(batch_predictions.values())))

        for scores in batch_predictions.values():
            assert scores.shape == (row_count,)
            assert np.all(np.isfinite(scores))
            assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_experiment_is_reproducible_with_fixed_seed(
    synthetic_parquet: Path,
    tmp_path: Path,
) -> None:
    config = make_config(synthetic_parquet, tmp_path)
    first_recorder = CapturingRecorder()
    second_recorder = CapturingRecorder()

    first = run_experiment(
        config,
        recorder=first_recorder,
        max_evaluation_rows=300,
    )
    second = run_experiment(
        config,
        recorder=second_recorder,
        max_evaluation_rows=300,
    )

    assert first_recorder.batch_ids == second_recorder.batch_ids
    assert first_recorder.source_rows == second_recorder.source_rows
    assert first_recorder.phases == second_recorder.phases
    assert first.feature_manifest == second.feature_manifest
    assert first.drift_events == second.drift_events

    for first_batch, second_batch in zip(
        first_recorder.predictions,
        second_recorder.predictions,
        strict=True,
    ):
        assert first_batch.keys() == second_batch.keys()

        for name in first_batch:
            np.testing.assert_array_equal(
                first_batch[name],
                second_batch[name],
            )


def test_invalid_smoke_run_limit_is_rejected(
    synthetic_parquet: Path,
    tmp_path: Path,
) -> None:
    config = make_config(synthetic_parquet, tmp_path)

    with pytest.raises(
        ValueError,
        match="max_evaluation_rows must be greater than zero",
    ):
        run_experiment(config, max_evaluation_rows=0)


def test_full_run_requires_verified_audit(
    synthetic_parquet: Path,
    tmp_path: Path,
) -> None:
    config = make_config(synthetic_parquet, tmp_path)

    with pytest.raises(
        FileNotFoundError,
        match="Dataset audit is required",
    ):
        run_experiment(config)


def test_experiment_artifacts_are_written(
    synthetic_parquet: Path,
    tmp_path: Path,
) -> None:
    config = make_config(synthetic_parquet, tmp_path)

    result = run_experiment(
        config,
        max_evaluation_rows=200,
    )

    output_directory = tmp_path / "experiment-artifacts"
    paths = write_experiment_artifacts(
        result,
        config,
        output_directory,
    )

    assert set(paths) == {
        "run_manifest",
        "feature_manifest",
        "drift_events",
    }

    for path in paths.values():
        assert path.is_file()

    run_manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
    feature_manifest = json.loads(paths["feature_manifest"].read_text(encoding="utf-8"))
    drift_events = json.loads(paths["drift_events"].read_text(encoding="utf-8"))

    assert run_manifest["summary"]["seed"] == config.seed
    assert run_manifest["summary"]["warmup_rows"] == 300
    assert run_manifest["summary"]["evaluation_rows"] == 200
    assert run_manifest["summary"]["model_names"] == list(result.summary.model_names)

    assert run_manifest["resolved_paths"]["dataset"] == str(config.dataset_path)
    assert run_manifest["resolved_paths"]["results"] == str(config.results_path)

    assert feature_manifest["n_output_features"] == (result.summary.feature_count)
    assert isinstance(drift_events, list)
    assert len(drift_events) == result.summary.drift_event_count
