from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import write_parquet

from sentinelflow.data.loader import (
    CorruptDatasetError,
    DatasetError,
    DatasetNotFoundError,
    DataSettings,
    SchemaError,
    TargetValueError,
    frame_hash,
    iter_cnp_chunks,
    load_warmup_and_stream,
    row_group_order,
)

CNP = ("cnp_online", "cnp_moto")


def _expected_cnp(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["card_present_cnp"].isin(CNP)].reset_index(drop=True)


def test_strict_cnp_counts_match_pandas(data_settings, synthetic_frame) -> None:
    chunks = list(iter_cnp_chunks(data_settings))
    expected = _expected_cnp(synthetic_frame)

    assert sum(chunk.row_count for chunk in chunks) == len(expected)
    assert sum(int(chunk.labels.sum()) for chunk in chunks) == int(expected["is_fraud"].sum())
    assert len(chunks) > 1, "fixture must exercise several row groups"


def test_only_strict_cnp_channels_survive(data_settings) -> None:
    for chunk in iter_cnp_chunks(data_settings):
        assert set(chunk.features["card_present_cnp"]).issubset(CNP)


def test_target_and_identifiers_never_reach_features(data_settings) -> None:
    for chunk in iter_cnp_chunks(data_settings):
        for column in ("is_fraud", "customer_id", "transaction_id"):
            assert column not in chunk.features.columns
        assert chunk.labels.dtype == np.int8


def test_labels_align_with_source_rows(data_settings, synthetic_frame) -> None:
    expected = _expected_cnp(synthetic_frame)
    chunks = list(iter_cnp_chunks(data_settings))
    labels = np.concatenate([chunk.labels for chunk in chunks])
    amounts = pd.concat([chunk.features["amount"] for chunk in chunks], ignore_index=True)

    np.testing.assert_array_equal(labels, expected["is_fraud"].to_numpy())
    pd.testing.assert_series_equal(amounts, expected["amount"], check_names=False)


def test_cnp_row_positions_are_contiguous(data_settings) -> None:
    positions = np.concatenate([chunk.cnp_rows for chunk in iter_cnp_chunks(data_settings)])
    np.testing.assert_array_equal(positions, np.arange(len(positions)))


def test_max_rows_limits_output(data_settings) -> None:
    chunks = list(iter_cnp_chunks(data_settings, max_rows=123))
    assert sum(chunk.row_count for chunk in chunks) == 123


def test_warmup_split_has_no_overlap_or_gap(data_settings) -> None:
    total = sum(chunk.row_count for chunk in iter_cnp_chunks(data_settings))
    warmup, evaluation = load_warmup_and_stream(data_settings)
    rest = list(evaluation)

    assert warmup.row_count == data_settings.warmup_rows
    eval_positions = np.concatenate([chunk.cnp_rows for chunk in rest])
    np.testing.assert_array_equal(warmup.cnp_rows, np.arange(data_settings.warmup_rows))
    np.testing.assert_array_equal(eval_positions, np.arange(data_settings.warmup_rows, total))


def test_warmup_larger_than_dataset_fails(data_settings) -> None:
    settings = DataSettings(dataset_path=data_settings.dataset_path, warmup_rows=10**9)
    with pytest.raises(DatasetError, match="warm-up needs"):
        load_warmup_and_stream(settings)


def test_repeated_loading_gives_identical_hashes(data_settings) -> None:
    def hashes() -> list[str]:
        return [frame_hash(c.features, c.labels) for c in iter_cnp_chunks(data_settings)]

    assert hashes() == hashes()


def test_frame_hash_detects_label_change(data_settings) -> None:
    chunk = next(iter_cnp_chunks(data_settings))
    flipped = chunk.labels.copy()
    flipped[0] = 1 - flipped[0]
    assert frame_hash(chunk.features, chunk.labels) != frame_hash(chunk.features, flipped)


def test_seeded_row_group_order_is_reproducible() -> None:
    assert row_group_order(10, seed=543) == row_group_order(10, seed=543)
    assert sorted(row_group_order(10, seed=543)) == list(range(10))
    assert row_group_order(4) == [0, 1, 2, 3]


def test_missing_file_has_clear_error(tmp_path: Path) -> None:
    settings = DataSettings(dataset_path=tmp_path / "absent.parquet")
    with pytest.raises(DatasetNotFoundError, match="mendeley"):
        next(iter_cnp_chunks(settings))


def test_missing_required_column(tmp_path: Path, synthetic_frame) -> None:
    path = write_parquet(synthetic_frame.drop(columns=["card_present_cnp"]), tmp_path / "x.parquet")
    with pytest.raises(SchemaError, match="card_present_cnp"):
        next(iter_cnp_chunks(DataSettings(dataset_path=path)))


def test_nonbinary_target_rejected(tmp_path: Path, synthetic_frame) -> None:
    frame = synthetic_frame.copy()
    frame["is_fraud"] = frame["is_fraud"].astype(int)
    frame.loc[frame["card_present_cnp"] == "cnp_online", "is_fraud"] = 2
    path = write_parquet(frame, tmp_path / "x.parquet")
    with pytest.raises(TargetValueError, match="binary"):
        list(iter_cnp_chunks(DataSettings(dataset_path=path)))


def test_dictionary_encoded_channel_is_supported(tmp_path: Path, synthetic_frame) -> None:
    frame = synthetic_frame.copy()
    frame["card_present_cnp"] = frame["card_present_cnp"].astype("category")
    path = write_parquet(frame, tmp_path / "x.parquet")
    rows = sum(c.row_count for c in iter_cnp_chunks(DataSettings(dataset_path=path)))
    assert rows == len(_expected_cnp(synthetic_frame))


def test_truncated_download_has_clear_error(data_settings, tmp_path: Path) -> None:
    content = Path(data_settings.dataset_path).read_bytes()
    path = tmp_path / "cut.parquet"
    path.write_bytes(content[: len(content) * 3 // 4])
    with pytest.raises(CorruptDatasetError, match="incomplete"):
        next(iter_cnp_chunks(DataSettings(dataset_path=path)))


def test_non_parquet_file_has_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "page.parquet"
    path.write_text("<html>not a dataset</html>")
    with pytest.raises(CorruptDatasetError, match="not a Parquet file"):
        next(iter_cnp_chunks(DataSettings(dataset_path=path)))


def test_row_group_shuffle_is_seeded_and_complete(data_settings) -> None:
    def settings(seed: int) -> DataSettings:
        return DataSettings(
            dataset_path=data_settings.dataset_path,
            warmup_rows=300,
            row_group_shuffle=True,
            shuffle_seed=seed,
        )

    def groups(seed: int) -> list[int]:
        return [chunk.row_group for chunk in iter_cnp_chunks(settings(seed))]

    file_order = [chunk.row_group for chunk in iter_cnp_chunks(data_settings)]
    assert groups(543) == groups(543)
    assert groups(543) != file_order
    assert sorted(groups(543)) == sorted(file_order)
    assert groups(543) != groups(1543)

    total = sum(chunk.row_count for chunk in iter_cnp_chunks(settings(543)))
    assert total == sum(chunk.row_count for chunk in iter_cnp_chunks(data_settings))


def test_shuffle_flag_read_from_config(tmp_path: Path, synthetic_parquet) -> None:
    import yaml

    from sentinelflow.config import load_config

    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs" / "full.yaml").read_text())
    raw["paths"]["dataset"] = str(synthetic_parquet)
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs" / "x.yaml"
    path.write_text(yaml.safe_dump(raw))

    settings = DataSettings.from_config(load_config(path))
    assert settings.row_group_shuffle is True
    assert settings.shuffle_seed == raw["project"]["random_seed"]
