from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from conftest import write_parquet

from sentinelflow.data.audit import (
    NO_EXPECTATIONS,
    REDACTED,
    ExpectedCounts,
    audit_dataset,
    sha256_file,
    write_audit_outputs,
)
from sentinelflow.data.loader import DataSettings

CNP = ("cnp_online", "cnp_moto")


def test_counts_match_pandas(data_settings, synthetic_frame) -> None:
    report = audit_dataset(data_settings, expected=NO_EXPECTATIONS)
    cnp = synthetic_frame[synthetic_frame["card_present_cnp"].isin(CNP)]

    assert report["counts"]["source_rows"] == len(synthetic_frame)
    assert report["counts"]["cnp_rows"] == len(cnp)
    assert report["counts"]["cnp_fraud_rows"] == int(cnp["is_fraud"].sum())
    assert report["passed"]


def test_null_counts_are_reported(data_settings, synthetic_frame) -> None:
    report = audit_dataset(data_settings, expected=NO_EXPECTATIONS)
    assert report["nulls"]["amount"] == int(synthetic_frame["amount"].isna().sum())


def test_expected_counts_mismatch_fails(data_settings) -> None:
    report = audit_dataset(data_settings, expected=ExpectedCounts(sha256=None, cnp_rows=1))
    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert "cnp_rows" in failed
    assert not report["passed"]


def test_checksum_matches_streamed_hash(data_settings) -> None:
    wanted = sha256_file(data_settings.dataset_path)
    expected = ExpectedCounts(sha256=wanted, source_rows=None, cnp_rows=None, cnp_fraud_rows=None)
    report = audit_dataset(data_settings, expected=expected)
    assert report["dataset"]["sha256"] == wanted
    assert report["passed"]


def test_nonbinary_target_fails_check(tmp_path: Path, synthetic_frame) -> None:
    frame = synthetic_frame.copy()
    frame["is_fraud"] = frame["is_fraud"].astype(int)
    frame.loc[0, "is_fraud"] = 3
    path = write_parquet(frame, tmp_path / "x.parquet")
    report = audit_dataset(DataSettings(dataset_path=path), expected=NO_EXPECTATIONS)
    assert not report["passed"]


def test_unknown_cnp_lookalike_is_warned(tmp_path: Path, synthetic_frame) -> None:
    frame = synthetic_frame.copy()
    frame.loc[:20, "card_present_cnp"] = "cnp_other"
    path = write_parquet(frame, tmp_path / "x.parquet")
    report = audit_dataset(DataSettings(dataset_path=path), expected=NO_EXPECTATIONS)
    assert any("cnp_other" in warning for warning in report["warnings"])


def test_leaky_column_is_flagged(tmp_path: Path, synthetic_frame) -> None:
    frame = synthetic_frame.copy()
    jitter = np.random.default_rng(0).random(len(frame)) * 0.05
    frame["risk_score"] = frame["is_fraud"] * 0.9 + jitter
    path = write_parquet(frame, tmp_path / "x.parquet")
    report = audit_dataset(DataSettings(dataset_path=path), expected=NO_EXPECTATIONS)
    flagged = {row["column"] for row in report["leakage_screen"]["results"] if row["flag"]}
    assert flagged == {"risk_score"}


def test_identifiers_excluded_from_leakage_screen(data_settings) -> None:
    report = audit_dataset(data_settings, expected=NO_EXPECTATIONS)
    screened = {row["column"] for row in report["leakage_screen"]["results"]}
    assert "transaction_id" not in screened
    assert "is_fraud" not in screened


def test_outputs_written(data_settings, tmp_path: Path) -> None:
    report = audit_dataset(data_settings, expected=NO_EXPECTATIONS)
    paths = write_audit_outputs(report, tmp_path / "audit")
    assert json.loads(paths["audit"].read_text())["counts"]["cnp_rows"] > 0
    header = paths["channels"].read_text().splitlines()[0]
    assert header == "channel,is_cnp,rows,fraud_rows,fraud_rate"
    assert paths["columns"].stat().st_size > 0


def test_identifier_examples_are_redacted(data_settings, synthetic_frame, tmp_path: Path) -> None:
    report = audit_dataset(data_settings, expected=NO_EXPECTATIONS)
    columns = {row["column"]: row for row in report["columns"]}

    for column in ("customer_id", "transaction_id"):
        assert columns[column]["examples"] == REDACTED

    assert columns["device_type"]["examples"] != REDACTED
    assert "mobile" in columns["device_type"]["examples"]

    paths = write_audit_outputs(report, tmp_path / "audit")
    written = paths["columns"].read_text() + paths["audit"].read_text()
    for value in synthetic_frame["transaction_id"].head(50):
        assert value not in written
    for value in synthetic_frame["customer_id"].unique()[:50]:
        assert value not in written
