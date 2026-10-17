"""Memory-safe audit of the CreditTransAct Parquet file (work package 2).

Run it from the repository root:

    python -m sentinelflow.data.audit --config configs/full.yaml

It makes one pass over the file, one row group at a time, and writes:

- ``dataset_audit.json``  checksum, schema, row counts, nulls, channel and fraud
  counts, pass/fail checks and warnings
- ``channel_summary.csv`` rows and fraud rate per ``card_present_cnp`` value
- ``column_profile.csv``  dtype, nulls, range or distinct count for every column
- ``leakage_screen.csv``  how well each column *alone* predicts ``is_fraud`` on
  a strict-CNP sample; near-perfect columns are flagged for human review

The process exits with status 1 if any acceptance check fails.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from sklearn.metrics import roc_auc_score

from sentinelflow.data.loader import (
    DataSettings,
    cnp_mask,
    feature_columns,
    open_parquet,
)

VERIFIED_SHA256 = "4a33623216213beb121cdc85006cf36f336601af7645e73feabe1abdf94e0823"
NULL_CHANNEL = "<null>"
LEAKAGE_AUC_THRESHOLD = 0.99
SUSPICIOUS_NAME = re.compile(
    r"fraud|label|target|chargeback|dispute|cluster|scenario|pattern|signal|noise",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExpectedCounts:
    """Reference values from the verified file; ``None`` skips that check."""

    sha256: str | None = VERIFIED_SHA256
    source_rows: int | None = 15_000_000
    cnp_rows: int | None = 3_843_653
    cnp_fraud_rows: int | None = 564_272


NO_EXPECTATIONS = ExpectedCounts(sha256=None, source_rows=None, cnp_rows=None, cnp_fraud_rows=None)


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Stream the file through SHA-256 without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


class _ColumnProfile:
    """Running per-column statistics accumulated across row groups."""

    def __init__(self, name: str, arrow_type: pa.DataType, max_distinct: int) -> None:
        self.name = name
        self.arrow_type = arrow_type
        self.max_distinct = max_distinct
        self.rows = 0
        self.nulls = 0
        self.minimum: Any = None
        self.maximum: Any = None
        self.total = 0.0
        self.distinct: set[Any] = set()
        self.distinct_capped = False

    @property
    def is_numeric(self) -> bool:
        kind = self.arrow_type
        return pa.types.is_integer(kind) or pa.types.is_floating(kind) or pa.types.is_decimal(kind)

    @property
    def tracks_distinct(self) -> bool:
        kind = self.arrow_type
        return (
            pa.types.is_string(kind)
            or pa.types.is_large_string(kind)
            or pa.types.is_dictionary(kind)
            or pa.types.is_boolean(kind)
        )

    def update(self, values: pa.ChunkedArray) -> None:
        self.rows += len(values)
        self.nulls += values.null_count
        if self.is_numeric and len(values) > values.null_count:
            bounds = pc.min_max(values)
            low, high = bounds["min"].as_py(), bounds["max"].as_py()
            self.minimum = low if self.minimum is None else min(self.minimum, low)
            self.maximum = high if self.maximum is None else max(self.maximum, high)
            self.total += float(pc.sum(values.cast(pa.float64())).as_py() or 0.0)
        elif pa.types.is_timestamp(self.arrow_type) or pa.types.is_date(self.arrow_type):
            if len(values) > values.null_count:
                bounds = pc.min_max(values)
                low, high = bounds["min"].as_py(), bounds["max"].as_py()
                self.minimum = low if self.minimum is None else min(self.minimum, low)
                self.maximum = high if self.maximum is None else max(self.maximum, high)
        if self.tracks_distinct and not self.distinct_capped:
            if pa.types.is_dictionary(self.arrow_type):
                values = values.cast(self.arrow_type.value_type)
            for value in pc.unique(values).to_pylist():
                if value is not None:
                    self.distinct.add(value)
            if len(self.distinct) > self.max_distinct:
                self.distinct_capped = True
                self.distinct = set(list(self.distinct)[: self.max_distinct])

    def record(self) -> dict[str, Any]:
        non_null = self.rows - self.nulls
        distinct: Any = None
        examples = ""
        if self.tracks_distinct:
            distinct = f">{self.max_distinct}" if self.distinct_capped else len(self.distinct)
            examples = "|".join(sorted(map(str, self.distinct))[:8])
        return {
            "column": self.name,
            "arrow_type": str(self.arrow_type),
            "rows": self.rows,
            "nulls": self.nulls,
            "null_fraction": round(self.nulls / self.rows, 6) if self.rows else 0.0,
            "min": _jsonable(self.minimum),
            "max": _jsonable(self.maximum),
            "mean": round(self.total / non_null, 6) if self.is_numeric and non_null else None,
            "distinct": distinct,
            "examples": examples,
        }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _channel_target_counts(table: pa.Table, settings: DataSettings) -> pd.DataFrame:
    channel = table.column(settings.cnp_column)
    if pa.types.is_dictionary(channel.type):
        channel = channel.cast(channel.type.value_type)
    channel = pc.fill_null(channel.cast(pa.string()), NULL_CHANNEL)
    target = table.column(settings.target_column).cast(pa.int64())
    grouped = (
        pa.table({"channel": channel, "target": target})
        .group_by(["channel", "target"])
        .aggregate([("target", "count")])
    )
    return grouped.to_pandas().rename(columns={"target_count": "rows"})


def _sample_screen(
    sample: pd.DataFrame,
    labels: np.ndarray,
    *,
    max_categories: int = 1000,
) -> list[dict[str, Any]]:
    """Single-column predictiveness on a strict-CNP sample.

    Numeric columns are scored directly. Categorical columns are scored out of
    sample: per-category fraud rates come from the first half of the sample and
    are evaluated on the second half, so high-cardinality IDs cannot look
    predictive just by memorising rows.
    """

    rows: list[dict[str, Any]] = []
    if len(np.unique(labels)) < 2:
        return rows

    half = len(labels) // 2
    for column in sample.columns:
        series = sample[column]
        record: dict[str, Any] = {
            "column": column,
            "kind": "",
            "auc": None,
            "strength": None,
            "name_flag": bool(SUSPICIOUS_NAME.search(column)),
            "flag": False,
            "note": "",
        }
        try:
            if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
                values = series.astype("float64")
                values = values.fillna(values.median() if values.notna().any() else 0.0)
                record["kind"] = "numeric"
                record["auc"] = float(roc_auc_score(labels, values))
            else:
                text = series.astype("object").where(series.notna(), NULL_CHANNEL).astype(str)
                distinct = text.nunique()
                record["kind"] = "categorical"
                if distinct > max_categories:
                    record["note"] = f"{distinct} categories; skipped (likely an identifier)"
                elif half < 2 or len(np.unique(labels[half:])) < 2:
                    record["note"] = "sample too small"
                else:
                    rates = pd.Series(labels[:half]).groupby(text.iloc[:half].to_numpy()).mean()
                    prior = float(labels[:half].mean())
                    scores = text.iloc[half:].map(rates).fillna(prior).to_numpy()
                    record["auc"] = float(roc_auc_score(labels[half:], scores))
        except (TypeError, ValueError) as error:
            record["note"] = f"not scored: {error}"

        if record["auc"] is not None:
            strength = max(record["auc"], 1.0 - record["auc"])
            record["auc"] = round(record["auc"], 6)
            record["strength"] = round(strength, 6)
            record["flag"] = strength >= LEAKAGE_AUC_THRESHOLD
        rows.append(record)

    rows.sort(key=lambda row: -(row["strength"] or 0.0))
    return rows


def audit_dataset(
    settings: DataSettings,
    *,
    expected: ExpectedCounts = ExpectedCounts(),  # noqa: B008 - immutable dataclass
    compute_checksum: bool = True,
    screen_rows: int = 200_000,
    max_distinct: int = 10_000,
) -> dict[str, Any]:
    """Audit the dataset in one bounded pass and return a JSON-ready report."""

    started = time.perf_counter()
    parquet = open_parquet(settings)
    path = Path(settings.dataset_path)
    schema = parquet.schema_arrow

    profiles = {
        field.name: _ColumnProfile(field.name, field.type, max_distinct) for field in schema
    }
    counts: list[pd.DataFrame] = []
    target_problems: list[str] = []
    screen_frames: list[pd.DataFrame] = []
    screen_labels: list[np.ndarray] = []
    screened = 0
    screen_columns = feature_columns(schema.names, settings)

    for group in range(parquet.num_row_groups):
        table = parquet.read_row_group(group)
        for name in table.column_names:
            profiles[name].update(table.column(name))

        target = table.column(settings.target_column)
        if target.null_count:
            target_problems.append(f"row group {group}: {target.null_count} null targets")
            table = table.filter(pc.is_valid(target))
            target = table.column(settings.target_column)
        distinct_targets = set(pc.unique(target.cast(pa.int64())).to_pylist())
        if not distinct_targets.issubset({0, 1}):
            target_problems.append(f"row group {group}: target values {sorted(distinct_targets)}")
            continue

        counts.append(_channel_target_counts(table, settings))

        if screened < screen_rows:
            cnp = table.filter(cnp_mask(table, settings)).slice(0, screen_rows - screened)
            if cnp.num_rows:
                screen_frames.append(cnp.select(screen_columns).to_pandas())
                screen_labels.append(cnp.column(settings.target_column).cast(pa.int8()).to_numpy())
                screened += cnp.num_rows

    channel_counts = (
        pd.concat(counts, ignore_index=True)
        .groupby(["channel", "target"], as_index=False)["rows"]
        .sum()
        if counts
        else pd.DataFrame(columns=["channel", "target", "rows"])
    )
    channel_summary = _channel_summary(channel_counts, settings.cnp_values)

    cnp_rows = int(channel_summary.loc[channel_summary["is_cnp"], "rows"].sum())
    cnp_fraud = int(channel_summary.loc[channel_summary["is_cnp"], "fraud_rows"].sum())
    total_fraud = int(channel_summary["fraud_rows"].sum())
    source_rows = int(parquet.metadata.num_rows)

    leakage = (
        _sample_screen(pd.concat(screen_frames, ignore_index=True), np.concatenate(screen_labels))
        if screen_frames
        else []
    )

    checksum = sha256_file(path) if compute_checksum else None
    checks = _checks(
        expected=expected,
        checksum=checksum,
        source_rows=source_rows,
        cnp_rows=cnp_rows,
        cnp_fraud=cnp_fraud,
        target_problems=target_problems,
    )
    warnings = _warnings(channel_summary, settings, leakage)

    return {
        "dataset": {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": checksum,
            "expected_sha256": expected.sha256,
            "num_row_groups": parquet.num_row_groups,
            "source_rows": source_rows,
            "num_columns": len(schema.names),
        },
        "schema": [{"name": field.name, "type": str(field.type)} for field in schema],
        "settings": {
            "target_column": settings.target_column,
            "cnp_column": settings.cnp_column,
            "cnp_values": list(settings.cnp_values),
            "drop_columns": list(settings.drop_columns),
            "warmup_rows": settings.warmup_rows,
        },
        "counts": {
            "source_rows": source_rows,
            "source_fraud_rows": total_fraud,
            "source_fraud_rate": round(total_fraud / source_rows, 6) if source_rows else None,
            "cnp_rows": cnp_rows,
            "cnp_fraud_rows": cnp_fraud,
            "cnp_fraud_rate": round(cnp_fraud / cnp_rows, 6) if cnp_rows else None,
            "warmup_rows": min(settings.warmup_rows, cnp_rows),
            "evaluation_rows": max(cnp_rows - settings.warmup_rows, 0),
        },
        "channels": channel_summary.to_dict(orient="records"),
        "nulls": {name: profile.nulls for name, profile in profiles.items() if profile.nulls},
        "columns": [profile.record() for profile in profiles.values()],
        "leakage_screen": {
            "sample_rows": screened,
            "auc_threshold": LEAKAGE_AUC_THRESHOLD,
            "results": leakage,
        },
        "checks": checks,
        "warnings": warnings,
        "passed": all(check["passed"] for check in checks),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }


def _channel_summary(counts: pd.DataFrame, cnp_values: Sequence[str]) -> pd.DataFrame:
    if counts.empty:
        return pd.DataFrame(columns=["channel", "is_cnp", "rows", "fraud_rows", "fraud_rate"])
    pivot = counts.pivot_table(
        index="channel", columns="target", values="rows", aggfunc="sum", fill_value=0
    )
    legit = pivot[0] if 0 in pivot.columns else 0
    fraud = pivot[1] if 1 in pivot.columns else 0
    summary = pd.DataFrame({"rows": legit + fraud, "fraud_rows": fraud}).reset_index()
    summary["rows"] = summary["rows"].astype(int)
    summary["fraud_rows"] = summary["fraud_rows"].astype(int)
    summary["fraud_rate"] = (summary["fraud_rows"] / summary["rows"]).round(6)
    summary.insert(1, "is_cnp", summary["channel"].isin(list(cnp_values)))
    return summary.sort_values("channel").reset_index(drop=True)


def _checks(
    *,
    expected: ExpectedCounts,
    checksum: str | None,
    source_rows: int,
    cnp_rows: int,
    cnp_fraud: int,
    target_problems: list[str],
) -> list[dict[str, Any]]:
    checks = [
        {
            "name": "target_is_binary",
            "passed": not target_problems,
            "detail": "; ".join(target_problems) or "all targets are 0/1 with no nulls",
        },
        {
            "name": "cnp_rows_present",
            "passed": cnp_rows > 0,
            "detail": f"{cnp_rows} strict-CNP rows",
        },
    ]

    def compare(name: str, actual: Any, wanted: Any) -> None:
        if wanted is None:
            return
        checks.append(
            {
                "name": name,
                "passed": actual == wanted,
                "detail": f"actual={actual} expected={wanted}",
            }
        )

    if checksum is not None:
        compare("sha256_matches", checksum, expected.sha256)
    compare("source_rows", source_rows, expected.source_rows)
    compare("cnp_rows", cnp_rows, expected.cnp_rows)
    compare("cnp_fraud_rows", cnp_fraud, expected.cnp_fraud_rows)
    return checks


def _warnings(
    summary: pd.DataFrame,
    settings: DataSettings,
    leakage: list[dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    observed = set(summary["channel"]) if not summary.empty else set()

    absent = sorted(set(settings.cnp_values) - observed)
    if absent:
        warnings.append(f"Configured CNP values never occur: {', '.join(absent)}")

    lookalikes = sorted(
        value
        for value in observed
        if "cnp" in str(value).lower() and value not in settings.cnp_values
    )
    if lookalikes:
        warnings.append(
            "Channel values that look CNP but are excluded by the strict filter: "
            + ", ".join(lookalikes)
        )
    if NULL_CHANNEL in observed:
        warnings.append("Some rows have a null channel value; the strict filter excludes them.")

    flagged = [row["column"] for row in leakage if row["flag"]]
    if flagged:
        warnings.append(
            "Columns that alone separate fraud almost perfectly (review for label leakage): "
            + ", ".join(flagged)
        )
    named = [row["column"] for row in leakage if row["name_flag"]]
    if named:
        warnings.append("Column names that suggest label-derived content: " + ", ".join(named))
    return warnings


def write_audit_outputs(report: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """Write the JSON report and its three CSV companions."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "audit": directory / "dataset_audit.json",
        "channels": directory / "channel_summary.csv",
        "columns": directory / "column_profile.csv",
        "leakage": directory / "leakage_screen.csv",
    }
    paths["audit"].write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    _write_csv(paths["channels"], report["channels"])
    _write_csv(paths["columns"], report["columns"])
    _write_csv(paths["leakage"], report["leakage_screen"]["results"])
    return paths


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(report: dict[str, Any], paths: dict[str, Path]) -> None:
    counts = report["counts"]
    print(f"Source rows:       {counts['source_rows']:,}")
    print(f"Strict-CNP rows:   {counts['cnp_rows']:,}")
    print(f"CNP fraud rows:    {counts['cnp_fraud_rows']:,} (rate {counts['cnp_fraud_rate']})")
    print(f"SHA-256:           {report['dataset']['sha256']}")
    print()
    for check in report["checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        print(f"[{mark}] {check['name']}: {check['detail']}")
    for warning in report["warnings"]:
        print(f"[WARN] {warning}")
    print()
    for path in paths.values():
        print(f"wrote {path}")


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sentinelflow.data.audit",
        description="Audit the CreditTransAct Parquet file without loading it into memory.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/full.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/metrics/audit"))
    parser.add_argument("--skip-checksum", action="store_true", help="Skip SHA-256 hashing.")
    parser.add_argument(
        "--no-expected",
        action="store_true",
        help="Do not compare against the verified reference counts (for other files).",
    )
    parser.add_argument("--screen-rows", type=int, default=200_000)
    parsed = parser.parse_args(arguments)

    from sentinelflow.config import load_config

    settings = DataSettings.from_config(load_config(parsed.config))
    report = audit_dataset(
        settings,
        expected=NO_EXPECTATIONS if parsed.no_expected else ExpectedCounts(),
        compute_checksum=not parsed.skip_checksum,
        screen_rows=parsed.screen_rows,
    )
    paths = write_audit_outputs(report, parsed.output)
    _print_summary(report, paths)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
