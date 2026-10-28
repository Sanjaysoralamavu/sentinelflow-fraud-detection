"""Value-level screen for near-deterministic fraud signals (issue #9).

The single-column screen in :mod:`sentinelflow.data.audit` scores each column
as a whole, so a column can look moderate (AUC 0.85) while one of its values is
almost always fraud. This module looks one level deeper, in one bounded pass
over the strict-CNP rows:

- ``value_screen``: rows, fraud rows and fraud rate for every value of every
  categorical or low-cardinality integer column. Values whose fraud rate is at
  least ``rate_threshold`` (or at most ``1 - rate_threshold``) with enough
  support are flagged as near-deterministic.
- ``numeric_support``: the legitimate-class range of each numeric column, and
  the share of fraud rows that fall outside it ("fraud-only tail"). A value no
  legitimate row ever takes separates the classes by construction.
- ``derived_pairs``: for each categorical "bin" column, whether its values map
  to non-overlapping ranges of one numeric column. Such a bin is a function of
  that numeric column, so excluding only the bin does not remove the signal.

Run it from the repository root:

    python -m sentinelflow.data.signal_audit --config configs/full.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sentinelflow.data.loader import DataSettings, iter_cnp_chunks

DEFAULT_RATE_THRESHOLD = 0.99
DEFAULT_MIN_SUPPORT = 1_000
DEFAULT_MAX_INTEGER_VALUES = 50


@dataclass
class _NumericRange:
    legit_min: float = np.inf
    legit_max: float = -np.inf
    fraud_rows: int = 0
    fraud_values: list[np.ndarray] | None = None

    def update(self, values: pd.Series, labels: np.ndarray) -> None:
        clean = values.to_numpy(dtype="float64", na_value=np.nan)
        legit = clean[(labels == 0) & ~np.isnan(clean)]
        if legit.size:
            self.legit_min = min(self.legit_min, float(legit.min()))
            self.legit_max = max(self.legit_max, float(legit.max()))
        fraud = clean[(labels == 1) & ~np.isnan(clean)]
        self.fraud_rows += int(fraud.size)
        if self.fraud_values is None:
            self.fraud_values = []
        self.fraud_values.append(fraud)


def _is_categorical(series: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def screen_signals(
    settings: DataSettings,
    *,
    rate_threshold: float = DEFAULT_RATE_THRESHOLD,
    min_support: int = DEFAULT_MIN_SUPPORT,
    max_integer_values: int = DEFAULT_MAX_INTEGER_VALUES,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Screen strict-CNP rows for near-deterministic values and fraud-only ranges."""

    if not 0.5 < rate_threshold < 1.0:
        raise ValueError("rate_threshold must be between 0.5 and 1.")

    counts: dict[tuple[str, str], list[int]] = {}
    numeric: dict[str, _NumericRange] = {}
    bin_ranges: dict[tuple[str, str, str], list[float]] = {}
    categorical: list[str] | None = None
    numeric_columns: list[str] = []
    total_rows = 0
    total_fraud = 0

    for chunk in iter_cnp_chunks(settings, max_rows=max_rows):
        frame, labels = chunk.features, chunk.labels
        if categorical is None:
            categorical = [column for column in frame.columns if _is_categorical(frame[column])]
            numeric_columns = [column for column in frame.columns if column not in categorical]
        total_rows += len(labels)
        total_fraud += int(labels.sum())

        for column in numeric_columns:
            numeric.setdefault(column, _NumericRange()).update(frame[column], labels)

        grouped_columns = [
            *categorical,
            *[c for c in numeric_columns if pd.api.types.is_integer_dtype(frame[c])],
        ]
        label_series = pd.Series(labels, index=frame.index)
        for column in grouped_columns:
            keys = frame[column].astype("object").where(frame[column].notna(), "<null>")
            stats = label_series.groupby(keys.to_numpy()).agg(["size", "sum"])
            for value, row in stats.iterrows():
                entry = counts.setdefault((column, str(value)), [0, 0])
                entry[0] += int(row["size"])
                entry[1] += int(row["sum"])

        for bin_column in categorical:
            ranges = frame.groupby(bin_column, observed=True)[numeric_columns].agg(["min", "max"])
            for value, row in ranges.iterrows():
                for numeric_column in numeric_columns:
                    key = (bin_column, numeric_column, str(value))
                    current = bin_ranges.setdefault(key, [np.inf, -np.inf])
                    current[0] = min(current[0], float(row[(numeric_column, "min")]))
                    current[1] = max(current[1], float(row[(numeric_column, "max")]))

    distinct: dict[str, int] = {}
    for column, _ in counts:
        distinct[column] = distinct.get(column, 0) + 1
    low_cardinality = {
        column
        for column, total in distinct.items()
        if column in (categorical or []) or total <= max_integer_values
    }
    counts = {key: value for key, value in counts.items() if key[0] in low_cardinality}

    values = _value_rows(counts, rate_threshold, min_support)
    support = _numeric_rows(numeric)
    pairs = _derived_pairs(bin_ranges, categorical or [])
    flagged_columns = sorted({row["column"] for row in values if row["flag"]})
    tail_columns = [row["column"] for row in support if row["fraud_only_share"] >= 0.01]

    return {
        "settings": {
            "rate_threshold": rate_threshold,
            "min_support": min_support,
            "max_integer_values": max_integer_values,
            "cnp_values": list(settings.cnp_values),
        },
        "rows": total_rows,
        "fraud_rows": total_fraud,
        "fraud_rate": round(total_fraud / total_rows, 6) if total_rows else None,
        "value_screen": values,
        "numeric_support": support,
        "derived_pairs": pairs,
        "summary": {
            "near_deterministic_values": [
                f"{row['column']}={row['value']}" for row in values if row["flag"]
            ],
            "columns_with_near_deterministic_values": flagged_columns,
            "columns_with_fraud_only_tail": tail_columns,
            "derived_bins": [f"{row['bin_column']} <- {row['numeric_column']}" for row in pairs],
        },
    }


def _value_rows(
    counts: dict[tuple[str, str], list[int]],
    rate_threshold: float,
    min_support: int,
) -> list[dict[str, Any]]:
    column_totals: dict[str, int] = {}
    for (column, _), (rows, _) in counts.items():
        column_totals[column] = column_totals.get(column, 0) + rows

    result = []
    for (column, value), (rows, fraud) in counts.items():
        rate = fraud / rows
        result.append(
            {
                "column": column,
                "value": value,
                "rows": rows,
                "fraud_rows": fraud,
                "fraud_rate": round(rate, 6),
                "share_of_rows": round(rows / column_totals[column], 6),
                "flag": rows >= min_support
                and (rate >= rate_threshold or rate <= 1.0 - rate_threshold),
            }
        )
    result.sort(key=lambda row: (not row["flag"], -row["fraud_rate"], row["column"]))
    return result


def _numeric_rows(numeric: dict[str, _NumericRange]) -> list[dict[str, Any]]:
    result = []
    for column, stats in numeric.items():
        fraud = (
            np.concatenate(stats.fraud_values) if stats.fraud_values else np.array([], dtype=float)
        )
        if np.isfinite(stats.legit_min):
            outside = int(((fraud < stats.legit_min) | (fraud > stats.legit_max)).sum())
        else:
            outside = int(fraud.size)
        result.append(
            {
                "column": column,
                "legit_min": stats.legit_min if np.isfinite(stats.legit_min) else None,
                "legit_max": stats.legit_max if np.isfinite(stats.legit_max) else None,
                "fraud_min": float(fraud.min()) if fraud.size else None,
                "fraud_max": float(fraud.max()) if fraud.size else None,
                "fraud_rows_outside_legit_range": outside,
                "fraud_only_share": round(outside / fraud.size, 6) if fraud.size else 0.0,
            }
        )
    result.sort(key=lambda row: -row["fraud_only_share"])
    return result


def _derived_pairs(
    bin_ranges: dict[tuple[str, str, str], list[float]],
    categorical: list[str],
) -> list[dict[str, Any]]:
    """Bins whose values occupy disjoint ranges of one numeric column."""

    grouped: dict[tuple[str, str], list[tuple[str, float, float]]] = {}
    for (bin_column, numeric_column, value), (low, high) in bin_ranges.items():
        grouped.setdefault((bin_column, numeric_column), []).append((value, low, high))

    pairs = []
    for (bin_column, numeric_column), ranges in grouped.items():
        if bin_column not in categorical or len(ranges) < 2:
            continue
        ordered = sorted(ranges, key=lambda item: item[1])
        disjoint = all(
            previous[2] < current[1]
            for previous, current in zip(ordered, ordered[1:], strict=False)
        )
        if disjoint:
            pairs.append(
                {
                    "bin_column": bin_column,
                    "numeric_column": numeric_column,
                    "ranges": {value: [low, high] for value, low, high in ordered},
                }
            )
    return pairs


def write_signal_outputs(report: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": directory / "signal_audit.json",
        "values": directory / "value_screen.csv",
        "numeric": directory / "numeric_support.csv",
    }
    paths["report"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for key, rows in (("values", report["value_screen"]), ("numeric", report["numeric_support"])):
        with paths[key].open("w", newline="", encoding="utf-8") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    return paths


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sentinelflow.data.signal_audit",
        description="Screen strict-CNP rows for near-deterministic fraud signals.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/full.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/metrics/feature_audit"))
    parser.add_argument("--rate-threshold", type=float, default=DEFAULT_RATE_THRESHOLD)
    parser.add_argument("--min-support", type=int, default=DEFAULT_MIN_SUPPORT)
    parser.add_argument("--max-rows", type=int, default=None)
    parsed = parser.parse_args(arguments)

    from sentinelflow.config import load_config

    settings = DataSettings.from_config(load_config(parsed.config))
    report = screen_signals(
        settings,
        rate_threshold=parsed.rate_threshold,
        min_support=parsed.min_support,
        max_rows=parsed.max_rows,
    )
    paths = write_signal_outputs(report, parsed.output)

    summary = report["summary"]
    print(f"Strict-CNP rows: {report['rows']:,} (fraud rate {report['fraud_rate']})")
    print("Near-deterministic values:", ", ".join(summary["near_deterministic_values"]) or "none")
    print("Fraud-only numeric tails:", ", ".join(summary["columns_with_fraud_only_tail"]) or "none")
    print("Bins derived from a numeric column:", ", ".join(summary["derived_bins"]) or "none")
    for path in paths.values():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
