"""Memory-safe strict-CNP loading for the CreditTransAct Parquet file (work package 2).

The source file holds 15 million rows, so nothing here reads it whole. Row groups
are read one at a time, filtered to the strict Card-Not-Present channel values,
and the binary ``is_fraud`` label is split off *before* any feature frame leaves
this module. Direct identifiers are dropped at the same point, so no caller can
accidentally feed them to a model.

What this module hands to the stream layer (work package 3) is a ``CNPChunk``:
a raw, untransformed feature frame plus aligned labels and global CNP row
positions. Stream batching, drift injection and preprocessing happen downstream.

Row order is deterministic. The CreditTransAct file is sorted by customer
segment (A, B, C, D) and each row group holds a single segment, so reading in
file order would make the warm-up window 100% segment A and add natural segment
shifts to the stream. When ``row_group_shuffle`` is on (``stream.row_group_shuffle``
in the config), row groups are visited in a permutation seeded by
``project.random_seed``; rows keep their file order inside each row group. The
warm-up window is "the first ``warmup_rows`` strict-CNP rows in that order".
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DEFAULT_IDENTIFIER_COLUMNS: tuple[str, ...] = ("customer_id", "transaction_id")


class DatasetError(ValueError):
    """Base class for dataset problems that must stop an experiment."""


class DatasetNotFoundError(DatasetError, FileNotFoundError):
    """Raised when the configured Parquet file does not exist."""


class SchemaError(DatasetError):
    """Raised when required columns are missing from the Parquet schema."""


class TargetValueError(DatasetError):
    """Raised when the target column holds anything other than binary 0/1 values."""


class CorruptDatasetError(DatasetError):
    """Raised when the file is not a complete Parquet file (usually a cut-off download)."""


PARQUET_MAGIC = b"PAR1"


def check_parquet_complete(path: Path) -> None:
    """Fail clearly if the file lacks the Parquet header or footer magic bytes."""

    size = path.stat().st_size
    with path.open("rb") as handle:
        head = handle.read(4)
        if size >= 8:
            handle.seek(-4, 2)
        tail = handle.read(4)

    if head != PARQUET_MAGIC:
        raise CorruptDatasetError(
            f"{path.name} is not a Parquet file (size {size:,} bytes). You may have saved "
            "the CSV, a zip archive or a web page; download credit_card_fraud.parquet again."
        )
    if size < 8 or tail != PARQUET_MAGIC:
        raise CorruptDatasetError(
            f"{path.name} is incomplete: it has a Parquet header but no footer "
            f"(size {size:,} bytes). The download was cut off; delete it and download "
            "it again, then verify the SHA-256 (see docs/DATASET.md)."
        )


@dataclass(frozen=True)
class DataSettings:
    """The data-layer slice of the project configuration."""

    dataset_path: Path
    target_column: str = "is_fraud"
    cnp_column: str = "card_present_cnp"
    cnp_values: tuple[str, ...] = ("cnp_online", "cnp_moto")
    drop_columns: tuple[str, ...] = DEFAULT_IDENTIFIER_COLUMNS
    warmup_rows: int = 500_000
    row_group_shuffle: bool = False
    shuffle_seed: int | None = None

    def __post_init__(self) -> None:
        if not self.target_column:
            raise ValueError("target_column cannot be empty.")
        if not self.cnp_column:
            raise ValueError("cnp_column cannot be empty.")
        if not self.cnp_values:
            raise ValueError("cnp_values cannot be empty.")
        if self.warmup_rows <= 0:
            raise ValueError("warmup_rows must be greater than zero.")
        if self.target_column in self.drop_columns:
            raise ValueError(
                "The target column is separated, not dropped; remove it from drop_columns."
            )

    @classmethod
    def from_config(cls, config: Any) -> DataSettings:
        """Build settings from a :class:`sentinelflow.config.ProjectConfig`."""

        data = config.section("data")
        return cls(
            dataset_path=config.dataset_path,
            target_column=config.target_column,
            cnp_column=config.cnp_column,
            cnp_values=config.cnp_values,
            drop_columns=configured_drop_columns(data),
            warmup_rows=config.warmup_rows,
            row_group_shuffle=bool(config.section("stream").get("row_group_shuffle", False)),
            shuffle_seed=config.seed,
        )

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (self.target_column, self.cnp_column)


@dataclass(frozen=True)
class CNPChunk:
    """A bounded slice of strict-CNP rows, still untransformed.

    Attributes:
        row_group: Source Parquet row group this chunk came from.
        cnp_rows: Global strict-CNP row positions (0-based, in stream order).
        features: Raw feature frame. Target and identifiers are already removed;
            the CNP channel column is kept as an ordinary categorical feature.
        labels: Binary ``is_fraud`` labels as ``int8``, aligned with ``features``.
    """

    row_group: int
    cnp_rows: np.ndarray
    features: pd.DataFrame
    labels: np.ndarray

    def __post_init__(self) -> None:
        if not (len(self.cnp_rows) == len(self.features) == len(self.labels)):
            raise ValueError("cnp_rows, features and labels must have the same length.")

    @property
    def row_count(self) -> int:
        return len(self.labels)

    def slice(self, start: int, stop: int) -> CNPChunk:
        return CNPChunk(
            row_group=self.row_group,
            cnp_rows=self.cnp_rows[start:stop],
            features=self.features.iloc[start:stop].reset_index(drop=True),
            labels=self.labels[start:stop],
        )


@dataclass(frozen=True)
class WarmupSet:
    """The concatenated warm-up window used to fit preprocessing and static models."""

    features: pd.DataFrame
    labels: np.ndarray
    cnp_rows: np.ndarray

    @property
    def row_count(self) -> int:
        return len(self.labels)


def configured_drop_columns(data: dict[str, Any]) -> tuple[str, ...]:
    """Identifier columns to exclude; the safe defaults apply when the key is absent."""

    if data.get("drop_columns") is None:
        return DEFAULT_IDENTIFIER_COLUMNS
    return tuple(str(column) for column in data["drop_columns"])


def open_parquet(settings: DataSettings) -> pq.ParquetFile:
    """Open the dataset and check that the required columns exist."""

    path = Path(settings.dataset_path)
    if not path.is_file():
        raise DatasetNotFoundError(
            f"Dataset not found at {path}. Download credit_card_fraud.parquet from "
            "https://data.mendeley.com/datasets/y64bbnm2s3/1 and place it there "
            "(see docs/DATASET.md)."
        )

    check_parquet_complete(path)
    try:
        parquet = pq.ParquetFile(path)
    except pa.ArrowInvalid as error:
        raise CorruptDatasetError(f"{path.name} could not be read as Parquet: {error}") from error
    missing = [name for name in settings.required_columns if name not in parquet.schema_arrow.names]
    if missing:
        raise SchemaError(f"Required columns missing from {path.name}: {', '.join(missing)}")
    return parquet


def feature_columns(schema_names: Sequence[str], settings: DataSettings) -> list[str]:
    """Columns that may reach a model: everything except the target and dropped IDs."""

    excluded = {settings.target_column, *settings.drop_columns}
    return [name for name in schema_names if name not in excluded]


def row_group_order(num_row_groups: int, *, seed: int | None = None) -> list[int]:
    """File order by default; a seeded permutation when ``seed`` is given.

    Shuffling row groups changes which rows form the warm-up window, so the
    stream owner and the data owner must agree on one place to apply it.
    """

    order = np.arange(num_row_groups)
    if seed is not None:
        order = np.random.default_rng(seed).permutation(num_row_groups)
    return [int(group) for group in order]


def validate_binary_target(values: pa.ChunkedArray | pa.Array, column: str) -> np.ndarray:
    """Return labels as ``int8``, raising if they are null or not exactly 0/1."""

    if values.null_count:
        raise TargetValueError(f"{column} contains {values.null_count} null values.")

    labels = values.to_numpy(zero_copy_only=False)
    if labels.dtype == bool:
        return labels.astype(np.int8)

    unique = np.unique(labels)
    if not set(unique.tolist()).issubset({0, 1}):
        shown = ", ".join(str(value) for value in unique[:10])
        raise TargetValueError(f"{column} must be binary 0/1; found values: {shown}")
    return labels.astype(np.int8)


def cnp_mask(table: pa.Table, settings: DataSettings) -> pa.ChunkedArray:
    """Boolean mask selecting strict-CNP rows; null channel values are excluded."""

    channel = table.column(settings.cnp_column)
    if pa.types.is_dictionary(channel.type):
        channel = channel.cast(pa.string())
    mask = pc.is_in(channel, value_set=pa.array(list(settings.cnp_values), type=channel.type))
    return pc.fill_null(mask, False)


def iter_cnp_chunks(
    settings: DataSettings,
    *,
    order: Sequence[int] | None = None,
    max_rows: int | None = None,
) -> Iterator[CNPChunk]:
    """Yield strict-CNP chunks one row group at a time.

    Args:
        settings: Data settings (path, columns, CNP values, identifiers).
        order: Row-group visiting order; file order when omitted.
        max_rows: Stop after this many strict-CNP rows (smoke runs).
    """

    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when given.")

    parquet = open_parquet(settings)
    columns = feature_columns(parquet.schema_arrow.names, settings)
    read_columns = [*columns, settings.target_column]
    if order is not None:
        groups = list(order)
    elif settings.row_group_shuffle:
        groups = row_group_order(parquet.num_row_groups, seed=settings.shuffle_seed)
    else:
        groups = list(range(parquet.num_row_groups))

    emitted = 0
    for group in groups:
        table = parquet.read_row_group(group, columns=read_columns)
        table = table.filter(cnp_mask(table, settings))
        if table.num_rows == 0:
            continue

        if max_rows is not None:
            remaining = max_rows - emitted
            if remaining <= 0:
                return
            table = table.slice(0, remaining)

        target = table.column(settings.target_column)
        labels = validate_binary_target(target, settings.target_column)
        features = table.select(columns).to_pandas()
        cnp_rows = np.arange(emitted, emitted + table.num_rows, dtype=np.int64)
        emitted += table.num_rows

        yield CNPChunk(row_group=group, cnp_rows=cnp_rows, features=features, labels=labels)

        if max_rows is not None and emitted >= max_rows:
            return


def split_warmup(
    chunks: Iterator[CNPChunk],
    warmup_rows: int,
) -> tuple[WarmupSet, Iterator[CNPChunk]]:
    """Collect the first ``warmup_rows`` rows; return them and an iterator over the rest.

    The evaluation iterator resumes exactly where warm-up stopped, splitting the
    boundary chunk, so no row appears in both windows and none is skipped.
    """

    if warmup_rows <= 0:
        raise ValueError("warmup_rows must be positive.")

    parts: list[CNPChunk] = []
    collected = 0
    leftover: CNPChunk | None = None

    for chunk in chunks:
        needed = warmup_rows - collected
        if chunk.row_count <= needed:
            parts.append(chunk)
            collected += chunk.row_count
        else:
            parts.append(chunk.slice(0, needed))
            leftover = chunk.slice(needed, chunk.row_count)
            collected += needed
        if collected >= warmup_rows:
            break

    if collected < warmup_rows:
        raise DatasetError(
            f"Only {collected} strict-CNP rows available; warm-up needs {warmup_rows}."
        )

    warmup = WarmupSet(
        features=pd.concat([part.features for part in parts], ignore_index=True),
        labels=np.concatenate([part.labels for part in parts]),
        cnp_rows=np.concatenate([part.cnp_rows for part in parts]),
    )

    def evaluation() -> Iterator[CNPChunk]:
        if leftover is not None and leftover.row_count:
            yield leftover
        yield from chunks

    return warmup, evaluation()


def load_warmup_and_stream(
    settings: DataSettings,
    *,
    order: Sequence[int] | None = None,
    max_rows: int | None = None,
) -> tuple[WarmupSet, Iterator[CNPChunk]]:
    """Convenience wrapper: open the file, then split warm-up from evaluation."""

    chunks = iter_cnp_chunks(settings, order=order, max_rows=max_rows)
    return split_warmup(chunks, settings.warmup_rows)


def frame_hash(features: pd.DataFrame, labels: np.ndarray | None = None) -> str:
    """SHA-256 of a frame's values, column names and (optionally) labels.

    Used as reproducibility evidence: the same seed and configuration must
    produce identical hashes for every batch.
    """

    digest = hashlib.sha256()
    digest.update("\x1f".join(map(str, features.columns)).encode("utf-8"))
    row_hashes = pd.util.hash_pandas_object(features, index=False).to_numpy()
    digest.update(row_hashes.tobytes())
    if labels is not None:
        digest.update(np.asarray(labels, dtype=np.int8).tobytes())
    return digest.hexdigest()
