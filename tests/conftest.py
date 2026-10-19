"""Shared fixtures: a small synthetic Parquet file shaped like CreditTransAct."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sentinelflow.data.loader import DataSettings

CHANNELS = np.array(["card_present", "cnp_online", "cnp_moto", "contactless"])


def make_frame(rows: int = 2_000, seed: int = 7) -> pd.DataFrame:
    """Synthetic transactions with IDs, mixed dtypes, nulls and every channel."""

    rng = np.random.default_rng(seed)
    channel = CHANNELS[rng.integers(0, len(CHANNELS), size=rows)]
    is_fraud = (rng.random(rows) < 0.15).astype(np.int8)
    amount = rng.gamma(2.0, 40.0, size=rows) + is_fraud * 60.0
    amount[rng.random(rows) < 0.05] = np.nan
    merchant_category = rng.choice(["grocery", "travel", "electronics", "fuel"], size=rows).astype(
        object
    )
    merchant_category[rng.random(rows) < 0.05] = None

    return pd.DataFrame(
        {
            "transaction_id": [f"t{i:07d}" for i in range(rows)],
            "customer_id": [f"c{i % 300:05d}" for i in range(rows)],
            "amount": amount,
            "hour": rng.integers(0, 24, size=rows),
            "is_international": rng.random(rows) < 0.2,
            "merchant_category": merchant_category,
            "device_type": rng.choice(["mobile", "desktop", "tablet"], size=rows),
            "merchant_name": [f"m{v}" for v in rng.integers(0, 600, size=rows)],
            "card_present_cnp": channel,
            "is_fraud": is_fraud,
        }
    )


def write_parquet(frame: pd.DataFrame, path: Path, row_group_size: int = 250) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, path, row_group_size=row_group_size)
    return path


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    return make_frame()


@pytest.fixture
def synthetic_parquet(tmp_path: Path, synthetic_frame: pd.DataFrame) -> Path:
    return write_parquet(synthetic_frame, tmp_path / "credit_card_fraud.parquet")


@pytest.fixture
def data_settings(synthetic_parquet: Path) -> DataSettings:
    return DataSettings(dataset_path=synthetic_parquet, warmup_rows=300)
