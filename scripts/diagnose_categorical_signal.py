"""Diagnose categorical predictive signal in the pre-drift evaluation data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sentinelflow.config import load_config
from sentinelflow.data.loader import (
    DataSettings,
    load_warmup_and_stream,
)
from sentinelflow.data.preprocessing import FraudPreprocessor
from sentinelflow.streaming.drift_injector import (
    DriftInjector,
    DriftSettings,
)
from sentinelflow.streaming.label_delay import (
    StreamSettings,
    batch_chunks,
)

EVALUATION_ROWS = 1_500_000
OUTPUT_DIR = Path("results/feature_diagnostic")


def main() -> None:
    config = load_config(Path("configs/full.yaml"))

    data_settings = DataSettings.from_config(config)

    stream_settings = StreamSettings.from_config(
        data=config.section("data"),
        stream=config.section("stream"),
        seed=config.seed,
    )

    drift_settings = DriftSettings.from_config(config.section("drift"))

    maximum_loaded_rows = config.warmup_rows + EVALUATION_ROWS

    warmup, evaluation_chunks = load_warmup_and_stream(
        data_settings,
        max_rows=maximum_loaded_rows,
    )

    preprocessor = FraudPreprocessor.from_config(config)

    preprocessor.fit(warmup.features)

    if preprocessor.manifest is None:
        raise RuntimeError("Preprocessor did not produce a feature manifest.")

    categorical_columns = list(preprocessor.manifest.categorical)

    # Drift disabled so we measure the original dataset signal.
    no_drift_settings = DriftSettings(
        enabled=False,
        start_fraction=drift_settings.start_fraction,
        end_fraction=drift_settings.end_fraction,
        max_intensity=drift_settings.max_intensity,
        random_seed=drift_settings.random_seed,
        shift_scale=drift_settings.shift_scale,
    )

    drift = DriftInjector(no_drift_settings)

    batches = batch_chunks(
        chunks=evaluation_chunks,
        total_rows=EVALUATION_ROWS,
        settings=stream_settings,
        drift=drift,
    )

    # Use only the rows corresponding to the experiment's pre-drift phase.
    feature_parts: list[pd.DataFrame] = []
    label_parts: list[np.ndarray] = []

    for batch in batches:
        # With drift disabled the phase is always pre_drift,
        # so determine the experimental phase from progress instead.
        if batch.progress >= drift_settings.start_fraction:
            break

        feature_parts.append(batch.features[categorical_columns].copy())

        label_parts.append(
            np.asarray(
                batch.labels,
                dtype=int,
            ).copy()
        )

    if not feature_parts:
        raise RuntimeError("No pre-drift evaluation rows were collected.")

    features = pd.concat(
        feature_parts,
        ignore_index=True,
    )

    labels = np.concatenate(label_parts)

    rows: list[dict[str, object]] = []

    detail_rows: list[dict[str, object]] = []

    for column in categorical_columns:
        frame = pd.DataFrame(
            {
                "value": (
                    features[column]
                    .astype("object")
                    .where(
                        features[column].notna(),
                        "__missing__",
                    )
                    .astype(str)
                ),
                "is_fraud": labels,
            }
        )

        grouped = (
            frame.groupby(
                "value",
                dropna=False,
            )["is_fraud"]
            .agg(
                fraud_rate="mean",
                row_count="size",
                fraud_count="sum",
            )
            .reset_index()
        )

        grouped = grouped.sort_values("fraud_rate")

        lowest = grouped.iloc[0]
        highest = grouped.iloc[-1]

        overall_rate = float(frame["is_fraud"].mean())

        rows.append(
            {
                "feature": column,
                "category_count": len(grouped),
                "overall_fraud_rate": overall_rate,
                "min_fraud_rate": float(lowest["fraud_rate"]),
                "max_fraud_rate": float(highest["fraud_rate"]),
                "fraud_rate_range": float(highest["fraud_rate"] - lowest["fraud_rate"]),
                "lowest_fraud_category": str(lowest["value"]),
                "highest_fraud_category": str(highest["value"]),
                "lowest_category_rows": int(lowest["row_count"]),
                "highest_category_rows": int(highest["row_count"]),
            }
        )

        for _, category in grouped.iterrows():
            detail_rows.append(
                {
                    "feature": column,
                    "category": str(category["value"]),
                    "row_count": int(category["row_count"]),
                    "fraud_count": int(category["fraud_count"]),
                    "fraud_rate": float(category["fraud_rate"]),
                }
            )

    summary = pd.DataFrame(rows).sort_values(
        "fraud_rate_range",
        ascending=False,
    )

    detail = pd.DataFrame(detail_rows)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = OUTPUT_DIR / "categorical_signal_summary.csv"

    detail_path = OUTPUT_DIR / "categorical_signal_detail.csv"

    summary.to_csv(
        summary_path,
        index=False,
    )

    detail.to_csv(
        detail_path,
        index=False,
    )

    print(f"Pre-drift rows analyzed: {len(labels):,}")

    print(f"Overall fraud rate: {labels.mean():.6f}")

    print()
    print("=== Strongest categorical signals ===")

    print(
        summary[
            [
                "feature",
                "category_count",
                "min_fraud_rate",
                "max_fraud_rate",
                "fraud_rate_range",
                "lowest_fraud_category",
                "highest_fraud_category",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print()
    print(f"Saved: {summary_path}")
    print(f"Saved: {detail_path}")


if __name__ == "__main__":
    main()
