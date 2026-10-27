"""Diagnose whether controlled drift changes fraud/legitimate feature separation."""

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
OUTPUT_DIR = Path("results/drift_diagnostic")


def standardized_mean_difference(
    legitimate: np.ndarray,
    fraud: np.ndarray,
) -> float:
    """Return absolute standardized mean difference between two classes."""

    legitimate = np.asarray(legitimate, dtype=float)
    fraud = np.asarray(fraud, dtype=float)

    legitimate = legitimate[np.isfinite(legitimate)]
    fraud = fraud[np.isfinite(fraud)]

    if len(legitimate) == 0 or len(fraud) == 0:
        return float("nan")

    legitimate_var = float(np.var(legitimate, ddof=0))
    fraud_var = float(np.var(fraud, ddof=0))

    pooled_std = np.sqrt(
        (legitimate_var + fraud_var) / 2.0
    )

    if pooled_std == 0.0:
        return 0.0

    return abs(
        float(np.mean(fraud))
        - float(np.mean(legitimate))
    ) / pooled_std


def main() -> None:
    config = load_config(
        Path("configs/full.yaml")
    )

    data_settings = DataSettings.from_config(
        config
    )

    stream_settings = StreamSettings.from_config(
        data=config.section("data"),
        stream=config.section("stream"),
        seed=config.seed,
    )

    drift_settings = DriftSettings.from_config(
        config.section("drift")
    )

    maximum_loaded_rows = (
        config.warmup_rows
        + EVALUATION_ROWS
    )

    warmup, evaluation_chunks = (
        load_warmup_and_stream(
            data_settings,
            max_rows=maximum_loaded_rows,
        )
    )

    # Fit preprocessing exactly as the real experiment does.
    preprocessor = FraudPreprocessor.from_config(
        config
    )

    preprocessor.fit(
        warmup.features
    )

    if preprocessor.manifest is None:
        raise RuntimeError(
            "Preprocessor did not produce a feature manifest."
        )

    numeric_columns = list(
        preprocessor.manifest.numeric
    )

    print(
        f"Numeric features: {len(numeric_columns)}"
    )

    for column in numeric_columns:
        print(f"  {column}")

    # Two injectors with identical settings/seed.
    #
    # drift_on reproduces the controlled drift.
    # drift_off keeps exactly the same stream rows unchanged.
    drift_on = DriftInjector(
        drift_settings
    )

    no_drift_settings = DriftSettings(
        enabled=False,
        start_fraction=drift_settings.start_fraction,
        end_fraction=drift_settings.end_fraction,
        max_intensity=drift_settings.max_intensity,
        random_seed=drift_settings.random_seed,
        shift_scale=drift_settings.shift_scale,
    )

    drift_off = DriftInjector(
        no_drift_settings
    )

    # Build the no-drift stream first from the same source chunks.
    #
    # We materialize only 1.5M evaluation rows, which keeps the
    # diagnostic deterministic and aligned with the experiment.
    raw_parts: list[pd.DataFrame] = []
    label_parts: list[np.ndarray] = []

    loaded = 0

    for chunk in evaluation_chunks:
        remaining = (
            EVALUATION_ROWS - loaded
        )

        if remaining <= 0:
            break

        take = min(
            chunk.row_count,
            remaining,
        )

        raw_parts.append(
            chunk.features.iloc[:take].copy()
        )

        label_parts.append(
            np.asarray(
                chunk.labels[:take]
            ).copy()
        )

        loaded += take

    raw_features = pd.concat(
        raw_parts,
        ignore_index=True,
    )

    labels = np.concatenate(
        label_parts
    )

    if len(raw_features) != EVALUATION_ROWS:
        raise RuntimeError(
            "Diagnostic did not load the expected "
            f"{EVALUATION_ROWS} evaluation rows."
        )

    # Recreate identical batches twice:
    # one control stream and one drifted stream.
    from sentinelflow.data.loader import CNPChunk

    diagnostic_chunk = CNPChunk(
        row_group=-1,
        cnp_rows=np.arange(
            EVALUATION_ROWS
        ),
        features=raw_features,
        labels=labels,
    )

    control_batches = list(
        batch_chunks(
            chunks=iter([diagnostic_chunk]),
            total_rows=EVALUATION_ROWS,
            settings=stream_settings,
            drift=drift_off,
        )
    )

    drifted_batches = list(
        batch_chunks(
            chunks=iter([diagnostic_chunk]),
            total_rows=EVALUATION_ROWS,
            settings=stream_settings,
            drift=drift_on,
        )
    )

    if len(control_batches) != len(drifted_batches):
        raise RuntimeError(
            "Control and drifted streams produced "
            "different batch counts."
        )

    rows: list[dict[str, object]] = []

    for control, drifted in zip(
        control_batches,
        drifted_batches,
        strict=True,
    ):
        if not np.array_equal(
            control.labels,
            drifted.labels,
        ):
            raise RuntimeError(
                "Control and drifted labels are not aligned."
            )

        labels_batch = control.labels

        legitimate_mask = (
            labels_batch == 0
        )

        fraud_mask = (
            labels_batch == 1
        )

        for column in numeric_columns:
            control_values = (
                control.features[column]
                .to_numpy(dtype=float)
            )

            drifted_values = (
                drifted.features[column]
                .to_numpy(dtype=float)
            )

            control_separation = (
                standardized_mean_difference(
                    control_values[
                        legitimate_mask
                    ],
                    control_values[
                        fraud_mask
                    ],
                )
            )

            drifted_separation = (
                standardized_mean_difference(
                    drifted_values[
                        legitimate_mask
                    ],
                    drifted_values[
                        fraud_mask
                    ],
                )
            )

            rows.append(
                {
                    "batch": control.batch_id,
                    "phase": drifted.phase,
                    "feature": column,
                    "control_separation": (
                        control_separation
                    ),
                    "drifted_separation": (
                        drifted_separation
                    ),
                    "separation_change": (
                        drifted_separation
                        - control_separation
                    ),
                }
            )

    detail = pd.DataFrame(rows)

    summary = (
        detail.groupby(
            ["phase", "feature"],
            as_index=False,
        )
        .agg(
            control_separation=(
                "control_separation",
                "mean",
            ),
            drifted_separation=(
                "drifted_separation",
                "mean",
            ),
            separation_change=(
                "separation_change",
                "mean",
            ),
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    detail_path = (
        OUTPUT_DIR
        / "feature_separation_by_batch.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "feature_separation_summary.csv"
    )

    detail.to_csv(
        detail_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=== Drifted phase ===")

    drifted_summary = (
        summary[
            summary["phase"] == "drifted"
        ]
        .sort_values(
            "separation_change",
            ascending=False,
        )
    )

    print(
        drifted_summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved: {detail_path}"
    )

    print(
        f"Saved: {summary_path}"
    )


if __name__ == "__main__":
    main()