"""Full-versus-reduced feature probe with the project's own pipeline (issue #9).

For each configuration this fits the configured preprocessor and the frozen
static baselines on the warm-up window, scores the next ``--eval-rows`` raw
evaluation rows (before any drift is injected), and writes:

- ``<output>/manifests/<config>.json``: the feature manifest for that config
- ``<output>/ablation_probe.csv`` and ``.json``: one metrics row per config and model

It isolates the feature set: stream, drift and adaptive learning are the
runner's job (``sentinelflow run --config ...``), so this answers only "how
much do the excluded features matter to a model trained on the same rows".

    python -m sentinelflow.data.ablation_probe \\
        --configs configs/full.yaml configs/ablation_reduced.yaml configs/ablation_strict.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sentinelflow.config import load_config
from sentinelflow.data.loader import DataSettings, load_warmup_and_stream
from sentinelflow.data.preprocessing import FraudPreprocessor

DEFAULT_EVAL_ROWS = 500_000


def _collect(chunks: Any, rows: int) -> tuple[pd.DataFrame, np.ndarray]:
    frames, labels, taken = [], [], 0
    for chunk in chunks:
        need = rows - taken
        part = chunk.slice(0, min(need, chunk.row_count))
        frames.append(part.features)
        labels.append(part.labels)
        taken += part.row_count
        if taken >= rows:
            break
    return pd.concat(frames, ignore_index=True), np.concatenate(labels)


def probe(config_path: Path, *, eval_rows: int, output: Path) -> list[dict[str, Any]]:
    from sentinelflow.suite import ModelSuite

    started = time.perf_counter()
    config = load_config(config_path)
    warmup, stream = load_warmup_and_stream(DataSettings.from_config(config))
    evaluation, labels = _collect(stream, eval_rows)

    preprocessor = FraudPreprocessor.from_config(config)
    train = preprocessor.fit_transform(warmup.features)
    assert preprocessor.manifest is not None
    manifest_path = output / "manifests" / f"{config_path.stem}.json"
    preprocessor.manifest.write_json(manifest_path)

    static = ModelSuite.from_config(config).static_models
    static.fit(train, warmup.labels)
    threshold = float(config.section("models")["threshold"])
    scores = static.predict_probabilities(preprocessor.transform(evaluation))

    rows = []
    for model, probability in scores.items():
        predicted = probability >= threshold
        rows.append(
            {
                "config": config_path.stem,
                "model": model,
                "features_in": len(preprocessor.manifest.included),
                "features_out": preprocessor.manifest.n_output_features,
                "excluded": len(preprocessor.manifest.excluded),
                "eval_rows": len(labels),
                "eval_fraud_rate": round(float(labels.mean()), 6),
                "roc_auc": round(float(roc_auc_score(labels, probability)), 6),
                "pr_auc": round(float(average_precision_score(labels, probability)), 6),
                "precision": round(float(precision_score(labels, predicted, zero_division=0)), 6),
                "recall": round(float(recall_score(labels, predicted, zero_division=0)), 6),
                "f1": round(float(f1_score(labels, predicted, zero_division=0)), 6),
                "seconds": round(time.perf_counter() - started, 1),
            }
        )
    return rows


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sentinelflow.data.ablation_probe",
        description="Compare feature configurations with the project's static baselines.",
    )
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-rows", type=int, default=DEFAULT_EVAL_ROWS)
    parser.add_argument("--output", type=Path, default=Path("results/metrics/feature_audit"))
    parsed = parser.parse_args(arguments)

    if parsed.eval_rows <= 0:
        parser.error("--eval-rows must be positive")

    results: list[dict[str, Any]] = []
    for path in parsed.configs:
        rows = probe(path, eval_rows=parsed.eval_rows, output=parsed.output)
        results.extend(rows)
        for row in rows:
            print(
                f"{row['config']:18s} {row['model']:22s} features={row['features_in']:2d} "
                f"ROC-AUC={row['roc_auc']:.4f} PR-AUC={row['pr_auc']:.4f} "
                f"P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f}"
            )

    parsed.output.mkdir(parents=True, exist_ok=True)
    (parsed.output / "ablation_probe.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    with (parsed.output / "ablation_probe.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {parsed.output / 'ablation_probe.csv'} and manifests/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
