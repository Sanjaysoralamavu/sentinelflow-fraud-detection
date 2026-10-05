from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from sentinelflow.config import ProjectConfig, load_config


def build_parser() -> argparse.ArgumentParser:
    """Build the SentinelFlow command-line parser."""

    parser = argparse.ArgumentParser(
        prog="sentinelflow",
        description=("Adaptive stream learning for Card-Not-Present credit-card fraud detection."),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    show_config_parser = subparsers.add_parser(
        "show-config",
        help="Load, validate, and display the resolved configuration.",
    )
    show_config_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/full.yaml"),
        help="Path to the YAML configuration file.",
    )
    show_config_parser.set_defaults(handler=_show_config)

    return parser


def _configuration_summary(config: ProjectConfig) -> dict[str, object]:
    return {
        "project": config.raw["project"]["name"],
        "random_seed": config.seed,
        "dataset_path": str(config.dataset_path),
        "results_path": str(config.results_path),
        "target_column": config.target_column,
        "cnp_column": config.cnp_column,
        "cnp_values": list(config.cnp_values),
        "warmup_rows": config.warmup_rows,
        "batch_size": config.batch_size,
        "label_delay_batches": config.label_delay_batches,
        "drift": config.section("drift"),
    }


def _show_config(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    summary = _configuration_summary(config)

    print(json.dumps(summary, indent=2, sort_keys=True))

    return 0


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the SentinelFlow command-line interface."""

    parser = build_parser()
    parsed_arguments = parser.parse_args(arguments)

    return int(parsed_arguments.handler(parsed_arguments))


if __name__ == "__main__":
    raise SystemExit(main())
