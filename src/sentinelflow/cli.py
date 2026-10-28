from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from sentinelflow.config import ProjectConfig, load_config
from sentinelflow.evaluation.pipeline import generate_evaluation_outputs
from sentinelflow.evaluation.recorder import EvaluationRecorder
from sentinelflow.experiment import (
    run_experiment,
    write_experiment_artifacts,
)


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

    run_parser = subparsers.add_parser(
        "run",
        help="Run the end-to-end SentinelFlow experiment.",
    )
    run_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/full.yaml"),
        help="Path to the YAML configuration file.",
    )
    run_parser.add_argument(
        "--max-evaluation-rows",
        type=int,
        default=None,
        help=(
            "Limit evaluation rows for a smoke run. "
            "Omit this option to process the complete audited stream."
        ),
    )
    run_parser.set_defaults(handler=_run_experiment)

    return parser


def _configuration_summary(
    config: ProjectConfig,
) -> dict[str, object]:
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


# def _run_experiment(arguments: argparse.Namespace) -> int:
#     config = load_config(arguments.config)

#     result = run_experiment(
#         config,
#         max_evaluation_rows=arguments.max_evaluation_rows,
#     )

#     output_directory = config.results_path / "metrics" / "experiment"

#     artifact_paths = write_experiment_artifacts(
#         result,
#         config,
#         output_directory,
#     )

#     result_data = result.to_dict()
#     output = {
#         "summary": result_data["summary"],
#         "drift_events": result_data["drift_events"],
#         "artifacts": {name: str(path) for name, path in artifact_paths.items()},
#     }

#     print(json.dumps(output, indent=2, sort_keys=True))


#     return 0
def _run_experiment(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)

    recorder = EvaluationRecorder.from_config(config)

    result = run_experiment(
        config,
        recorder=recorder,
        max_evaluation_rows=arguments.max_evaluation_rows,
    )

    output_directory = config.results_path / "metrics" / "experiment"

    artifact_paths = write_experiment_artifacts(
        result,
        config,
        output_directory,
    )

    evaluation_output_directory = config.results_path / "metrics" / "evaluation"

    evaluation_paths = generate_evaluation_outputs(
        recorder,
        result,
        evaluation_output_directory,
    )

    result_data = result.to_dict()

    output = {
        "summary": result_data["summary"],
        "drift_events": result_data["drift_events"],
        "artifacts": {name: str(path) for name, path in artifact_paths.items()},
        "evaluation_artifacts": {name: str(path) for name, path in evaluation_paths.items()},
    }

    print(json.dumps(output, indent=2, sort_keys=True))

    return 0


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the SentinelFlow command-line interface."""

    parser = build_parser()
    parsed_arguments = parser.parse_args(arguments)

    return int(parsed_arguments.handler(parsed_arguments))


if __name__ == "__main__":
    raise SystemExit(main())
