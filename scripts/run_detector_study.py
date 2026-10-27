"""Run the ADWIN/DDM detector study over several seeds and drift scenarios.

Examples::

    # No-drift control and the current controlled drift, three seeds
    python scripts/run_detector_study.py --seeds 543 544 545

    # Add a scenario by overriding keys of the ``drift`` config section
    python scripts/run_detector_study.py --seeds 543 544 545 \\
        --scenario no_drift:enabled=false \\
        --scenario controlled_drift:enabled=true \\
        --scenario degrading_drift:enabled=true,mode=degrading
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from sentinelflow.config import load_config
from sentinelflow.models.detector_study import parse_scenario, run_detector_study


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/full.yaml"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[543, 544, 545])
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="name:key=value[,key=value] overrides for the drift section. Repeatable.",
    )
    parser.add_argument(
        "--max-evaluation-rows",
        type=int,
        default=None,
        help="Limit evaluation rows for a smoke run; omit for the full audited stream.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: <results>/metrics/experiment/detector_study).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    scenarios = None if args.scenario is None else dict(map(parse_scenario, args.scenario))
    # Inside the git-ignored generated-artifacts directory.
    output = args.output or config.results_path / "metrics" / "experiment" / "detector_study"

    study = run_detector_study(
        config,
        seeds=args.seeds,
        scenarios=scenarios,
        max_evaluation_rows=args.max_evaluation_rows,
        output_directory=output,
    )

    for run in study.runs:
        print(
            json.dumps(
                {
                    "scenario": run.scenario,
                    "seed": run.seed,
                    "drift_start_batch": run.drift_start_batch,
                    "last_scored_batch": run.last_scored_batch,
                    "alarm_counts": run.alarm_counts(),
                    "recovery_violations": len(run.recovery_violations),
                }
            )
        )
    print(f"Artifacts written to {output}")

    return 1 if any(run.recovery_violations for run in study.runs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
