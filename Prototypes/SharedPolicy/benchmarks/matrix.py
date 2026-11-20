"""Plan and execute repeated benchmark trials in fresh worker processes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

from benchmarks.benchmark import (
    TrialSpec,
    load_experiment_settings,
    run_benchmark_trial,
    write_report,
)
from benchmarks.models import load_benchmark_config
from benchmarks.runtime import SUPPORTED_ARCHITECTURES


DEFAULT_MEASURED_ARCHITECTURES = ("C", "D")


def _validate_architectures(
    architectures: Sequence[str],
) -> tuple[str, ...]:
    selected = tuple(architectures)
    if not selected:
        raise ValueError("at least one measured architecture is required")
    if len(set(selected)) != len(selected):
        raise ValueError("measured architectures must be unique")
    unsupported = sorted(set(selected) - set(SUPPORTED_ARCHITECTURES))
    if unsupported:
        raise ValueError(f"unsupported measured architectures: {unsupported}")
    return selected


def build_matrix_specs(
    config_path: str | Path,
    measured_architectures: Sequence[str] = DEFAULT_MEASURED_ARCHITECTURES,
) -> tuple[TrialSpec, ...]:
    """Expand the configured profile/workload matrix into unique trial specs."""

    config = load_benchmark_config(config_path)
    experiment = load_experiment_settings(config_path)
    selected = _validate_architectures(measured_architectures)
    specs: list[TrialSpec] = []
    sequence = 0
    for profile_name in config.profiles:
        for batch_size in experiment.workload.batch_sizes:
            for fraction in experiment.workload.expert_0_fractions:
                for trial_index in range(experiment.measurement.trials):
                    specs.append(
                        TrialSpec(
                            profile_name=profile_name,
                            batch_size=batch_size,
                            expert_0_fraction=fraction,
                            warmup_requests=(
                                experiment.measurement.warmup_requests
                            ),
                            measured_requests=(
                                experiment.measurement.measured_requests
                            ),
                            trial_index=trial_index,
                            seed=config.seed + sequence,
                            measured_architectures=selected,
                            correctness_oracle="A",
                        )
                    )
                    sequence += 1
    return tuple(specs)


def _trial_identity_from_spec(spec: TrialSpec) -> tuple[object, ...]:
    return (
        spec.profile_name,
        spec.batch_size,
        spec.expert_0_fraction,
        spec.trial_index,
        spec.seed,
    )


def _trial_identity_from_report(
    report: Mapping[str, object],
) -> tuple[object, ...]:
    workload = report["workload"]
    measurement = report["measurement"]
    if not isinstance(workload, dict) or not isinstance(measurement, dict):
        raise ValueError("worker report is missing workload or measurement data")
    return (
        report["profile"],
        workload["batch_size"],
        workload["requested_expert_0_fraction"],
        measurement["trial_index"],
        report["seed"],
    )


def validate_matrix_trials(
    specs: Sequence[TrialSpec],
    reports: Sequence[dict[str, object]],
) -> None:
    """Reject missing, duplicate, incorrect, or overlapping worker reports."""

    expected = [_trial_identity_from_spec(spec) for spec in specs]
    actual = [_trial_identity_from_report(report) for report in reports]
    if len(actual) != len(set(actual)):
        raise ValueError("matrix worker reports contain duplicate trial identities")
    if set(actual) != set(expected) or len(actual) != len(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(
            "matrix worker reports do not match planned trials; "
            f"missing={missing}, extra={extra}"
        )

    for spec, report in zip(specs, reports, strict=True):
        if _trial_identity_from_report(report) != _trial_identity_from_spec(spec):
            raise ValueError("matrix worker reports are out of planned order")
        if report.get("correctness", {}).get("passed") is not True:
            raise ValueError("matrix worker report failed its correctness gate")
        if set(report.get("results", {})) != set(spec.measured_architectures):
            raise ValueError("matrix worker report has unexpected measured results")
    validate_worker_intervals(reports)


def validate_worker_intervals(reports: Sequence[Mapping[str, object]]) -> None:
    """Reject missing worker timing or workers that ran concurrently."""

    previous_finished_ns: int | None = None
    for report in reports:
        worker = report.get("worker")
        if not isinstance(worker, dict):
            raise ValueError("matrix worker report is missing worker identity")
        started_ns = worker.get("started_wall_time_ns")
        finished_ns = worker.get("finished_wall_time_ns")
        if not isinstance(started_ns, int) or not isinstance(finished_ns, int):
            raise ValueError("matrix worker report has invalid timing metadata")
        if finished_ns < started_ns:
            raise ValueError("matrix worker finished before it started")
        if previous_finished_ns is not None and started_ns < previous_finished_ns:
            raise ValueError("matrix worker intervals overlap")
        previous_finished_ns = finished_ns


def _worker_command(
    config_path: Path,
    artifacts_dir: Path,
    output_path: Path,
    spec: TrialSpec,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "benchmarks.matrix",
        "--worker",
        "--config",
        str(config_path),
        "--artifacts-dir",
        str(artifacts_dir),
        "--output",
        str(output_path),
        "--spec-json",
        json.dumps(asdict(spec), separators=(",", ":")),
    ]


def run_sequential_workers(
    count: int,
    build_command: Callable[[int, Path], list[str]],
) -> list[dict[str, object]]:
    """Run `count` worker processes one at a time and collect their reports.

    `build_command(index, output_path)` must return a command that writes one
    JSON object to `output_path`. The first failure stops the run.
    """

    reports: list[dict[str, object]] = []
    project_dir = Path(__file__).parents[1]
    with tempfile.TemporaryDirectory(prefix="shared-policy-matrix-") as temporary:
        temporary_dir = Path(temporary)
        for index in range(count):
            worker_output = temporary_dir / f"trial-{index:04d}.json"
            completed = subprocess.run(
                build_command(index, worker_output),
                cwd=project_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"matrix worker {index} failed with exit code "
                    f"{completed.returncode}; stdout={completed.stdout!r}; "
                    f"stderr={completed.stderr!r}"
                )
            if not worker_output.is_file():
                raise RuntimeError(
                    f"matrix worker {index} succeeded without writing a report"
                )
            decoded = json.loads(worker_output.read_text(encoding="utf-8"))
            if not isinstance(decoded, dict):
                raise ValueError(f"matrix worker {index} report must be an object")
            reports.append(decoded)
    return reports


def run_benchmark_matrix(
    config_path: str | Path,
    artifacts_dir: str | Path,
    measured_architectures: Sequence[str] = DEFAULT_MEASURED_ARCHITECTURES,
) -> dict[str, object]:
    """Run every configured trial sequentially in a fresh Python process."""

    config_file = Path(config_path).resolve()
    artifact_root = Path(artifacts_dir).resolve()
    config = load_benchmark_config(config_file)
    experiment = load_experiment_settings(config_file)
    specs = build_matrix_specs(config_file, measured_architectures)
    reports = run_sequential_workers(
        len(specs),
        lambda index, output: _worker_command(
            config_file,
            artifact_root,
            output,
            specs[index],
        ),
    )

    validate_matrix_trials(specs, reports)
    first_report = reports[0]
    case_index = [
        {
            "profile": report["profile"],
            "batch_size": report["workload"]["batch_size"],
            "expert_0_fraction": report["workload"][
                "requested_expert_0_fraction"
            ],
            "trial_index": report["measurement"]["trial_index"],
            "seed": report["seed"],
            "worker": report["worker"],
            "summaries": {
                architecture: result["summary"]
                for architecture, result in report["results"].items()
            },
        }
        for report in reports
    ]
    return {
        "mode": "matrix",
        "seed": config.seed,
        "correctness_oracle": "A",
        "measured_architectures": list(specs[0].measured_architectures),
        "matrix": {
            "profiles": list(config.profiles),
            "batch_sizes": list(experiment.workload.batch_sizes),
            "expert_0_fractions": list(
                experiment.workload.expert_0_fractions
            ),
            "trials_per_case": experiment.measurement.trials,
            "trial_count": len(reports),
            "fresh_process_per_trial": True,
            "workers_sequential": True,
        },
        "environment": first_report["environment"],
        "runtime_settings": {
            key: value
            for key, value in first_report["runtime"].items()
            if key != "session_initialization_ns"
        },
        "case_index": case_index,
        "trials": reports,
        "validation": {
            "passed": True,
            "completed_trials": len(reports),
            "correctness_failures": 0,
            "overlapping_workers": 0,
        },
    }


def _decode_spec(encoded: str) -> TrialSpec:
    raw = json.loads(encoded)
    if not isinstance(raw, dict):
        raise ValueError("worker trial spec must be a JSON object")
    raw["measured_architectures"] = tuple(raw["measured_architectures"])
    return TrialSpec(**raw)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Internal fresh-process worker for benchmark matrix trials."
    )
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spec-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    spec = _decode_spec(args.spec_json)
    started_ns = time.time_ns()
    report = run_benchmark_trial(args.config, args.artifacts_dir, spec)
    finished_ns = time.time_ns()
    report["worker"] = {
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "started_wall_time_ns": started_ns,
        "finished_wall_time_ns": finished_ns,
    }
    write_report(report, args.output)


if __name__ == "__main__":
    main()
