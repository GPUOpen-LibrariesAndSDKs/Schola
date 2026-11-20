"""Time A against C as the number of policy heads grows."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import onnx
import onnxruntime as ort

from benchmarks.benchmark import (
    Tolerances,
    gate_and_measure,
    load_experiment_settings,
    write_report,
)
from benchmarks.matrix import run_sequential_workers, validate_worker_intervals
from benchmarks.models import (
    DEFAULT_CONFIG_PATH,
    BenchmarkConfig,
    load_benchmark_config,
)
from benchmarks.runtime import BenchmarkRuntime, load_runtime_settings


DEFAULT_SCALING_ARTIFACTS_DIR = Path(__file__).with_name("artifacts") / "scaling"
DEFAULT_SCALING_OUTPUT = Path(__file__).with_name("runs") / "scaling-results.json"
SCALING_ARCHITECTURES = ("A", "C")
EXPORTABLE_ARCHITECTURES = ("A", "C", "E")
SCALING_MIXES = ("equal", "homogeneous")
EXCLUDED_ARCHITECTURES = {
    "B": "not timed: it executes every head like A, so it cannot skip work",
    "D": (
        "not exported: ConditionalPolicyModel is a two-branch ONNX If and "
        "does not generalize past two roles"
    ),
}


@dataclass(frozen=True)
class ScalingSettings:
    """Role counts, profiles, and mixes for the A-versus-C scaling run."""

    role_counts: tuple[int, ...]
    profiles: tuple[str, ...]
    mixes: tuple[str, ...]
    batch_sizes: tuple[int, ...]
    warmup_requests: int
    measured_requests: int
    trials: int
    tolerances: Tolerances


@dataclass(frozen=True)
class ScalingTrialSpec:
    """One profile/role-count/batch/mix trial measured in one worker."""

    profile_name: str
    role_count: int
    batch_size: int
    mix: str
    warmup_requests: int
    measured_requests: int
    trial_index: int
    seed: int

    def __post_init__(self) -> None:
        if not self.profile_name:
            raise ValueError("profile_name must not be empty")
        if self.role_count < 2:
            raise ValueError("role_count must be at least 2")
        if self.mix not in SCALING_MIXES:
            raise ValueError(f"mix must be one of {SCALING_MIXES}")
        for name in ("batch_size", "warmup_requests", "measured_requests"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.trial_index < 0 or self.seed < 0:
            raise ValueError("trial_index and seed must be nonnegative")

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            self.profile_name,
            self.role_count,
            self.batch_size,
            self.mix,
            self.trial_index,
            self.seed,
        )


def scaling_roles(role_count: int) -> tuple[str, ...]:
    """Name K roles `expert_0` … `expert_{K-1}`; K=2 matches the C/D config."""

    if isinstance(role_count, bool) or not isinstance(role_count, int):
        raise ValueError("role_count must be an integer")
    if role_count < 2:
        raise ValueError("role_count must be at least 2")
    return tuple(f"expert_{index}" for index in range(role_count))


def scaling_config(config: BenchmarkConfig, role_count: int) -> BenchmarkConfig:
    """Return the same seed and widths with K roles.

    Heads are created in role order from one seeded generator, so the backbone
    and the first heads keep identical weights as K grows.
    """

    return replace(config, roles=scaling_roles(role_count))


def load_scaling_settings(config_path: str | Path) -> ScalingSettings:
    """Read the scaling keys and reuse batch, measurement, and tolerance settings."""

    path = Path(config_path)
    config = load_benchmark_config(path)
    experiment = load_experiment_settings(path)
    try:
        workloads = json.loads(path.read_text(encoding="utf-8"))["workloads"]
        raw_counts = workloads["scaling_role_counts"]
        raw_profiles = workloads["scaling_profiles"]
        raw_mixes = workloads["scaling_mixes"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid scaling config at {path}: {error}") from error

    if not isinstance(raw_counts, list) or not raw_counts:
        raise ValueError("workloads.scaling_role_counts must be a nonempty list")
    role_counts = tuple(raw_counts)
    for value in role_counts:
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(
                "workloads.scaling_role_counts entries must be integers >= 2"
            )
    if len(set(role_counts)) != len(role_counts):
        raise ValueError("workloads.scaling_role_counts must be unique")

    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError("workloads.scaling_profiles must be a nonempty list")
    unknown_profiles = sorted(set(raw_profiles) - set(config.profiles))
    if unknown_profiles:
        raise ValueError(f"unknown scaling profiles: {unknown_profiles}")

    if not isinstance(raw_mixes, list) or not raw_mixes:
        raise ValueError("workloads.scaling_mixes must be a nonempty list")
    unknown_mixes = sorted(set(raw_mixes) - set(SCALING_MIXES))
    if unknown_mixes:
        raise ValueError(f"unknown scaling mixes: {unknown_mixes}")

    return ScalingSettings(
        role_counts=role_counts,
        profiles=tuple(raw_profiles),
        mixes=tuple(raw_mixes),
        batch_sizes=experiment.workload.batch_sizes,
        warmup_requests=experiment.measurement.warmup_requests,
        measured_requests=experiment.measurement.measured_requests,
        trials=experiment.measurement.trials,
        tolerances=experiment.tolerances,
    )


def generate_scaling_workload(
    seed: int,
    observation_size: int,
    roles: tuple[str, ...],
    batch_size: int,
    mix: str,
) -> tuple[npt.NDArray[np.float32], tuple[str, ...]]:
    """Create deterministic observations and a shuffled K-role assignment.

    `equal` spreads rows as evenly as integers allow, so every head is present
    once the batch has at least K rows. `homogeneous` sends every row to
    `roles[0]`, so C makes one call while A still runs every head.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if mix == "equal":
        base, extra = divmod(batch_size, len(roles))
        counts = [base + (1 if index < extra else 0) for index in range(len(roles))]
    elif mix == "homogeneous":
        counts = [batch_size] + [0] * (len(roles) - 1)
    else:
        raise ValueError(f"mix must be one of {SCALING_MIXES}; received {mix!r}")

    role_values = np.array(
        [role for role, count in zip(roles, counts) for _ in range(count)],
        dtype=object,
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(role_values)
    observations = rng.normal(
        size=(batch_size, observation_size),
    ).astype(np.float32)
    return observations, tuple(str(role) for role in role_values)


def _role_count_dir(artifacts_dir: str | Path, role_count: int) -> Path:
    return Path(artifacts_dir) / f"roles_{role_count}"


def export_scaling_artifacts(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_SCALING_ARTIFACTS_DIR,
    role_counts: Sequence[int] | None = None,
    profiles: Sequence[str] | None = None,
    architectures: Sequence[str] = SCALING_ARCHITECTURES,
) -> Path:
    """Export the selected architectures for each role count and profile."""

    from benchmarks.export_models import OPSET_VERSION, export_profile

    unsupported = sorted(set(architectures) - set(EXPORTABLE_ARCHITECTURES))
    if unsupported:
        raise ValueError(
            f"scaling export supports {EXPORTABLE_ARCHITECTURES}; "
            f"received {unsupported}"
        )
    config = load_benchmark_config(config_path)
    settings = load_scaling_settings(config_path)
    selected_counts = tuple(settings.role_counts if role_counts is None else role_counts)
    selected_profiles = tuple(settings.profiles if profiles is None else profiles)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 1,
        "seed": config.seed,
        "opset": OPSET_VERSION,
        "architectures": list(architectures),
        "excluded_architectures": EXCLUDED_ARCHITECTURES,
        "role_counts": {
            str(role_count): {
                profile_name: export_profile(
                    scaling_config(config, role_count),
                    profile_name,
                    _role_count_dir(destination, role_count),
                    architectures,
                )
                for profile_name in selected_profiles
            }
            for role_count in selected_counts
        },
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def load_scaling_runtime(
    config_path: str | Path,
    artifacts_dir: str | Path,
    role_count: int,
    profile_name: str,
    architectures: Sequence[str] = SCALING_ARCHITECTURES,
) -> BenchmarkRuntime:
    """Load persistent sessions for one exported role count and profile."""

    return BenchmarkRuntime.from_config(
        scaling_config(load_benchmark_config(config_path), role_count),
        load_runtime_settings(config_path),
        _role_count_dir(artifacts_dir, role_count),
        profile_name,
        architectures,
    )


def artifact_bytes_by_architecture(
    profile_manifest: Mapping[str, object],
) -> dict[str, int]:
    """Sum ONNX file bytes per architecture; C is the sum of its K files."""

    totals: dict[str, int] = {}
    for artifact in profile_manifest["artifacts"].values():
        architecture = artifact["architecture"]
        totals[architecture] = totals.get(architecture, 0) + artifact["bytes"]
    return totals


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
    }


def run_scaling_trial(
    config_path: str | Path,
    artifacts_dir: str | Path,
    spec: ScalingTrialSpec,
) -> dict[str, object]:
    """Gate C on A, then time A and C for one K-role workload."""

    config = scaling_config(load_benchmark_config(config_path), spec.role_count)
    settings = load_scaling_settings(config_path)
    runtime_settings = load_runtime_settings(config_path)
    if spec.profile_name not in config.profiles:
        raise ValueError(f"unknown benchmark profile: {spec.profile_name}")

    observations, roles = generate_scaling_workload(
        spec.seed,
        config.observation_size,
        config.roles,
        spec.batch_size,
        spec.mix,
    )

    initialization_started_ns = time.perf_counter_ns()
    runtime = load_scaling_runtime(
        config_path,
        artifacts_dir,
        spec.role_count,
        spec.profile_name,
    )
    initialization_ns = time.perf_counter_ns() - initialization_started_ns

    first_runs, results, measured_orders = gate_and_measure(
        runtime,
        observations,
        roles,
        measured_architectures=SCALING_ARCHITECTURES,
        correctness_oracle="A",
        warmup_requests=spec.warmup_requests,
        measured_requests=spec.measured_requests,
        seed=spec.seed,
        tolerances=settings.tolerances,
    )

    role_counts = {role: roles.count(role) for role in config.roles}
    heads_present = sum(1 for count in role_counts.values() if count > 0)
    expected_calls = {"A": 1, "C": heads_present}
    for architecture, expected in expected_calls.items():
        actual = results[architecture]["call_count_per_request"]
        if actual != expected:
            raise RuntimeError(
                f"architecture {architecture} made {actual} model calls; "
                f"expected {expected}"
            )
        # Per-call records grow with K and are already summarized by the count.
        del results[architecture]["call_samples"]

    manifest = json.loads(
        (Path(artifacts_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    profile_manifest = manifest["role_counts"][str(spec.role_count)][
        spec.profile_name
    ]
    return {
        "mode": "scaling_trial",
        "seed": spec.seed,
        "profile": spec.profile_name,
        "role_count": spec.role_count,
        "workload": {
            "batch_size": spec.batch_size,
            "mix": spec.mix,
            "actual_role_counts": role_counts,
            "heads_present": heads_present,
        },
        "measurement": {
            "trial_index": spec.trial_index,
            "correctness_oracle": "A",
            "measured_architectures": list(SCALING_ARCHITECTURES),
            "warmup_requests_per_architecture": spec.warmup_requests,
            "measured_requests_per_architecture": spec.measured_requests,
            "candidate_orders": measured_orders,
        },
        "runtime": {
            **asdict(runtime_settings),
            "session_initialization_ns": initialization_ns,
        },
        "environment": _environment(),
        "onnx_bytes": artifact_bytes_by_architecture(profile_manifest),
        "correctness": {
            "passed": True,
            "relative_tolerance": settings.tolerances.relative,
            "absolute_tolerance": settings.tolerances.absolute,
            "first_runs": first_runs,
        },
        "results": results,
    }


def build_scaling_specs(config_path: str | Path) -> tuple[ScalingTrialSpec, ...]:
    """Expand profile × role count × batch × mix × trial into seeded specs."""

    config = load_benchmark_config(config_path)
    settings = load_scaling_settings(config_path)
    specs: list[ScalingTrialSpec] = []
    for profile_name in settings.profiles:
        for role_count in settings.role_counts:
            for batch_size in settings.batch_sizes:
                for mix in settings.mixes:
                    for trial_index in range(settings.trials):
                        specs.append(
                            ScalingTrialSpec(
                                profile_name=profile_name,
                                role_count=role_count,
                                batch_size=batch_size,
                                mix=mix,
                                warmup_requests=settings.warmup_requests,
                                measured_requests=settings.measured_requests,
                                trial_index=trial_index,
                                seed=config.seed + len(specs),
                            )
                        )
    return tuple(specs)


def _identity_from_report(report: Mapping[str, object]) -> tuple[object, ...]:
    return (
        report["profile"],
        report["role_count"],
        report["workload"]["batch_size"],
        report["workload"]["mix"],
        report["measurement"]["trial_index"],
        report["seed"],
    )


def validate_scaling_trials(
    specs: Sequence[ScalingTrialSpec],
    reports: Sequence[Mapping[str, object]],
) -> None:
    """Reject missing, reordered, incorrect, or overlapping worker reports."""

    if len(reports) != len(specs):
        raise ValueError(
            f"expected {len(specs)} scaling reports; received {len(reports)}"
        )
    for spec, report in zip(specs, reports, strict=True):
        if _identity_from_report(report) != spec.identity:
            raise ValueError("scaling reports do not match the planned trials")
        if report.get("correctness", {}).get("passed") is not True:
            raise ValueError("scaling report failed its correctness gate")
        if set(report.get("results", {})) != set(SCALING_ARCHITECTURES):
            raise ValueError("scaling report has unexpected measured results")
    validate_worker_intervals(reports)


def _verdict(a_values: Sequence[float], c_values: Sequence[float]) -> str:
    """Name a winner only when the trial ranges do not overlap."""

    if max(c_values) < min(a_values):
        return "C faster"
    if max(a_values) < min(c_values):
        return "A faster"
    return "inconclusive"


def summarize_scaling(reports: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Compare trial p50s per cell and find where C starts beating A."""

    grouped: dict[tuple[str, int, str, int], list[Mapping[str, object]]] = {}
    for report in reports:
        key = (
            report["profile"],
            report["workload"]["batch_size"],
            report["workload"]["mix"],
            report["role_count"],
        )
        grouped.setdefault(key, []).append(report)

    cells: list[dict[str, object]] = []
    for (profile, batch_size, mix, role_count), group in sorted(grouped.items()):
        p50s = {
            architecture: [
                float(report["results"][architecture]["summary"]["p50_request_ns"])
                for report in group
            ]
            for architecture in SCALING_ARCHITECTURES
        }
        a_median = median(p50s["A"])
        c_median = median(p50s["C"])
        first = group[0]
        cells.append(
            {
                "profile": profile,
                "batch_size": batch_size,
                "mix": mix,
                "role_count": role_count,
                "trials": len(group),
                "heads_present": first["workload"]["heads_present"],
                "model_calls": {
                    architecture: first["results"][architecture][
                        "call_count_per_request"
                    ]
                    for architecture in SCALING_ARCHITECTURES
                },
                "onnx_bytes": {
                    architecture: first["onnx_bytes"][architecture]
                    for architecture in SCALING_ARCHITECTURES
                },
                "p50_request_ns": {
                    architecture: {
                        "median": median(values),
                        "min": min(values),
                        "max": max(values),
                        "trials": values,
                    }
                    for architecture, values in p50s.items()
                },
                "c_over_a_median_ratio": c_median / a_median,
                "verdict": _verdict(p50s["A"], p50s["C"]),
            }
        )

    crossovers: list[dict[str, object]] = []
    series_keys = sorted({(c["profile"], c["batch_size"], c["mix"]) for c in cells})
    for profile, batch_size, mix in series_keys:
        series = sorted(
            (
                cell
                for cell in cells
                if (cell["profile"], cell["batch_size"], cell["mix"])
                == (profile, batch_size, mix)
            ),
            key=lambda cell: cell["role_count"],
        )
        verdicts = [cell["verdict"] for cell in series]
        sustained = next(
            (
                series[index]["role_count"]
                for index in range(len(series))
                if all(verdict == "C faster" for verdict in verdicts[index:])
            ),
            None,
        )
        crossovers.append(
            {
                "profile": profile,
                "batch_size": batch_size,
                "mix": mix,
                "verdicts_by_role_count": {
                    str(cell["role_count"]): cell["verdict"] for cell in series
                },
                "c_faster_from_role_count": sustained,
            }
        )
    return {
        "rule": (
            "C faster (or A faster) only when every trial p50 for one "
            "architecture is below every trial p50 for the other; otherwise "
            "inconclusive. c_faster_from_role_count is the smallest K from "
            "which C is faster at that K and every larger K tested."
        ),
        "cells": cells,
        "crossovers": crossovers,
    }


def format_summary_table(summary: Mapping[str, object]) -> str:
    """Render one line per cell for terminal review."""

    lines = [
        f"{'profile':<16}{'batch':>6} {'mix':<12}{'K':>3}{'heads':>6}"
        f"{'A us':>10}{'C us':>10}{'C/A':>7}{'C calls':>8}"
        f"{'A bytes':>11}{'C bytes':>11}  verdict"
    ]
    for cell in summary["cells"]:
        p50 = cell["p50_request_ns"]
        lines.append(
            f"{cell['profile']:<16}{cell['batch_size']:>6} {cell['mix']:<12}"
            f"{cell['role_count']:>3}{cell['heads_present']:>6}"
            f"{p50['A']['median'] / 1000:>10.1f}{p50['C']['median'] / 1000:>10.1f}"
            f"{cell['c_over_a_median_ratio']:>7.2f}{cell['model_calls']['C']:>8}"
            f"{cell['onnx_bytes']['A']:>11}{cell['onnx_bytes']['C']:>11}"
            f"  {cell['verdict']}"
        )
    lines.append("")
    for crossover in summary["crossovers"]:
        lines.append(
            f"{crossover['profile']} batch={crossover['batch_size']} "
            f"mix={crossover['mix']}: C faster from K="
            f"{crossover['c_faster_from_role_count']}"
        )
    return "\n".join(lines)


def _worker_command(
    config_path: Path,
    artifacts_dir: Path,
    output_path: Path,
    spec: ScalingTrialSpec,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "benchmarks.scaling",
        "worker",
        "--config",
        str(config_path),
        "--artifacts-dir",
        str(artifacts_dir),
        "--output",
        str(output_path),
        "--spec-json",
        json.dumps(asdict(spec), separators=(",", ":")),
    ]


def run_scaling_matrix(
    config_path: str | Path,
    artifacts_dir: str | Path,
) -> dict[str, object]:
    """Run every scaling trial sequentially, each in a fresh process."""

    config_file = Path(config_path).resolve()
    artifact_root = Path(artifacts_dir).resolve()
    config = load_benchmark_config(config_file)
    settings = load_scaling_settings(config_file)
    specs = build_scaling_specs(config_file)
    reports = run_sequential_workers(
        len(specs),
        lambda index, output: _worker_command(
            config_file,
            artifact_root,
            output,
            specs[index],
        ),
    )
    validate_scaling_trials(specs, reports)
    return {
        "mode": "scaling",
        "seed": config.seed,
        "correctness_oracle": "A",
        "measured_architectures": list(SCALING_ARCHITECTURES),
        "excluded_architectures": EXCLUDED_ARCHITECTURES,
        "matrix": {
            "profiles": list(settings.profiles),
            "role_counts": list(settings.role_counts),
            "batch_sizes": list(settings.batch_sizes),
            "mixes": list(settings.mixes),
            "trials_per_case": settings.trials,
            "trial_count": len(reports),
            "fresh_process_per_trial": True,
            "workers_sequential": True,
        },
        "environment": reports[0]["environment"],
        "runtime_settings": {
            key: value
            for key, value in reports[0]["runtime"].items()
            if key != "session_initialization_ns"
        },
        "summary": summarize_scaling(reports),
        "trials": reports,
        "validation": {
            "passed": True,
            "completed_trials": len(reports),
            "correctness_failures": 0,
            "overlapping_workers": 0,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export and time A versus C as the number of heads grows.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export", help="Export per-K A/C artifacts.")
    export.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    export.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SCALING_ARTIFACTS_DIR,
    )

    run = commands.add_parser("run", help="Run the scaling matrix.")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    run.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_SCALING_ARTIFACTS_DIR,
    )
    run.add_argument("--output", type=Path, default=DEFAULT_SCALING_OUTPUT)

    show = commands.add_parser("summarize", help="Print a saved report's table.")
    show.add_argument("--input", type=Path, default=DEFAULT_SCALING_OUTPUT)

    worker = commands.add_parser("worker", help="Internal fresh-process worker.")
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--artifacts-dir", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--spec-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "export":
        manifest_path = export_scaling_artifacts(args.config, args.output_dir)
        print(f"Exported and validated scaling models: {manifest_path}")
    elif args.command == "run":
        report = run_scaling_matrix(args.config, args.artifacts_dir)
        output_path = write_report(report, args.output)
        print(format_summary_table(report["summary"]))
        print(f"Correctness passed; scaling report: {output_path}")
    elif args.command == "summarize":
        report = json.loads(args.input.read_text(encoding="utf-8"))
        print(format_summary_table(report["summary"]))
    else:
        spec = ScalingTrialSpec(**json.loads(args.spec_json))
        started_ns = time.time_ns()
        report = run_scaling_trial(args.config, args.artifacts_dir, spec)
        report["worker"] = {
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "started_wall_time_ns": started_ns,
            "finished_wall_time_ns": time.time_ns(),
        }
        write_report(report, args.output)


if __name__ == "__main__":
    main()
