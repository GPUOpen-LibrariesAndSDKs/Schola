"""Tests for the N-role A-versus-C scaling benchmark."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from benchmarks.export_models import _component_hashes
from benchmarks.models import build_seeded_policy, load_benchmark_config
from benchmarks.scaling import (
    build_scaling_specs,
    export_scaling_artifacts,
    generate_scaling_workload,
    load_scaling_runtime,
    load_scaling_settings,
    run_scaling_matrix,
    scaling_roles,
    summarize_scaling,
)


CONFIG_PATH = Path(__file__).parents[1] / "config.json"
TESTED_ROLE_COUNTS = (2, 4, 8)


@pytest.fixture(scope="module")
def scaling_artifacts(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output_dir = tmp_path_factory.mktemp("scaling-artifacts")
    export_scaling_artifacts(
        CONFIG_PATH,
        output_dir,
        role_counts=TESTED_ROLE_COUNTS,
        profiles=("small",),
        architectures=("A", "C", "E"),
    )
    return output_dir


def test_loads_scaling_settings() -> None:
    settings = load_scaling_settings(CONFIG_PATH)

    assert settings.role_counts == (2, 4, 8, 16)
    assert settings.profiles == ("balanced", "expensive_heads")
    assert settings.mixes == ("equal", "homogeneous")
    assert settings.batch_sizes == (1, 16, 64, 256)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("scaling_role_counts", [1, 4], ">= 2"),
        ("scaling_role_counts", [4, 4], "unique"),
        ("scaling_profiles", ["huge"], "unknown scaling profiles"),
        ("scaling_mixes", ["skewed"], "unknown scaling mixes"),
    ],
)
def test_rejects_invalid_scaling_settings(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    raw = deepcopy(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    raw["workloads"][field] = value
    config_path = tmp_path / "invalid.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_scaling_settings(config_path)


def test_equal_mix_is_as_even_as_integers_allow_and_shuffled() -> None:
    roles = scaling_roles(4)
    first_observations, first = generate_scaling_workload(301, 64, roles, 10, "equal")
    second_observations, second = generate_scaling_workload(
        301, 64, roles, 10, "equal"
    )

    assert [first.count(role) for role in roles] == [3, 3, 2, 2]
    assert first == second
    np.testing.assert_array_equal(first_observations, second_observations)
    assert first != tuple(sorted(first))


def test_homogeneous_mix_sends_every_row_to_the_first_role() -> None:
    _, roles = generate_scaling_workload(301, 64, scaling_roles(8), 16, "homogeneous")

    assert roles == ("expert_0",) * 16


def test_rejects_unknown_mix() -> None:
    with pytest.raises(ValueError, match="mix must be one of"):
        generate_scaling_workload(301, 64, scaling_roles(4), 4, "skewed")


def test_manifest_excludes_d_and_records_why(scaling_artifacts: Path) -> None:
    manifest = json.loads((scaling_artifacts / "manifest.json").read_text())

    assert "two-branch ONNX If" in manifest["excluded_architectures"]["D"]
    for role_count in TESTED_ROLE_COUNTS:
        artifacts = manifest["role_counts"][str(role_count)]["small"]["artifacts"]
        assert "D_conditional" not in artifacts
        assert "B_masked" not in artifacts
        assert sum(name.startswith("C_") for name in artifacts) == role_count
        assert sum(name.endswith("_head") for name in artifacts) == role_count
        assert not (
            scaling_artifacts / f"roles_{role_count}" / "small" / "D_conditional.onnx"
        ).exists()


def test_backbone_and_shared_heads_keep_weights_as_role_count_grows(
    scaling_artifacts: Path,
) -> None:
    manifest = json.loads((scaling_artifacts / "manifest.json").read_text())
    two_role_hashes = _component_hashes(
        build_seeded_policy(load_benchmark_config(CONFIG_PATH), "small")
    )

    for role_count in TESTED_ROLE_COUNTS:
        hashes = manifest["role_counts"][str(role_count)]["small"][
            "source_parameter_hashes"
        ]
        for component in ("backbone", "expert_0_head", "expert_1_head"):
            assert hashes[component] == two_role_hashes[component]


@pytest.mark.parametrize("role_count", TESTED_ROLE_COUNTS)
@pytest.mark.parametrize("mix", ["equal", "homogeneous"])
def test_c_and_e_match_a_at_n_roles(
    scaling_artifacts: Path,
    role_count: int,
    mix: str,
) -> None:
    runtime = load_scaling_runtime(
        CONFIG_PATH,
        scaling_artifacts,
        role_count,
        "small",
        ("A", "C", "E"),
    )
    observations, roles = generate_scaling_workload(
        301, 64, runtime.roles, 37, mix
    )
    heads_present = len(set(roles))
    expected = runtime.run("A", observations, roles)

    assert expected.call_count == 1
    assert heads_present == (role_count if mix == "equal" else 1)
    for architecture, calls in (("C", heads_present), ("E", 1 + heads_present)):
        result = runtime.run(architecture, observations, roles)
        np.testing.assert_allclose(
            result.actions,
            expected.actions,
            rtol=1e-4,
            atol=1e-5,
        )
        assert result.call_count == calls


def test_runtime_rejects_an_architecture_it_did_not_load(
    scaling_artifacts: Path,
) -> None:
    runtime = load_scaling_runtime(CONFIG_PATH, scaling_artifacts, 4, "small")

    with pytest.raises(ValueError, match="D was not loaded"):
        runtime.run("D", np.zeros((1, 64), dtype=np.float32), ("expert_0",))


def _cell_report(role_count: int, a_p50s: float, c_p50s: float) -> dict:
    def result(p50: float, calls: int) -> dict:
        return {
            "call_count_per_request": calls,
            "summary": {"p50_request_ns": p50},
        }

    return {
        "profile": "small",
        "role_count": role_count,
        "workload": {"batch_size": 16, "mix": "equal", "heads_present": role_count},
        "onnx_bytes": {"A": 100, "C": 100 * role_count},
        "results": {"A": result(a_p50s, 1), "C": result(c_p50s, role_count)},
    }


def test_summary_names_a_winner_only_without_overlapping_trials() -> None:
    reports = [
        # K=2: C clearly slower in every trial.
        _cell_report(2, 10, 20),
        _cell_report(2, 11, 21),
        # K=4: ranges overlap.
        _cell_report(4, 10, 9),
        _cell_report(4, 12, 13),
        # K=8 and K=16: C clearly faster in every trial.
        _cell_report(8, 30, 10),
        _cell_report(8, 31, 11),
        _cell_report(16, 60, 12),
        _cell_report(16, 61, 13),
    ]

    summary = summarize_scaling(reports)

    assert [cell["verdict"] for cell in summary["cells"]] == [
        "A faster",
        "inconclusive",
        "C faster",
        "C faster",
    ]
    assert summary["cells"][2]["model_calls"] == {"A": 1, "C": 8}
    assert summary["cells"][2]["onnx_bytes"] == {"A": 100, "C": 800}
    (crossover,) = summary["crossovers"]
    assert crossover["c_faster_from_role_count"] == 8


def test_summary_has_no_crossover_when_c_falls_behind_again() -> None:
    summary = summarize_scaling(
        [_cell_report(2, 30, 10), _cell_report(4, 10, 30)]
    )

    assert summary["crossovers"][0]["c_faster_from_role_count"] is None


def _write_reduced_config(tmp_path: Path) -> Path:
    raw = deepcopy(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    raw["profiles"] = {"small": raw["profiles"]["small"]}
    raw["workloads"]["batch_sizes"] = [4]
    raw["workloads"]["scaling_role_counts"] = [2, 4]
    raw["workloads"]["scaling_profiles"] = ["small"]
    raw["measurement"] = {
        "warmup_requests": 1,
        "measured_requests": 2,
        "trials": 1,
    }
    config_path = tmp_path / "scaling-config.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    return config_path


def test_reduced_scaling_matrix_runs_in_fresh_sequential_workers(
    tmp_path: Path,
) -> None:
    config_path = _write_reduced_config(tmp_path)
    artifacts_dir = tmp_path / "artifacts"
    export_scaling_artifacts(config_path, artifacts_dir)

    report = run_scaling_matrix(config_path, artifacts_dir)

    assert report["validation"]["completed_trials"] == 4
    assert len(build_scaling_specs(config_path)) == 4
    assert report["measured_architectures"] == ["A", "C"]
    for trial in report["trials"]:
        role_count = trial["role_count"]
        heads = trial["workload"]["heads_present"]
        assert heads == (role_count if trial["workload"]["mix"] == "equal" else 1)
        assert trial["results"]["A"]["call_count_per_request"] == 1
        assert trial["results"]["C"]["call_count_per_request"] == heads
        assert set(trial["correctness"]["first_runs"]) == {"A", "C"}
        assert all(
            np.isfinite(trial["results"][architecture]["summary"]["p50_request_ns"])
            for architecture in ("A", "C")
        )
        profile_dir = artifacts_dir / f"roles_{role_count}" / "small"
        assert trial["onnx_bytes"] == {
            "A": (profile_dir / "A_all_heads.onnx").stat().st_size,
            "C": sum(path.stat().st_size for path in profile_dir.glob("C_*.onnx")),
        }
    workers = [trial["worker"] for trial in report["trials"]]
    assert len({worker["pid"] for worker in workers}) == 4
    assert len(report["summary"]["cells"]) == 4
    assert len(report["summary"]["crossovers"]) == 2


def test_scaling_matrix_stops_when_a_worker_fails(tmp_path: Path) -> None:
    config_path = _write_reduced_config(tmp_path)

    with pytest.raises(RuntimeError, match="matrix worker 0 failed"):
        run_scaling_matrix(config_path, tmp_path / "missing-artifacts")
