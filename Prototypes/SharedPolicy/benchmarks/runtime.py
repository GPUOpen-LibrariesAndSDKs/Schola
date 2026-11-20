"""Run A–E ONNX architectures behind one mixed-role request contract."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from benchmarks.models import (
    DEFAULT_CONFIG_PATH,
    BenchmarkConfig,
    load_benchmark_config,
)


FloatArray = npt.NDArray[np.float32]
SUPPORTED_ARCHITECTURES = ("A", "B", "C", "D", "E")


@dataclass(frozen=True)
class RuntimeSettings:
    """ONNX Runtime settings shared by every architecture session."""

    provider: str
    execution_mode: str
    intra_op_threads: int
    inter_op_threads: int
    graph_optimization: str


@dataclass(frozen=True)
class ModelCall:
    """One successful ONNX invocation made for a request."""

    artifact: str
    rows: int
    duration_ns: int


@dataclass(frozen=True)
class RunResult:
    """Restored actions plus the model calls required to produce them."""

    actions: FloatArray
    calls: tuple[ModelCall, ...]

    @property
    def call_count(self) -> int:
        return len(self.calls)


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_runtime_settings(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> RuntimeSettings:
    """Read and validate the ONNX Runtime portion of the JSON config."""

    path = Path(config_path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))["runtime"]
        if not isinstance(raw, dict):
            raise ValueError("runtime must be a JSON object")
        settings = RuntimeSettings(
            provider=raw["provider"],
            execution_mode=raw["execution_mode"],
            intra_op_threads=_positive_int(
                "runtime.intra_op_threads",
                raw["intra_op_threads"],
            ),
            inter_op_threads=_positive_int(
                "runtime.inter_op_threads",
                raw["inter_op_threads"],
            ),
            graph_optimization=raw["graph_optimization"],
        )
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid runtime config at {path}: {error}") from error

    if not isinstance(settings.provider, str) or not settings.provider:
        raise ValueError("runtime.provider must be a non-empty string")
    if settings.execution_mode not in {"sequential", "parallel"}:
        raise ValueError("runtime.execution_mode must be sequential or parallel")
    if settings.graph_optimization not in {
        "disabled",
        "basic",
        "extended",
        "all",
    }:
        raise ValueError(
            "runtime.graph_optimization must be disabled, basic, extended, or all"
        )
    return settings


def create_session_options(
    settings: RuntimeSettings,
    profile_prefix: str | Path | None = None,
) -> ort.SessionOptions:
    """Translate human-readable config values into ORT options."""

    options = ort.SessionOptions()
    options.execution_mode = {
        "sequential": ort.ExecutionMode.ORT_SEQUENTIAL,
        "parallel": ort.ExecutionMode.ORT_PARALLEL,
    }[settings.execution_mode]
    options.intra_op_num_threads = settings.intra_op_threads
    options.inter_op_num_threads = settings.inter_op_threads
    options.graph_optimization_level = {
        "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }[settings.graph_optimization]
    if profile_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    return options


def _validate_architectures(architectures: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(architectures)
    if not selected:
        raise ValueError("at least one architecture is required")
    if len(set(selected)) != len(selected):
        raise ValueError("architectures must be unique")
    unsupported = sorted(set(selected) - set(SUPPORTED_ARCHITECTURES))
    if unsupported:
        raise ValueError(f"unsupported architectures: {unsupported}")
    return selected


def _expected_contracts(
    roles: tuple[str, ...],
    architectures: Sequence[str] = SUPPORTED_ARCHITECTURES,
) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """Return required public input and output names for selected artifacts."""

    contracts: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    if "A" in architectures:
        contracts["A_all_heads"] = (
            ("observations",),
            tuple(f"{role}_actions" for role in roles),
        )
    if "B" in architectures:
        contracts["B_masked"] = (
            ("observations", "role_masks"),
            ("selected_actions",),
        )
    if "D" in architectures:
        contracts["D_conditional"] = (
            ("observations", "use_first_expert"),
            ("actions",),
        )
    if "E" in architectures:
        contracts["E_backbone"] = (("observations",), ("features",))
    for role in roles:
        if "C" in architectures:
            contracts[f"C_{role}"] = (("observations",), ("actions",))
        if "E" in architectures:
            contracts[f"E_{role}_head"] = (("features",), ("actions",))
    return contracts


def _validate_session_contracts(
    sessions: Mapping[str, ort.InferenceSession],
    roles: tuple[str, ...],
    architectures: Sequence[str] = SUPPORTED_ARCHITECTURES,
) -> None:
    """Reject missing or unexpectedly named public tensors before requests."""

    contracts = _expected_contracts(roles, architectures)
    if set(sessions) != set(contracts):
        missing = sorted(set(contracts) - set(sessions))
        extra = sorted(set(sessions) - set(contracts))
        raise ValueError(
            f"session set does not match artifacts; missing={missing}, extra={extra}"
        )

    for artifact, (expected_inputs, expected_outputs) in contracts.items():
        session = sessions[artifact]
        actual_inputs = tuple(value.name for value in session.get_inputs())
        actual_outputs = tuple(value.name for value in session.get_outputs())
        if actual_inputs != expected_inputs or actual_outputs != expected_outputs:
            raise ValueError(
                f"{artifact} tensor contract mismatch: expected inputs "
                f"{expected_inputs} and outputs {expected_outputs}; received "
                f"inputs {actual_inputs} and outputs {actual_outputs}"
            )


def _load_sessions(
    profile_dir: Path,
    roles: tuple[str, ...],
    settings: RuntimeSettings,
    architectures: Sequence[str] = SUPPORTED_ARCHITECTURES,
) -> dict[str, ort.InferenceSession]:
    """Create every session once so requests never reload model files."""

    available_providers = ort.get_available_providers()
    if settings.provider not in available_providers:
        raise ValueError(
            f"provider {settings.provider!r} is unavailable; "
            f"available providers: {available_providers}"
        )

    sessions: dict[str, ort.InferenceSession] = {}
    for artifact in _expected_contracts(roles, architectures):
        model_path = profile_dir / f"{artifact}.onnx"
        if not model_path.is_file():
            raise FileNotFoundError(f"missing ONNX artifact: {model_path}")
        sessions[artifact] = ort.InferenceSession(
            str(model_path),
            sess_options=create_session_options(settings),
            providers=[settings.provider],
        )
    _validate_session_contracts(sessions, roles, architectures)
    return sessions


class BenchmarkRuntime:
    """Persistent sessions for selected A–E architectures with shared routing."""

    def __init__(
        self,
        config: BenchmarkConfig,
        profile_name: str,
        sessions: Mapping[str, ort.InferenceSession],
        architectures: Sequence[str] = SUPPORTED_ARCHITECTURES,
    ) -> None:
        self.dimensions = config.dimensions(profile_name)
        self.roles = self.dimensions.roles
        self.architectures = _validate_architectures(architectures)
        self._role_indices = {
            role: index for index, role in enumerate(self.roles)
        }
        self._sessions = dict(sessions)
        _validate_session_contracts(
            self._sessions,
            self.roles,
            self.architectures,
        )

    @classmethod
    def from_paths(
        cls,
        config_path: str | Path,
        artifacts_dir: str | Path,
        profile_name: str,
    ) -> BenchmarkRuntime:
        """Load configuration and persistent sessions for one model profile."""

        return cls.from_config(
            load_benchmark_config(config_path),
            load_runtime_settings(config_path),
            artifacts_dir,
            profile_name,
        )

    @classmethod
    def from_config(
        cls,
        config: BenchmarkConfig,
        settings: RuntimeSettings,
        artifacts_dir: str | Path,
        profile_name: str,
        architectures: Sequence[str] = SUPPORTED_ARCHITECTURES,
    ) -> BenchmarkRuntime:
        """Load sessions for an explicit config, such as an N-role variant."""

        selected = _validate_architectures(architectures)
        sessions = _load_sessions(
            Path(artifacts_dir) / profile_name,
            config.roles,
            settings,
            selected,
        )
        return cls(config, profile_name, sessions, selected)

    @property
    def loaded_artifacts(self) -> tuple[str, ...]:
        """List the persistent model sessions available to requests."""

        return tuple(self._sessions)

    def _validate_request(
        self,
        architecture: str,
        observations: FloatArray,
        roles: Sequence[str],
    ) -> tuple[np.ndarray, tuple[tuple[str, np.ndarray], ...]]:
        if architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"unknown architecture {architecture!r}; expected A, B, C, D, or E"
            )
        if architecture not in self.architectures:
            raise ValueError(
                f"architecture {architecture} was not loaded; loaded "
                f"{list(self.architectures)}"
            )
        if not isinstance(observations, np.ndarray):
            raise TypeError("observations must be a NumPy array")
        if observations.dtype != np.float32:
            raise TypeError("observations must use float32")
        if observations.ndim != 2:
            raise ValueError("observations must have shape [batch, observation]")
        if observations.shape[1] != self.dimensions.observation:
            raise ValueError(
                "observations have the wrong feature width; expected "
                f"{self.dimensions.observation}, received {observations.shape[1]}"
            )
        if observations.shape[0] == 0:
            raise ValueError("observations must contain at least one row")
        if len(roles) != observations.shape[0]:
            raise ValueError("roles must contain one entry per observation")

        unknown_roles = sorted(set(roles) - set(self.roles))
        if unknown_roles:
            raise ValueError(f"unknown roles: {unknown_roles}")

        role_ids = np.fromiter(
            (self._role_indices[role] for role in roles),
            dtype=np.int64,
            count=len(roles),
        )
        groups = tuple(
            (role, np.flatnonzero(role_ids == role_index))
            for role_index, role in enumerate(self.roles)
            if np.any(role_ids == role_index)
        )
        return role_ids, groups

    def _run_session(
        self,
        artifact: str,
        inputs: dict[str, np.ndarray],
        rows: int,
        calls: list[ModelCall],
    ) -> list[np.ndarray]:
        started_ns = time.perf_counter_ns()
        try:
            outputs = self._sessions[artifact].run(None, inputs)
        except Exception as error:
            raise RuntimeError(f"{artifact} inference failed") from error
        duration_ns = time.perf_counter_ns() - started_ns
        calls.append(
            ModelCall(
                artifact=artifact,
                rows=rows,
                duration_ns=duration_ns,
            )
        )
        return outputs

    def run(
        self,
        architecture: str,
        observations: FloatArray,
        roles: Sequence[str],
    ) -> RunResult:
        """Run one mixed-role request and restore actions to original row order."""

        architecture = architecture.upper()
        role_ids, groups = self._validate_request(
            architecture,
            observations,
            roles,
        )
        batch_size = observations.shape[0]
        calls: list[ModelCall] = []

        if architecture == "A":
            all_actions = self._run_session(
                "A_all_heads",
                {"observations": observations},
                batch_size,
                calls,
            )
            stacked_actions = np.stack(all_actions, axis=1)
            actions = stacked_actions[np.arange(batch_size), role_ids]

        elif architecture == "B":
            role_masks = np.eye(len(self.roles), dtype=np.float32)[role_ids]
            actions = self._run_session(
                "B_masked",
                {"observations": observations, "role_masks": role_masks},
                batch_size,
                calls,
            )[0]

        elif architecture in {"C", "D"}:
            actions = np.empty(
                (batch_size, self.dimensions.action),
                dtype=np.float32,
            )
            for role, row_indices in groups:
                if architecture == "C":
                    artifact = f"C_{role}"
                    inputs = {"observations": observations[row_indices]}
                else:
                    artifact = "D_conditional"
                    inputs = {
                        "observations": observations[row_indices],
                        "use_first_expert": np.array(
                            self._role_indices[role] == 0,
                            dtype=np.bool_,
                        ),
                    }
                actions[row_indices] = self._run_session(
                    artifact,
                    inputs,
                    len(row_indices),
                    calls,
                )[0]

        else:
            features = self._run_session(
                "E_backbone",
                {"observations": observations},
                batch_size,
                calls,
            )[0]
            actions = np.empty(
                (batch_size, self.dimensions.action),
                dtype=np.float32,
            )
            for role, row_indices in groups:
                actions[row_indices] = self._run_session(
                    f"E_{role}_head",
                    {"features": features[row_indices]},
                    len(row_indices),
                    calls,
                )[0]

        return RunResult(
            actions=np.asarray(actions, dtype=np.float32),
            calls=tuple(calls),
        )
