# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Dataclasses for ``schola fabrica`` (CLI + runtime)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Union

from cyclopts import Parameter
from cyclopts.types import PositiveInt

from schola.scripts.common.settings import (
    EnvironmentSettings,
    UnrealExecutableSimulatorConfig,
    UnrealProjectSimulatorConfig,
)
from schola.scripts.sb3.train.settings import (
    IgnoreParameter,
    PPOTrainSettings,
    SACTrainSettings,
    Sb3CheckpointSettings,
    Sb3LoggingSettings,
    Sb3NetworkArchitectureSettings,
    Sb3ResumeSettings,
    Sb3TrainScriptSettings,
    Sb3TrainingSettings,
)


@dataclass
class FabricaLLMSettings:
    """OpenAI-compatible LLM settings for the Fabrica reward Deep Agent."""

    api_key: Optional[str] = None
    "API key (defaults to ``OPENAI_API_KEY``)."

    base_url: Optional[str] = None
    "OpenAI-compatible API base URL (for example ``https://api.openai.com/v1``)."

    model: str = "gpt-4o-mini"
    "Chat model name passed to LangChain ``init_chat_model``."

    headers: Dict[str, str] = field(default_factory=dict)
    "Extra HTTP headers on each LLM request (e.g. ``--headers.Ocp-Apim-Subscription-Key=key --headers.user=you``)."

    temperature: float = 0.2
    "Sampling temperature."

    max_tokens: Optional[int] = None
    "Maximum tokens per LLM completion (omit to use the provider default)."

    timeout_s: float = 120.0
    "HTTP request timeout in seconds for each LLM call."

    verify_ssl: bool = True
    "Verify TLS certificates for LLM HTTP requests (disable only on trusted corporate networks)."


@dataclass
class FabricaEditorSnapshotSettings:
    """Optional Unreal Editor Python world snapshot settings."""

    enabled: bool = False
    "Run Unreal Editor Python world snapshot."

    max_actors: int = 500
    "Cap actors serialized in snapshot."

    class_filter_substrings: Annotated[
        List[str],
        Parameter(consume_multiple=True),
    ] = field(default_factory=list)
    "Substrings matched against actor class names; only matching actors are included in the snapshot. Repeat the flag for multiple filters."

    editor_path: Optional[Path] = None
    "Override path to UnrealEditor-Cmd if engine auto-detect fails."

    timeout_s: float = 600.0
    "Maximum seconds to wait for UnrealEditor-Cmd to finish the world snapshot."


@dataclass
class FabricaLoopSettings:
    """Outer Fabrica loop: iterations, samples, and agent step budget."""

    iterations: int = 5
    "Outer Fabrica iterations."

    samples: int = 5
    "Reward samples per iteration."

    train_timesteps_per_sample: int = 5000
    "SB3 timesteps per reward sample when scoring a candidate."

    reward_agent_max_steps: int = 40
    "LangGraph recursion / agent step budget."

    fabrica_info_prefix: str = "fabrica_r:"
    "Prefix for per-component reward keys in ``FAgentState::Info`` (must match ``UFabricaRewardInfo::GetComponentPrefix()`` in C++)."

    fabrica_task_success_key: str = "fabrica_ts"
    "``Info`` key for the scalar task-success metric logged each episode (must match ``UFabricaRewardInfo::GetTaskSuccessInfoPrefix()`` in C++)."

    policy_feedback_interval: PositiveInt = 1
    "Interval, in episodes, to use episode metrics for ``feedback``."

    pbar: bool = False
    "Enable progress bar."

    def __post_init__(self) -> None:
        if self.policy_feedback_interval <= 0:
            raise ValueError(
                "policy_feedback_interval must be greater than 0 "
                f"(got {self.policy_feedback_interval})."
            )


@dataclass
class FabricaPathsSettings:
    """Paths for the environment header, task description, generated C++, and run artifacts."""

    env_header: Annotated[
        Path,
        Parameter(name=("--env-header", "-H")),
    ] = Path(".")
    "Environment header declaring ``AYourEnv : AFabricaEnvironment``."

    task_description: Annotated[
        Path,
        Parameter(name=("--task-description", "-T")),
    ] = Path(".")
    "Path to a text/markdown task description for the reward."

    code_gen_folder: Optional[Path] = None
    "Optional folder for ``*.fabrica.gen.cpp``. Default: mirror ``Public`` → ``Private`` on ``env_header``, or the header directory when ``Public`` is absent."

    output_root: Path = Path("./fabrica_outputs")
    "Directory for ``fabrica_outputs/<timestamp>`` artifacts."

    code_roots: Annotated[List[Path], Parameter(consume_multiple=True)] = field(
        default_factory=list
    )
    "Extra allowed roots for UE file tools, added to the env-header parent and uproject parent. Nested roots collapse to the shortest ancestor."

    _run_artifact_dir: Annotated[Optional[Path], Parameter(parse=False, show=False)] = (
        field(init=False, default=None)
    )
    "Path to the run artifacts directory (generated on object initialization)."

    @property
    def run_artifact_dir(self) -> Path:
        if self._run_artifact_dir is None:
            raise ValueError("run_artifact_dir is not set")
        return self._run_artifact_dir

    @property
    def snapshot_json_path(self) -> Path:
        return self.run_artifact_dir / "world_snapshot.json"

    def __post_init__(self) -> None:
        self._run_artifact_dir = self.output_root / datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )


@dataclass
class FabricaScriptSettings:
    """
    Top-level dataclass for configuring the script arguments used in the Fabrica launcher.

    Paths, LLM, editor snapshot, and loop settings sit alongside the same SB3 option groups
    as ``schola sb3 train`` (flat on this dataclass).

    ``MetaAlgCommand`` sets ``algorithm_settings`` from the algorithm subcommand and assigns
    ``environment_settings.simulator_settings`` from the simulator subcommand.
    """

    paths_settings: Annotated[
        FabricaPathsSettings,
        Parameter(name="*", group="Fabrica paths"),
    ] = field(default_factory=FabricaPathsSettings)
    "Paths for the environment header, task description, generated C++, and run artifacts."

    llm_settings: Annotated[
        FabricaLLMSettings,
        Parameter(name="*", group="Fabrica LLM"),
    ] = field(default_factory=FabricaLLMSettings)
    "Settings for the OpenAI-compatible LLM used by the reward Deep Agent."

    editor_snapshot_settings: Annotated[
        FabricaEditorSnapshotSettings,
        Parameter(name="*", group="Fabrica editor"),
    ] = field(default_factory=FabricaEditorSnapshotSettings)
    "Settings for the optional Unreal Editor world snapshot."

    loop_settings: Annotated[
        FabricaLoopSettings,
        Parameter(name="*", group="Fabrica loop"),
    ] = field(default_factory=FabricaLoopSettings)
    "Outer Fabrica loop: iterations, samples, and agent step budget."

    code_ignore_globs: Annotated[
        List[str],
        Parameter(parse=False, show=False),
    ] = field(
        default_factory=lambda: [
            "**/Binaries/**",
            "**/DerivedDataCache/**",
            "**/Intermediate/**",
            "**/.git/**",
        ]
    )
    training_settings: Annotated[
        Sb3TrainingSettings,
        Parameter(group="Training Arguments", name="*"),
    ] = field(default_factory=Sb3TrainingSettings)
    logging_settings: Annotated[
        Sb3LoggingSettings,
        Parameter(group="Logging Arguments", name="*"),
    ] = field(default_factory=Sb3LoggingSettings)
    resume_settings: Annotated[
        Sb3ResumeSettings,
        Parameter(group="Resume Arguments", name="*"),
    ] = field(default_factory=Sb3ResumeSettings)
    checkpoint_settings: Annotated[
        Sb3CheckpointSettings,
        Parameter(group="Checkpoint Arguments", name="*"),
    ] = field(default_factory=Sb3CheckpointSettings)
    network_architecture_settings: Annotated[
        Sb3NetworkArchitectureSettings,
        Parameter(group="Network Architecture Arguments", name="*"),
    ] = field(default_factory=Sb3NetworkArchitectureSettings)
    algorithm_settings: Annotated[
        Union[PPOTrainSettings, SACTrainSettings],
        Parameter(show=False, parse=False),
    ] = field(default_factory=PPOTrainSettings)
    custom_callbacks: Annotated[
        list[Any],
        IgnoreParameter,
    ] = field(default_factory=list)
    "Same role as :attr:`~schola.scripts.sb3.train.settings.Sb3TrainScriptSettings.custom_callbacks` (``stable_baselines3.common.callbacks.BaseCallback``). Not exposed on the CLI."

    environment_settings: Annotated[
        EnvironmentSettings,
        Parameter(group="Environment Arguments", name="*"),
    ] = field(default_factory=EnvironmentSettings)

    def resolved_uproject_path(self) -> Optional[Path]:
        """``.uproject`` from the ``project`` simulator settings."""
        sim = self.environment_settings.simulator_settings
        if isinstance(sim, UnrealProjectSimulatorConfig):
            return sim.uproject_path.resolve()
        return None

    def resolved_snapshot_map(self) -> Optional[str]:
        """Map to load for the editor world snapshot (from simulator settings)."""
        sim = self.environment_settings.simulator_settings
        if isinstance(
            sim, (UnrealProjectSimulatorConfig, UnrealExecutableSimulatorConfig)
        ):
            return sim.map
        return None

    @cached_property
    def resolved_code_roots(self) -> List[Path]:
        roots: List[Path] = [
            self.paths_settings.env_header.resolve().parent,
        ]
        uproject = self.resolved_uproject_path()
        if uproject is not None:
            roots.append(uproject.parent.resolve())
        roots.extend(self.paths_settings.code_roots)
        return _collapse_nested_code_roots(roots)


def _collapse_nested_code_roots(roots: List[Path]) -> List[Path]:
    """Drop duplicate paths and descendants when a shorter ancestor root is present."""
    unique: List[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    unique.sort(key=lambda p: (len(p.parts), str(p)))
    kept: List[Path] = []

    for candidate in unique:
        if any(candidate.is_relative_to(kept_root) for kept_root in kept):
            continue
        kept.append(candidate)
    return kept


def make_sb3_train_settings(
    run: FabricaScriptSettings,
    artifact_dir: Path,
    executable_simulator_settings: UnrealExecutableSimulatorConfig,
    *,
    log_name: str = "sb3",
) -> Sb3TrainScriptSettings:
    """
    Build an :class:`~schola.scripts.sb3.train.settings.Sb3TrainScriptSettings` from a fabrica run.

    When the run uses :class:`~schola.scripts.common.settings.UnrealProjectSimulatorConfig`,
    the Unreal build runs first and the returned settings use
    :class:`~schola.scripts.common.settings.UnrealExecutableSimulatorConfig`.
    Build output is written to ``artifact_dir/unreal_build_log``.

    Call when SB3 training is enabled (algorithm is PPO or SAC).
    """
    log_dir = (artifact_dir / log_name).resolve()
    return Sb3TrainScriptSettings(
        training_settings=Sb3TrainingSettings(
            run.loop_settings.train_timesteps_per_sample,
            disable_eval=True, # we get our metrics from training rather than eval
        ),
        logging_settings=replace(run.logging_settings, log_dir=log_dir),
        resume_settings=run.resume_settings,
        checkpoint_settings=run.checkpoint_settings,
        network_architecture_settings=run.network_architecture_settings,
        algorithm_settings=run.algorithm_settings,
        custom_callbacks=run.custom_callbacks,
        environment_settings=replace(
            run.environment_settings,
            simulator_settings=executable_simulator_settings,
        ),
    )
