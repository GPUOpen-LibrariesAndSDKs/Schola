# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Run SB3 training for one Fabrica sample; episode metrics come from ``EpisodeMetricsCallback``."""

from __future__ import annotations

import logging
from pathlib import Path

from schola.scripts.common.console import supplemental_file_logging
from schola.core.utils.ubt import (
    expected_executable_path,
    resolve_build_dir,
    run_ubt_project_build,
)
from schola.scripts.common.settings import (
    UnrealExecutableSimulatorConfig,
    UnrealProjectSimulatorConfig,
)
from schola.scripts.fabrica.episode_metrics_callback import FabricaEpisodeMetrics
from schola.scripts.fabrica.settings import FabricaScriptSettings, make_sb3_train_settings

logger = logging.getLogger(__name__)


def build_unreal_environment_from_settings(
    settings: FabricaScriptSettings,
    artifact_dir: Path,
) -> UnrealExecutableSimulatorConfig:
    """
    Build the Unreal project for one Fabrica sample and return executable simulator settings.

    When ``settings.environment_settings.simulator_settings`` is
    :class:`~schola.scripts.common.settings.UnrealProjectSimulatorConfig`, the standalone
    executable is staged under ``artifact_dir``. UBT stdout/stderr are written to
    ``artifact_dir/unreal_build_log``.

    If the run already uses
    :class:`~schola.scripts.common.settings.UnrealExecutableSimulatorConfig`, that config is
    returned unchanged.
    """
    sim = settings.environment_settings.simulator_settings
    if not isinstance(sim, UnrealProjectSimulatorConfig):
        raise TypeError(
            "Fabrica requires Unreal project simulator settings "
            f"(got {type(sim).__name__})."
        )

    artifact_dir = artifact_dir.resolve()
    build_artifact_dir = artifact_dir / "unreal_build"
    build_artifact_dir.mkdir(parents=True, exist_ok=True)

    uproject_file = sim.uproject_path.resolve()
    
    # Build under the sample directory if no build directory is specified.
    if sim.build_dir is None:
        build_dir = build_artifact_dir
    else:
        build_dir = sim.resolved_build_dir

    logger.info(
        "Building Unreal project %s to sample directory %s",
        uproject_file,
        build_dir,
    )
    completed_build = run_ubt_project_build(
        uproject_file,
        build_dir,
        ubt_path=sim.ubt_path,
        map=sim.map,
    )
    (build_artifact_dir / "ubt_stdout.txt").write_text(
        completed_build.stdout.decode("utf-8", errors="replace"),
        encoding="utf-8",
    )
    (build_artifact_dir / "ubt_stderr.txt").write_text(
        completed_build.stderr.decode("utf-8", errors="replace"),
        encoding="utf-8",
    )

    if completed_build.returncode != 0:
        exception_message = (
            "Unreal build failed with return code "
            f"{completed_build.returncode}. See {build_artifact_dir} for UBT output."
        )
        raise RuntimeError(exception_message)

    executable_path = expected_executable_path(uproject_file, build_dir)
    if not executable_path.exists():
        raise FileNotFoundError(f"Build had valid return code but executable not found at {executable_path}")

    logger.info("Built Unreal executable for Fabrica sample: %s", executable_path)
    return UnrealExecutableSimulatorConfig(
        executable_path=executable_path,
        disable_script=sim.disable_script,
        headless=sim.headless,
        map=sim.map,
        fps=sim.fps,
        display_logs=sim.display_logs,
        num_simulators=sim.num_simulators,
    )


def run_sb3_training_from_settings(
    settings: FabricaScriptSettings,
    artifact_dir: Path,
    executable_simulator_settings: UnrealExecutableSimulatorConfig,
    log_name: str = "sb3",
) -> FabricaEpisodeMetrics:
    """
    Run ``schola.scripts.sb3.train.train.main`` in-process with per-sample ``log_dir``.

    Injects :class:`~schola.scripts.fabrica.episode_metrics_callback.EpisodeMetricsCallback`
    (plus SB3 train's built-in wiring) so mean return, reward-component sums, and the
    scalar task-success value are available from the callback after training without reading Monitor CSV.

    Expects ``executable_simulator_settings`` from
    :func:`build_unreal_environment_from_settings` when the Fabrica run uses
    :class:`~schola.scripts.common.settings.UnrealProjectSimulatorConfig`.
    """
    try:
        from schola.scripts.fabrica.episode_metrics_callback import (
            EpisodeMetricsCallback,
        )
        from schola.scripts.sb3.train.train import main as sb3_train_main
        from schola.scripts.sb3.utils import CustomProgressBarCallback
    except ImportError as e:
        raise RuntimeError(
            "Schola SB3 training is not available in this environment. "
            "Install SB3 extras, e.g. `pip install 'schola[sb3]'`."
        ) from e

    log_dir = (artifact_dir / log_name).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    sb3_settings = make_sb3_train_settings(settings, artifact_dir, executable_simulator_settings, log_name=log_name)

    episode_metrics_cb = EpisodeMetricsCallback(
        reward_component_prefix=settings.loop_settings.fabrica_info_prefix,
        task_success_key=settings.loop_settings.fabrica_task_success_key,
    )
    merged_callbacks: list = [episode_metrics_cb, *settings.custom_callbacks]
    if settings.loop_settings.pbar:
        try:
            import tqdm  # noqa: F401
        except ImportError:
            logger.warning("tqdm not installed. Disabling SB3 progress bar.")
        else:
            merged_callbacks.append(CustomProgressBarCallback(leave=False))
    logger.info("SB3 training (main) log_dir=%s", log_dir)
    sb3_settings.custom_callbacks = merged_callbacks
    with supplemental_file_logging(log_dir / "sb3.log"):
        sb3_train_main(sb3_settings)

    return episode_metrics_cb.metrics
