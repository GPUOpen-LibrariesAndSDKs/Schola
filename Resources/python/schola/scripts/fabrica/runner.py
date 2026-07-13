# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Outer Fabrica loop: snapshot → Deep Agent → SB3."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

from schola.scripts.fabrica.langchain_client import build_chat_model
from schola.scripts.common.console import maybe_tqdm

from schola.scripts.fabrica.episode_metrics_callback import FabricaEpisodeMetrics
from schola.scripts.fabrica import codegen
from schola.scripts.fabrica import world_snapshot
from schola.scripts.fabrica.agent_debug_md import write_agent_debug_markdown
from schola.scripts.fabrica.reward_deep_agent import (
    build_reward_agent_messages,
    run_reward_deep_agent,
)
from schola.scripts.fabrica.codegen import CodegenEnv
from schola.scripts.fabrica.codegen_validation import (
    validate_fabrica_codegen_data,
)

from schola.scripts.fabrica.sample_summary import FabricaSampleSummary

from schola.scripts.fabrica.settings import (
    FabricaScriptSettings,
)
from schola.scripts.fabrica import train_adapter

logger = logging.getLogger(__name__)


def log_run_settings(
    root: Path, uproject: Optional[Path], codegen_env: CodegenEnv
) -> None:
    (root / "settings.json").write_text(
        json.dumps(
            {
                "env_header": str(codegen_env.env_header_path),
                "generated_cpp": str(codegen_env.generated_cpp_path),
                "code_gen_folder": (str(codegen_env.code_gen_folder)),
                "uproject": str(uproject) if uproject is not None else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_fabrica_loop(settings: FabricaScriptSettings) -> None:
    """Execute iterations × samples of codegen + optional SB3 training."""

    uproject = settings.resolved_uproject_path()

    root = settings.paths_settings.run_artifact_dir
    root.mkdir(parents=True, exist_ok=True)
    codegen_env = codegen.CodegenEnv.from_env_header_and_code_gen_folder(
        env_header_path=settings.paths_settings.env_header,
        code_gen_folder=settings.paths_settings.code_gen_folder,
    )
    log_run_settings(root, uproject, codegen_env)

    stale_cleanup = codegen.cleanup_fabrica_run_artifacts(
        env_header_path=codegen_env.env_header_path,
        gen_path=codegen_env.generated_cpp_path,
    )
    if stale_cleanup.changed:
        logger.info(
            "Cleaned up stale Fabrica artifacts from a prior interrupted run "
            "(header changed=%s, generated C++ changed=%s).",
            stale_cleanup.header_changed,
            stale_cleanup.gen_cpp_changed,
        )

    # One editor snapshot at run start; ``snapshot_json_path`` is reused for every iteration.
    if settings.editor_snapshot_settings.enabled and uproject is not None:
        try:
            world_snapshot.run_world_snapshot(
                uproject,
                settings.paths_settings.snapshot_json_path,
                settings.editor_snapshot_settings,
                map=settings.resolved_snapshot_map(),
            )
            logger.info(
                "Editor world snapshot for this run saved to: %s",
                settings.paths_settings.snapshot_json_path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("World snapshot failed: %s", exc)

    try:
        decl_result = codegen.inject_env_header_declarations(codegen_env)
        if decl_result.changed:
            logger.info(
                "Injected temporary Fabrica hook declarations into %s",
                codegen_env.env_header_path,
            )
        _run_fabrica_iterations(
            settings,
            root,
            codegen_env,
        )
    finally:
        final_cleanup_result = codegen.cleanup_fabrica_run_artifacts(
            env_header_path=codegen_env.env_header_path,
            gen_path=codegen_env.generated_cpp_path,
        )
        if final_cleanup_result.header_changed:
            logger.info(
                "Removed temporary Fabrica hook declarations from %s",
                codegen_env.env_header_path,
            )
        if final_cleanup_result.gen_cpp_changed:
            logger.info(
                "Removed generated Fabrica file at %s",
                codegen_env.generated_cpp_path,
            )

    logger.info("fabrica run complete. Artifacts under %s", root)


def _run_fabrica_sample(
    settings: FabricaScriptSettings,
    context: CodegenEnv,
    model: "BaseChatModel",
    iteration: int,
    sample_index: int,
    sample_dir: Path,
    best_iteration: Optional[FabricaSampleSummary],
) -> Optional[FabricaSampleSummary]:
    try:
        agent_messages = build_reward_agent_messages(
            settings,
            feedback=best_iteration,
        )
        try:
            code_gen_result = run_reward_deep_agent(
                model,
                settings,
                messages=agent_messages,
                recursion_limit=settings.loop_settings.reward_agent_max_steps,
            )

        except Exception as exc:  # noqa: BLE001
            logger.exception("Reward agent failed")
            (sample_dir / "agent_error.txt").write_text(str(exc), encoding="utf-8")
            return None

        try:
            validate_fabrica_codegen_data(code_gen_result)
        except Exception as exc:
            logger.exception("Fabrica C++ validation failed")
            (sample_dir / "codegen_validation_error.txt").write_text(
                str(exc), encoding="utf-8"
            )
            return None

        try:
            codegen.write_gen_cpp(
                context=context,
                generated_code=code_gen_result,
                create_if_missing=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Fabrica C++ merge failed")
            (sample_dir / "codegen_merge_error.txt").write_text(
                str(exc), encoding="utf-8"
            )
            return None

        executable_simulator_settings = None
        try:
            executable_simulator_settings = (
                train_adapter.build_unreal_environment_from_settings(
                    settings,
                    sample_dir,
                )
            )

        except Exception as exc:  # noqa: BLE001
            logger.error("Unreal build failed. Skipping sample.")
            return None

        fabrica_sample_metrics: Optional[FabricaEpisodeMetrics] = None
        try:
            fabrica_sample_metrics = train_adapter.run_sb3_training_from_settings(
                settings,
                sample_dir,
                executable_simulator_settings,
            )

        except Exception as exc:  # noqa: BLE001
            logger.error("SB3 training failed. Skipping sample.")

        sample_summary = FabricaSampleSummary(
            sample_index,
            iteration,
            response=code_gen_result,
            metrics=fabrica_sample_metrics,
            messages=agent_messages,
        )
        write_agent_debug_markdown(
            sample_dir,
            sample_summary,
            iteration=iteration,
            sample_index=sample_index,
            env_class_name=context.class_name,
            policy_feedback_interval=settings.loop_settings.policy_feedback_interval,
        )
        return sample_summary

    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Fabrica sample failed unexpectedly (iteration %s, sample %s)",
            iteration,
            sample_index,
        )
        (sample_dir / "sample_error.txt").write_text(str(exc), encoding="utf-8")
        return None
    finally:
        try:
            codegen.clean_gen_cpp_regions(
                context=context,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not clean Fabrica regions on %s after sample %s: %s",
                context.generated_cpp_path,
                sample_index,
                exc,
            )


def _run_fabrica_iterations(
    settings: FabricaScriptSettings,
    root: Path,
    context: CodegenEnv,
) -> None:

    best_iteration: Optional[FabricaSampleSummary] = None
    model = build_chat_model(settings.llm_settings)

    with maybe_tqdm(settings.loop_settings.pbar) as tqdm:
        for it in tqdm(range(settings.loop_settings.iterations)):
            it_dir = root / f"iter_{it:03d}"
            it_dir.mkdir(parents=True, exist_ok=True)

            iteration_candidates: List[FabricaSampleSummary] = []
            for s in tqdm(range(settings.loop_settings.samples), leave=False):
                s_dir = it_dir / f"sample_{s:03d}"
                s_dir.mkdir(parents=True, exist_ok=True)
                sample_summary = _run_fabrica_sample(
                    settings,
                    context,
                    model,
                    it,
                    s,
                    s_dir,
                    best_iteration,
                )
                if sample_summary is not None:
                    iteration_candidates.append(sample_summary)
            valid_iterations = [
                x
                for x in iteration_candidates + [best_iteration]
                if x is not None and x.metrics is not None and x.metrics.episodes
            ]
            if valid_iterations:
                best_iteration = max(
                    valid_iterations, key=lambda x: x.metrics.mean().task_success
                )
                logger.info(
                    "Best Sample after iteration %s: (iteration: %s, sample: %s)",
                    it,
                    best_iteration.iteration_index,
                    best_iteration.sample_index,
                )
            else:
                logger.info("No valid samples after iteration %s", it)
