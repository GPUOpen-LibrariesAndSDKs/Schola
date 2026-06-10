# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Evaluate a trained RLlib algorithm from a checkpoint using ``Algorithm.evaluate``.
"""

import logging
from typing import Any, Dict

from cyclopts import App
from schola.scripts.common.command_template import MetaNoAlgCommand
from schola.scripts.rllib.eval.settings import RllibEvalScriptSettings

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

logger = logging.getLogger(__name__)


def _apply_eval_episode_budget(algo: Any, n_episodes: int) -> None:
    """Best-effort override of evaluation length before ``Algorithm.evaluate``."""
    cfg = getattr(algo, "config", None)
    if cfg is None:
        return
    if not (
        hasattr(cfg, "evaluation_duration") and hasattr(cfg, "evaluation_duration_unit")
    ):
        return
    try:
        cfg.evaluation_duration = n_episodes
        cfg.evaluation_duration_unit = "episodes"
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("Could not override evaluation duration: %s", e)


def _build_env_config(args: RllibEvalScriptSettings) -> Dict[str, Any]:
    """Build a complete ``env_config`` from the CLI environment settings.

    The checkpoint's baked-in ``env_config`` is ignored so the CLI always wins
    (unset flags fall back to dataclass defaults). Mirrors the training script.
    """
    from schola.core.protocols.protobuf.grpc_protocol import GrpcProtocol
    from schola.core.simulators.unreal.executable_simulator import UnrealExecutable
    from schola.core.simulators.external_simulator import ExternalSimulator

    protocol_args = args.environment_settings.protocol_settings
    sim_args = args.environment_settings.simulator_settings
    primary_sim = sim_args.make()
    is_external = isinstance(primary_sim, ExternalSimulator)

    return {
        "protocol": GrpcProtocol,
        "protocol_args": {
            "url": protocol_args.url,
            "port": protocol_args.port,
            "credential_mode": protocol_args.credential_mode.value,
            "environment_start_timeout": protocol_args.environment_start_timeout,
        },
        "port_offset_mode": protocol_args.port_offset_mode.value,
        "simulator": ExternalSimulator if is_external else UnrealExecutable,
        "simulator_args": (
            primary_sim.get_simulator_args()
            if is_external
            else primary_sim.get_executable_args()
        ),
        "options": dict(args.environment_settings.env_options),
    }


def _apply_env_config(algo: Any, env_config: Dict[str, Any]) -> None:
    """Override the restored algorithm's ``env_config`` and rebuild its envs.

    ``from_checkpoint`` already built the env runners against the baked-in
    ``env_config``; rewriting the config and calling ``make_env()`` rebuilds
    them with the CLI settings. The env is therefore opened twice (by
    ``from_checkpoint`` then here).
    """

    def _rebuild(env_runner: Any) -> None:
        # The built config is frozen; copy unfrozen, then use the public setter.
        cfg = env_runner.config.copy(copy_frozen=False)
        cfg.environment(env_config=env_config)
        env_runner.config = cfg
        env_runner.make_env()

    # Training group always exists; evaluation group only when
    # ``evaluation_num_env_runners > 0`` was baked into the checkpoint.
    for group in (
        algo.env_runner_group,
        getattr(algo, "eval_env_runner_group", None),
    ):
        if group is not None:
            group.foreach_env_runner(_rebuild)


def main(args: RllibEvalScriptSettings) -> Dict[str, Any]:
    """
    Restore an RLlib ``Algorithm`` from ``checkpoint`` and run built-in evaluation.

    Parameters
    ----------
    args : RllibEvalScriptSettings
        CLI / script configuration.

    Returns
    -------
    dict
        RLlib evaluation ``ResultDict`` (metrics keys vary by Ray version).
    """

    import ray
    from ray.rllib.algorithms.algorithm import Algorithm

    if not args.resource_settings.using_cluster:
        ray.init(
            num_cpus=args.resource_settings.num_cpus,
            num_gpus=args.resource_settings.num_gpus,
        )
    else:
        if args.resource_settings.num_cpus > 1:
            logger.warning(
                "--num-cpus is non-default but connecting to an existing cluster; "
                "this parameter will be ignored."
            )
        if args.resource_settings.num_gpus > 0:
            logger.warning(
                "--num-gpus is non-default but connecting to an existing cluster; "
                "this parameter will be ignored."
            )

    try:
        algo = Algorithm.from_checkpoint(str(args.checkpoint.resolve()))
        _apply_eval_episode_budget(algo, args.n_eval_episodes)
        _apply_env_config(algo, _build_env_config(args))
        logger.info(
            "Running RLlib Algorithm.evaluate() for up to %d episodes (if supported by checkpoint config).",
            args.n_eval_episodes,
        )
        results = algo.evaluate()
        logger.info("Evaluation finished. Metrics: %s", results)
        algo.stop()
        return results
    finally:
        if not args.resource_settings.using_cluster:
            ray.shutdown()


app = App(name="eval", help="Evaluate a trained RLlib policy from a checkpoint")


class RllibEvalCommand(MetaNoAlgCommand[RllibEvalScriptSettings]):
    """Cyclopts wiring for ``schola rllib eval``."""

    pass


app = RllibEvalCommand(app, RllibEvalScriptSettings, main, logger).make()

if __name__ == "__main__":
    app.meta()
