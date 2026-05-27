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


def _apply_env_options(algo: Any, env_options: Dict[str, Any]) -> None:
    """Stage CLI ``--env-options.k=v`` on the restored algorithm's live envs.

    Eval differs from training: by the time we reach ``algo.evaluate()`` the
    env runners have already been constructed by ``Algorithm.from_checkpoint``
    against the checkpoint's baked-in ``env_config``, so re-writing
    ``env_config["options"]`` would not reach them. The mechanism that *does*
    reach them is ``foreach_env_runner(env.set_options(...))``: the next
    ``reset()`` that fires during ``algo.evaluate()`` then picks the options
    up one-shot, mirroring SB3's pattern.
    """
    if not env_options:
        return

    opts = dict(env_options)

    def _stage(env_runner: Any) -> None:
        env_runner.env.set_options(opts)

    # Training group always exists; evaluation group only when
    # ``evaluation_num_env_runners > 0`` was baked into the checkpoint.
    for group in (
        algo.env_runner_group,
        getattr(algo, "eval_env_runner_group", None),
    ):
        if group is not None:
            group.foreach_env_runner(_stage)


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
        algo = Algorithm.from_checkpoint(str(args.checkpoint))
        _apply_eval_episode_budget(algo, args.n_eval_episodes)
        _apply_env_options(algo, args.environment_settings.env_options)
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
