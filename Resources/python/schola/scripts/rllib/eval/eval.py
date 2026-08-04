# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Evaluate a trained RLlib checkpoint: load ``MultiRLModule`` weights, build an eval
``EnvRunnerGroup`` from CLI settings, and sample episodes.

Parallelism follows ``-n`` / ``num_simulators`` like training. The eval
``AlgorithmConfig`` is built from the CLI (not from the checkpoint pickle).

Policy routing priority: CLI ``policy_map``, then the checkpoint's
``schola_policy_mapping`` Checkpointable component, then a temporary environment
discovery pass (with warnings if the live env disagrees with an explicit CLI or
checkpoint map).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, cast

from cyclopts import App

from schola.scripts.common.command_template import ScholaCommandTemplate
from schola.scripts.rllib.eval.settings import RllibEvalScriptSettings

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
logger = logging.getLogger(__name__)

def _build_eval_config(
    env_config: dict[str, Any],
    *,
    num_env_runners: int,
    spec: Any,
    policies: dict[str, Any],
    policy_mapping_fn: Callable[..., str],
    rllib_log_level: str,
) -> Any:
    """Build an inference-only ``AlgorithmConfig`` for eval ``EnvRunnerGroup``.

    Wires ``ScholaEnvRunner`` and ``schola_env_to_module_flatten_connector`` like training.
    Uses CLI ``env_config``, ``-n``-derived ``num_env_runners``, and the restored
    ``rl_module_spec``. ``PPOConfig`` is only a concrete base (bare ``AlgorithmConfig``
    cannot supply a default module spec); the restored spec defines the network.
    """
    from ray.rllib.algorithms.ppo import PPOConfig

    from schola.rllib.connectors import schola_env_to_module_flatten_connector
    from schola.rllib.env_runner import ScholaEnvRunner

    return (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(env_config=env_config)
        .framework("torch")
        .env_runners(
            env_runner_cls=ScholaEnvRunner,
            num_env_runners=num_env_runners,
            env_to_module_connector=schola_env_to_module_flatten_connector,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,  # type: ignore
        )
        .debugging(log_level=rllib_log_level)
        .rl_module(rl_module_spec=spec)
    )

def _shape_env_runner_metrics(
    episode_returns: list[float], episode_lens: list[int]
) -> dict[str, Any]:
    """Turn parallel lists of per-episode return and length into an ``env_runners`` dict."""
    if not episode_returns or not episode_lens:
        raise RuntimeError("Cannot shape eval metrics: no episodes were collected.")
    if len(episode_returns) != len(episode_lens):
        raise ValueError(
            "Episode return and length counts must match: "
            f"{len(episode_returns)} returns vs {len(episode_lens)} lengths."
        )

    mean_ret = float(sum(episode_returns) / len(episode_returns))
    mean_len = float(sum(episode_lens) / len(episode_lens))

    return {
        "env_runners": {
            "episode_reward_mean": mean_ret,
            "episode_len_mean": mean_len,
            "num_episodes": float(len(episode_returns)),
            "hist_stats": {
                "episode_reward": list(episode_returns),
                "episode_lengths": list(episode_lens),
            },
        }
    }

def _sample_eval_episodes_via_env_runners(
    config: Any,
    marl: Any,
    n_eval_episodes: int,
    num_env_runners: int,
) -> tuple[list[float], list[int]]:
    """Sample up to ``n_eval_episodes`` through ``EnvRunnerGroup``.

    Starts envs from ``config``, loads ``marl`` weights into each runner, and samples
    with ``explore=False``. Returns ``(episode_returns, episode_lengths)``.

    Logs a warning and requests another round when a runner batch returns fewer
    episodes than still needed. Raises ``RuntimeError`` if a round returns no
    episodes.
    """
    import math

    from ray.rllib.core import COMPONENT_RL_MODULE
    from ray.rllib.env.env_runner_group import EnvRunnerGroup

    local_only = num_env_runners == 0
    group = EnvRunnerGroup(
        config=config,
        local_env_runner=local_only,
    )

    episode_returns: list[float] = []
    episode_lens: list[int] = []
    try:
        rl_module_state = marl.get_state(inference_only=True)
        group.foreach_env_runner(
            lambda r: r.set_state({COMPONENT_RL_MODULE: rl_module_state}), # type: ignore
            local_env_runner=local_only,
        )

        num_samplers = 1 if local_only else max(1, group.num_healthy_remote_workers())
        while len(episode_returns) < n_eval_episodes:
            remaining = n_eval_episodes - len(episode_returns)
            per_runner = max(1, math.ceil(remaining / num_samplers))
            sampled = group.foreach_env_runner(
                lambda r: r.sample(num_episodes=per_runner, explore=False),
                local_env_runner=local_only,
            )
            new_episodes = 0
            for episodes in sampled:
                for eps in episodes:
                    episode_returns.append(float(eps.get_return()))
                    episode_lens.append(int(eps.env_steps()))
                    new_episodes += 1
            if new_episodes == 0:
                raise RuntimeError(
                    "Env runners produced no episodes in a sampling round "
                    f"({len(episode_returns)}/{n_eval_episodes} collected)."
                )
            if new_episodes < remaining:
                logger.warning(
                    "Runners returned %d episode(s) this round but %d were needed "
                    "(%d/%d collected); topping up.",
                    new_episodes,
                    remaining,
                    len(episode_returns),
                    n_eval_episodes,
                )
    finally:
        try:
            group.stop()
        except Exception as exc:  # pragma: no cover - defensive cleanup
            logger.debug("EnvRunnerGroup stop failed: %s", exc)

    return episode_returns, episode_lens

def main(args: RllibEvalScriptSettings) -> dict[str, Any]:
    """Entry point for ``schola rllib eval``: load modules, run env sampling, return metrics.

    Returns a dict with an ``env_runners`` namespace (``episode_reward_mean``,
    ``hist_stats``, etc.). Raises ``RuntimeError`` if sampling does not complete
    the requested number of episodes.
    """
    import ray
    from ray.rllib.core.rl_module.multi_rl_module import (
        MultiRLModule,
        MultiRLModuleSpec,
    )
    from ray.rllib.policy.policy import PolicySpec

    from schola.rllib.checkpoint import rl_module_dir_from_algorithm_checkpoint
    from schola.rllib.policy_mapping import (
        make_policy_mapping_fn_from_dict,
        resolve_policy_mapping_for_eval,
    )
    from schola.scripts.rllib.utils import discover_env_metadata

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

    if args.checkpoint is None:
        raise ValueError("Checkpoint is required")
    ckpt = args.checkpoint.resolve()
    n_sim = args.environment_settings.simulator_settings.num_simulators
    num_env_runners = 0 if n_sim <= 1 else int(n_sim)
    cli_policy_map = args.policy_map or None

    try:
        rl_dir = rl_module_dir_from_algorithm_checkpoint(ckpt)
        logger.info("Loading MultiRLModule from %s", rl_dir)
        marl = cast(MultiRLModule, MultiRLModule.from_checkpoint(rl_dir))

        agent_ids, env_agent_to_policy, env_config = discover_env_metadata(
            args.environment_settings,
            schola_verbosity=args.logging_settings.schola_verbosity,
        )
        agent_to_policy = resolve_policy_mapping_for_eval(
            agent_ids=agent_ids,
            module_ids=marl.keys(),
            checkpoint=ckpt,
            env_agent_to_policy=env_agent_to_policy,
            cli_agent_to_policy=cli_policy_map,
        )

        spec = MultiRLModuleSpec.from_module(marl)
        policies = {module_id: PolicySpec() for module_id in marl.keys()}

        config = _build_eval_config(
            env_config,
            num_env_runners=num_env_runners,
            spec=spec,
            policies=policies,
            policy_mapping_fn=make_policy_mapping_fn_from_dict(agent_to_policy),
            rllib_log_level=args.logging_settings.rllib_log_level,
        )

        logger.info(
            "Evaluating with %d env runner(s) for %d episode(s).",
            num_env_runners or 1,
            args.n_eval_episodes,
        )
        episode_returns, episode_lens = _sample_eval_episodes_via_env_runners(
            config,
            marl,
            args.n_eval_episodes,
            num_env_runners,
        )
        results = _shape_env_runner_metrics(episode_returns, episode_lens)
        logger.info("Evaluation finished. Metrics: %s", results)
        return results
    finally:
        if not args.resource_settings.using_cluster:
            ray.shutdown()

app = App(name="eval", help="Evaluate a trained RLlib policy from a checkpoint")

class RllibEvalCommand(ScholaCommandTemplate[RllibEvalScriptSettings]):
    @property
    def algorithm_table(self) -> dict[str, type[Any]]:
        return {}

    @property
    def script_args_type(self) -> type[RllibEvalScriptSettings]:
        return RllibEvalScriptSettings

    @property
    def main_func(self) -> Callable[[RllibEvalScriptSettings], Any]:
        return main

app = RllibEvalCommand(app, logger).make()

if __name__ == "__main__":
    app.meta()
