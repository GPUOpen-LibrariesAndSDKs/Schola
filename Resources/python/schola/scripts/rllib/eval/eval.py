# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Evaluate a trained RLlib checkpoint: load ``MultiRLModule`` weights, build an eval
``EnvRunnerGroup`` from CLI settings, and sample episodes.

Parallelism follows ``-n`` / ``num_simulators`` like training. The eval
``AlgorithmConfig`` is built from the CLI (not from the checkpoint pickle).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Type

from cyclopts import App
from schola.scripts.common.command_template import ScholaCommandTemplate
from schola.scripts.rllib.eval.settings import RllibEvalScriptSettings

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

logger = logging.getLogger(__name__)


def _rl_module_dir_from_algorithm_checkpoint(checkpoint: Path) -> Path:
    """Return the on-disk ``MultiRLModule`` root under an Algorithm checkpoint."""
    from ray.rllib.core import (
        COMPONENT_LEARNER,
        COMPONENT_LEARNER_GROUP,
        COMPONENT_RL_MODULE,
    )

    primary = (
        checkpoint / COMPONENT_LEARNER_GROUP / COMPONENT_LEARNER / COMPONENT_RL_MODULE
    )
    if primary.is_dir():
        return primary
    legacy = checkpoint / "learner" / COMPONENT_RL_MODULE
    if legacy.is_dir():
        return legacy
    raise FileNotFoundError(
        f"No RLModule checkpoint directory found under {checkpoint}. "
        "Expected `learner_group/learner/rl_module/` (new API stack checkpoint)."
    )


def _num_env_runners_from_settings(args: RllibEvalScriptSettings) -> int:
    """
    Map ``-n`` / ``num_simulators`` to ``num_env_runners`` (mirrors training).

    A single simulator runs on a local env runner (``num_env_runners == 0``);
    ``N > 1`` simulators run on ``N`` remote env runners (per-worker ports).
    """
    n_sim = getattr(args.environment_settings.simulator_settings, "num_simulators", 1)
    return 0 if n_sim <= 1 else int(n_sim)


def _initial_policy_mapping_fn_from_module_ids(marl: Any) -> Callable[..., str]:
    """Guess ``agent_id -> module_id`` using only restored ``marl`` module ids.

    One module: map every agent to that id (shared-policy checkpoints). Several
    modules: use ``agent_id`` when it is a module id, otherwise the first module id
    as a fallback until ``_refine_policy_mapping_from_runners`` can replace it for
    multi-module / AgentType-keyed runs.
    """
    module_ids = list(marl.keys())
    id_set = set(module_ids)
    fallback = module_ids[0] if module_ids else None

    def module_id_or_fallback_policy_mapping_fn(
        agent_id: Any, *args: Any, **kwargs: Any
    ) -> str:
        agent_id = str(agent_id)
        if agent_id in id_set:
            return agent_id
        return fallback

    return module_id_or_fallback_policy_mapping_fn


def _refine_policy_mapping_from_runners(
    group: Any, marl: Any, local_only: bool
) -> None:
    """Set each runner's ``policy_mapping_fn`` from the live env mapping rule.

    Uses ``RayVecEnv.make_policy_mapping_fn()`` (same AgentType / agent-id rule as
    training), checks outputs against ``marl`` module ids, and writes via unfrozen
    ``AlgorithmConfig.multi_agent``. Applies on the runner's next ``sample`` (new
    episodes read the updated config).
    """
    module_ids = set(marl.keys())

    # Delegate to the env's ``make_policy_mapping_fn`` (training rule).
    base_mapping_fns = group.foreach_env_runner(
        lambda r: r.env.make_policy_mapping_fn(),
        local_env_runner=local_only,
    )
    base_mapping_fn = next((fn for fn in base_mapping_fns if fn is not None), None)
    if base_mapping_fn is None:
        logger.warning(
            "No env runner returned a policy mapping fn (no healthy runners?); "
            "keeping the initial module-id-based policy mapping."
        )
        return

    def policy_mapping_fn(agent_id: Any, *args: Any, **kwargs: Any) -> str:
        policy_id = base_mapping_fn(agent_id, *args, **kwargs)
        if policy_id in module_ids:
            return policy_id
        raise KeyError(
            f"Agent {str(agent_id)!r} maps to policy {policy_id!r}, which is not among "
            f"the restored module ids {sorted(module_ids)}; module ids and AgentTypes "
            "disagree."
        )

    def _set_mapping(runner: Any) -> None:
        cfg = runner.config.copy(copy_frozen=False)
        cfg.multi_agent(policy_mapping_fn=policy_mapping_fn)
        runner.config = cfg

    group.foreach_env_runner(_set_mapping, local_env_runner=local_only)


def _build_eval_config(
    env_config: Dict[str, Any],
    *,
    num_env_runners: int,
    spec: Any,
    policies: Dict[str, Any],
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

    from schola.rllib.env_runner import (
        ScholaEnvRunner,
        schola_env_to_module_flatten_connector,
    )

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
    episode_returns: List[float], episode_lens: List[int]
) -> Dict[str, Any]:
    """Turn parallel lists of per-episode return and length into an ``env_runners`` dict.

    Empty input yields zero means and empty ``hist_stats`` (no division by zero).
    """
    mean_ret = (
        float(sum(episode_returns) / len(episode_returns)) if episode_returns else 0.0
    )
    mean_len = float(sum(episode_lens) / len(episode_lens)) if episode_lens else 0.0

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


def _collect_eval_metrics_via_env_runners(
    config: Any,
    marl: Any,
    n_eval_episodes: int,
    num_env_runners: int,
) -> Dict[str, Any]:
    """Run up to ``n_eval_episodes`` through ``EnvRunnerGroup``, then aggregate metrics.

    Starts envs from ``config``, loads ``marl`` weights into each runner, samples with
    ``explore=False``, and returns an ``env_runners``-shaped dict.
    """
    import math

    from ray.rllib.core import COMPONENT_RL_MODULE
    from ray.rllib.env.env_runner_group import EnvRunnerGroup

    local_only = num_env_runners == 0
    group = EnvRunnerGroup(
        config=config,
        local_env_runner=local_only,
    )

    episode_returns: List[float] = []
    episode_lens: List[int] = []
    try:
        # Multi-module: refine agent-to-module mapping from live runners (needs env).
        if len(marl.keys()) > 1:
            _refine_policy_mapping_from_runners(group, marl, local_only)

        rl_module_state = marl.get_state(inference_only=True)
        group.foreach_env_runner(
            lambda r: r.set_state({COMPONENT_RL_MODULE: rl_module_state}),
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
                logger.warning(
                    "Env runners produced no episodes in a sampling round; stopping early."
                )
                break
    finally:
        try:
            group.stop()
        except Exception as exc:  # pragma: no cover - defensive cleanup
            logger.debug("EnvRunnerGroup stop failed: %s", exc)

    return _shape_env_runner_metrics(episode_returns, episode_lens)


def main(args: RllibEvalScriptSettings) -> Dict[str, Any]:
    """Entry point for ``schola rllib eval``: load modules, run env sampling, return metrics.

    Returns a dict with an ``env_runners`` namespace (``episode_reward_mean``,
    ``hist_stats``, etc.).
    """
    import ray
    from ray.rllib.core.rl_module.multi_rl_module import (
        MultiRLModule,
        MultiRLModuleSpec,
    )
    from ray.rllib.policy.policy import PolicySpec

    from schola.scripts.rllib.utils import build_env_config

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

    ckpt = args.checkpoint.resolve()
    env_config = build_env_config(args.environment_settings)
    num_env_runners = _num_env_runners_from_settings(args)

    try:
        rl_dir = _rl_module_dir_from_algorithm_checkpoint(Path(ckpt))
        logger.info("Loading MultiRLModule from %s", rl_dir)
        marl = MultiRLModule.from_checkpoint(rl_dir)

        spec = MultiRLModuleSpec.from_module(marl)
        policies = {module_id: PolicySpec() for module_id in marl.keys()}
        policy_mapping_fn = _initial_policy_mapping_fn_from_module_ids(marl)

        config = _build_eval_config(
            env_config,
            num_env_runners=num_env_runners,
            spec=spec,
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            rllib_log_level=args.logging_settings.rllib_log_level,
        )

        logger.info(
            "Evaluating with %d env runner(s) for %d episode(s).",
            num_env_runners or 1,
            args.n_eval_episodes,
        )
        results = _collect_eval_metrics_via_env_runners(
            config,
            marl,
            args.n_eval_episodes,
            num_env_runners,
        )
        logger.info("Evaluation finished. Metrics: %s", results)
        return results
    finally:
        if not args.resource_settings.using_cluster:
            ray.shutdown()


app = App(name="eval", help="Evaluate a trained RLlib policy from a checkpoint")


class RllibEvalCommand(ScholaCommandTemplate[RllibEvalScriptSettings]):

    @property
    def algorithm_table(self) -> Dict[str, Type[Any]]:
        return {}


app = RllibEvalCommand(app, RllibEvalScriptSettings, main, logger).make()

if __name__ == "__main__":
    app.meta()
