# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All Rights Reserved.
from __future__ import annotations

"""
Script to train an rllib model using Schola.
"""

import logging

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol, cast

from schola.scripts.common.settings import (
    get_activation_function,
)
from schola.scripts.common.command_template import AlgorithmSpec, ScholaCommandTemplate

from schola.scripts.rllib.settings import (
    APPOSettings,
    PPOSettings,
    SACSettings,
    IMPALASettings,
    ResourceSettings,
)
from schola.scripts.rllib.train.settings import RllibScriptSettings

from cyclopts import App

if TYPE_CHECKING:
    import gymnasium as gym
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.tune import ExperimentAnalysis
    from schola.scripts.common.settings import CheckpointSettings
    from schola.scripts.rllib.settings import LoggingSettings, OfflineRllibAlgorithmSettings
    from schola.scripts.rllib.train.settings import ResumeSettings
# Logging setup
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
logger = logging.getLogger(__name__)


app = App(name="train", help="Train a Model using ray")
STOP_METRIC = "num_env_steps_sampled_lifetime"
# Offline algorithms never sample from an environment, so the online stop metric
# stays at zero and Tune would never stop. Count timesteps trained on instead.
OFFLINE_STOP_METRIC = "learners/__all_modules__/num_env_steps_trained_lifetime"


class TuneRunSettings(Protocol):
    """Fields ``_run_training`` reads from online or offline script settings."""

    resource_settings: ResourceSettings
    checkpoint_settings: CheckpointSettings
    resume_settings: ResumeSettings
    logging_settings: LoggingSettings


@dataclass(frozen=True)
class ResourcePlan:
    """Explicit Ray resource requirements for one training mode."""

    ray_cpus: int
    requested_cpus: int
    minimum_cpus: int
    description: str

    @classmethod
    def online(cls, args: TuneRunSettings) -> "ResourcePlan":
        return cls(
            ray_cpus=args.resource_settings.num_cpus,
            requested_cpus=args.resource_settings.num_cpus,
            minimum_cpus=args.resource_settings.num_cpus,
            description="online environment training",
        )

    @classmethod
    def offline(
        cls, args: TuneRunSettings, settings: OfflineRllibAlgorithmSettings
    ) -> "ResourcePlan":
        remote_learner_cpus = (
            args.resource_settings.num_learners
            * args.resource_settings.num_cpus_per_learner
        )
        minimum_cpus = (
            1
            + settings.offline_data_workers
            + settings.offline_read_cpus
            + remote_learner_cpus
        )
        return cls(
            ray_cpus=max(args.resource_settings.num_cpus, minimum_cpus),
            requested_cpus=args.resource_settings.num_cpus,
            minimum_cpus=minimum_cpus,
            description=(
                "offline training "
                f"(one Tune trial CPU, {settings.offline_data_workers} "
                f"pre-learner CPUs, {settings.offline_read_cpus} read CPUs, and "
                f"{remote_learner_cpus} remote learner CPUs)"
            ),
        )


@dataclass
class TrainingPlan:
    """The mode-specific values consumed by the common Tune lifecycle."""

    config: AlgorithmConfig
    trainable: type[Algorithm]
    stop: dict[str, int]
    resource_plan: ResourcePlan
    label: str
    export_observation_space: gym.Space[Any] | None = None
    export_action_space: gym.Space[Any] | None = None


def _get_restored_env_steps(checkpoint_path: Path | None) -> int:
    """
    Read the sampled environment-step count from an RLlib checkpoint.

    Used by ``_make_stop_criterion`` to support both stopping modes:
    the default absolute-lifetime-cap mode and the additive
    ``reset_timestep=True`` mode where ``--timesteps`` means additional
    steps beyond the checkpoint.

    RLlib stores this checkpoint metadata as pickle. Only pass checkpoints
    from trusted sources to --resume-from, because unpickling can execute code.
    """
    if checkpoint_path is None:
        return 0

    checkpoint_dir = (
        checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
    )
    env_runner_state_path = checkpoint_dir / "env_runner" / "state.pkl"

    if not env_runner_state_path.is_file():
        logger.warning(
            "Could not determine restored RLlib timestep count from %s. "
            + "Using --timesteps as a lifetime stop target.",
            checkpoint_path,
        )
        return 0

    try:
        import pickle

        with env_runner_state_path.open("rb") as state_file:
            state = pickle.load(state_file)
    except Exception as exc:
        logger.warning(
            "Could not read restored RLlib timestep count from %s: %s. "
            + "Using --timesteps as a lifetime stop target.",
            env_runner_state_path,
            exc,
        )
        return 0

    if not isinstance(state, dict):
        logger.warning(
            "Unexpected RLlib env runner state in %s. "
            + "Using --timesteps as a lifetime stop target.",
            env_runner_state_path,
        )
        return 0

    restored_steps = state.get("num_env_steps_sampled_lifetime", 0)
    try:
        restored_steps = int(restored_steps)
    except (TypeError, ValueError):
        logger.warning(
            "Unexpected RLlib restored timestep value %r in %s. "
            + "Using --timesteps as a lifetime stop target.",
            restored_steps,
            env_runner_state_path,
        )
        return 0
    if restored_steps < 0:
        logger.warning(
            "Unexpected negative RLlib restored timestep value %s in %s. "
            + "Using --timesteps as a lifetime stop target.",
            restored_steps,
            env_runner_state_path,
        )
        return 0
    return restored_steps


def _make_stop_criterion(
    timesteps: int,
    checkpoint_path: Path | None,
    reset_timestep: bool = False,
) -> dict[str, int]:
    """
    Build Ray Tune's stop criterion for ``num_env_steps_sampled_lifetime``.

    When ``reset_timestep`` is ``False`` (default), ``--timesteps`` is the
    absolute cap on lifetime env steps regardless of the checkpoint, matching
    RLlib/Tune behavior and allowing the same command to resume without
    hand-tuning totals.

    When ``reset_timestep`` is ``True``, the restored step count from the
    checkpoint is added to ``--timesteps`` so that ``--timesteps`` means
    "train for this many additional steps", matching SB3's
    ``reset_num_timesteps=True`` behaviour.
    """
    if checkpoint_path is not None:
        restored_timesteps = _get_restored_env_steps(checkpoint_path)
        if reset_timestep and restored_timesteps:
            target = timesteps + restored_timesteps
            logger.info(
                "Resuming from checkpoint with %s sampled environment steps so far; "
                "training for %s more steps until %s total lifetime steps.",
                restored_timesteps,
                timesteps,
                target,
            )
            return {STOP_METRIC: target}
        if restored_timesteps:
            logger.info(
                "Resuming from checkpoint with %s sampled environment steps so far; "
                "stop target remains %s total lifetime steps (--timesteps).",
                restored_timesteps,
                timesteps,
            )

    return {
        STOP_METRIC: timesteps,
    }


def _make_tune_callbacks(should_persist: bool) -> list[Any]:
    """Create the shared callback set for an RLlib Tune run."""
    if not should_persist:
        return []
    try:
        from ray.tune.logger import TBXLoggerCallback

        return [TBXLoggerCallback()]
    except ImportError:
        logger.warning(
            "tensorboardX is not installed; TensorBoard logging will be skipped. "
            "Install tensorboardX to enable TensorBoard logging with RLlib."
        )
        return []


def _run_training(args: TuneRunSettings, plan: TrainingPlan) -> ExperimentAnalysis:
    """Run mode-specific preparation through the one common Tune lifecycle."""
    import ray
    from ray import air, tune

    resources = args.resource_settings
    if not resources.using_cluster:
        if plan.resource_plan.ray_cpus != plan.resource_plan.requested_cpus:
            logger.warning(
                "%s needs at least %s CPUs, but --num-cpus is %s. "
                "Starting local Ray with %s CPUs.",
                plan.resource_plan.description,
                plan.resource_plan.minimum_cpus,
                plan.resource_plan.requested_cpus,
                plan.resource_plan.ray_cpus,
            )
        ray.init(
            num_cpus=plan.resource_plan.ray_cpus,
            num_gpus=resources.num_gpus,
        )
    else:
        logger.info(
            "Using an existing Ray cluster for %s; it must provide at least %s CPUs.",
            plan.resource_plan.description,
            plan.resource_plan.minimum_cpus,
        )
        if resources.num_cpus > 1 or resources.num_gpus > 0:
            logger.warning(
                "Resource flags are ignored while connecting to an existing Ray "
                "cluster; make sure it has capacity for %s.",
                plan.resource_plan.description,
            )

    checkpoint = args.checkpoint_settings
    logger.info("Starting %s", plan.label)
    try:
        results = tune.run(
            plan.trainable,
            config=plan.config,  # type: ignore
            stop=plan.stop,
            checkpoint_config=air.CheckpointConfig(  # pyright: ignore[reportArgumentType]
                checkpoint_frequency=(
                    checkpoint.save_freq if checkpoint.enable_checkpoints else 0
                ),
                checkpoint_at_end=(
                    checkpoint.save_final_policy or checkpoint.export_onnx
                ),
            ),
            restore=(
                str(args.resume_settings.resume_from)
                if args.resume_settings.resume_from
                else None
            ),
            verbose=args.logging_settings.rllib_verbosity,
            storage_path=checkpoint.storage_path,
            callbacks=_make_tune_callbacks(checkpoint.should_persist),
        )
        logger.info("Training complete")
    finally:
        if not resources.using_cluster:
            ray.shutdown()

    if checkpoint.export_onnx:
        from schola.rllib.export import export_onnx_from_training

        export_onnx_from_training(
            results,
            plan.export_observation_space,
            plan.export_action_space,
        )
    return results


def main_online(args: RllibScriptSettings) -> ExperimentAnalysis:
    """
    Train an online RLlib algorithm against a Schola environment.

    Parameters
    ----------
    args : RllibScriptSettings
        The arguments for the script as a dataclass

    Returns
    -------
    tune.ExperimentAnalysis
        The results of the training
    """
    # Import ray and rllib dependencies lazily when the command is actually executed
    from ray.rllib.policy.policy import PolicySpec
    from ray.rllib.algorithms.algorithm import Algorithm
    from schola.rllib.connectors import schola_env_to_module_flatten_connector
    from schola.rllib.env_runner import ScholaEnvRunner
    from schola.rllib.policy_mapping import (
        ENV_CONFIG_POLICY_MAPPING_RECORD_KEY,
        make_policy_mapping_fn_from_dict,
        schola_algorithm_subclass,
    )
    from schola.scripts.rllib.utils import discover_env_metadata
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

    from schola.scripts.common.settings import GymSimulatorConfig

    sim_args = args.environment_settings.simulator_settings
    n_sim = sim_args.num_simulators
    # Run locally if we are only running one simulator (including vectorized gym)
    num_env_runners = (
        0 if isinstance(sim_args, GymSimulatorConfig) or n_sim == 1 else n_sim
    )

    # Discover policy metadata + env_config via a temporary environment that is
    # always cleaned up, even if construction fails (no leaked Unreal process).
    agent_ids, agent_to_policy, env_config = discover_env_metadata(
        args.environment_settings,
        schola_verbosity=args.logging_settings.schola_verbosity,
    )

    policies = {}
    for agent_id in agent_ids:
        policy_id = agent_to_policy[agent_id]
        if policy_id not in policies:
            policies[policy_id] = PolicySpec()

    # Pass the frozen mapping to the ScholaAlgorithm via env_config (ignored by
    # make_env) so it gets checkpointed as an RLlib subcomponent.
    env_config[ENV_CONFIG_POLICY_MAPPING_RECORD_KEY] = dict(agent_to_policy)

    typed_policy_ids = {
        agent_id: policy_id
        for agent_id, policy_id in agent_to_policy.items()
        if policy_id != agent_id
    }
    if typed_policy_ids:
        logger.info(
            "Using RLlib AgentType policy mappings: %s",
            ", ".join(
                f"{agent_id}={policy_id}"
                for agent_id, policy_id in sorted(typed_policy_ids.items())
            ),
        )

    # Get activation function for model config
    activation_fn = get_activation_function(
        args.network_architecture_settings.activation
    )

    # make a new variable to get typing information
    algorithm_config: AlgorithmConfig = args.algorithm_settings.rllib_config()

    # Use NEW API stack with RayEnv/RayVecEnv (new stack interface)
    # Auto-assignment: RayEnv for local runner (num_env_runners=0), RayVecEnv for remote runners
    config = (
        algorithm_config.api_stack(
            enable_rl_module_and_learner=True,  # Enable new stack
            enable_env_runner_and_connector_v2=True,  # Enable EnvRunner
        )
        .environment(
            env_config=env_config,
        )
        .framework("torch")
        .env_runners(
            env_runner_cls=ScholaEnvRunner,
            num_env_runners=num_env_runners,
            env_to_module_connector=schola_env_to_module_flatten_connector,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=make_policy_mapping_fn_from_dict(agent_to_policy),  # type: ignore
        )
        .resources(
            num_gpus=args.resource_settings.num_gpus,
        )
        .learners(
            # When num_cpus=1, use 0 learners (local learning on main process)
            # This avoids resource conflicts with env_runner
            num_learners=(
                args.resource_settings.num_learners
                if args.resource_settings.num_learners > 0
                or args.resource_settings.num_cpus > 1
                else 0
            ),
            num_gpus_per_learner=args.resource_settings.num_gpus_per_learner,
            num_cpus_per_learner=args.resource_settings.num_cpus_per_learner,
        )
        .rl_module(
            model_config={
                "fcnet_hiddens": args.network_architecture_settings.fcnet_hiddens,
                "fcnet_activation": activation_fn,
                "use_lstm": args.network_architecture_settings.use_lstm,
                "lstm_cell_size": args.network_architecture_settings.lstm_cell_size,
                "max_seq_len": args.network_architecture_settings.max_seq_len,
            },
        )
        .training(
            lr=args.training_settings.learning_rate,
            gamma=args.training_settings.gamma,
            num_epochs=args.training_settings.num_epochs,
            train_batch_size_per_learner=args.training_settings.train_batch_size_per_learner,
            minibatch_size=args.training_settings.minibatch_size,
            **cast(Any, args.algorithm_settings.get_settings_dict()),
        )
        .debugging(
            log_level=args.logging_settings.rllib_log_level,
            seed=args.environment_settings.seed,
        )
    )  # type: ignore

    # Use the new API stack metric name for stopping criterion.
    # Old stack used "timesteps_total", new stack uses "num_env_steps_sampled_lifetime".
    stop = _make_stop_criterion(
        args.training_settings.timesteps,
        args.resume_settings.resume_from,
        args.resume_settings.reset_timestep,
    )

    ckpt = args.checkpoint_settings
    if ckpt.export_onnx and not ckpt.save_final_policy:
        logger.info(
            "export_onnx without save_final_policy: saving an end-of-run snapshot so "
            + "the exported model matches the final training weights (writes an end-of-run checkpoint)."
        )
    # Train through a Schola Algorithm subclass so the frozen policy mapping is
    # saved/restored as a native RLlib Checkpointable subcomponent,
    # mirroring RLlib's own checkpoint behavior.
    algo_class = config.algo_class
    if algo_class is None:
        raise RuntimeError("RLlib config did not define an algorithm class.")
    schola_algorithm_cls = schola_algorithm_subclass(cast(type[Algorithm], algo_class))
    return _run_training(
        args,
        TrainingPlan(
            config=config,
            trainable=schola_algorithm_cls,
            stop=stop,
            resource_plan=ResourcePlan.online(args),
            label="online RLlib training",
        ),
    )


def main(args: RllibScriptSettings) -> ExperimentAnalysis:
    """Run an RLlib training settings object from the Python API.

    CLI commands select their handler declaratively through ``AlgorithmSpec``.
    The narrow type dispatch here preserves the public ``main(settings)`` entry
    point for callers that construct settings objects themselves.
    """
    return main_online(args)


RLIB_TRAIN_ALGORITHMS: dict[str, AlgorithmSpec] = {
    "sac": AlgorithmSpec(
        SACSettings, "Train a model using Soft Actor-Critic (SAC) with RLlib."
    ),
    "ppo": AlgorithmSpec(
        PPOSettings,
        "Train a model using Proximal Policy Optimization (PPO) with RLlib.",
    ),
    "impala": AlgorithmSpec(IMPALASettings, "Train a model using IMPALA with RLlib."),
    "appo": AlgorithmSpec(
        APPOSettings,
        "Train a model using Asynchronous Proximal Policy Optimization (APPO) with RLlib.",
    ),
}


class RllibTrainCommand(ScholaCommandTemplate[RllibScriptSettings]):
    """
    ``ScholaCommandTemplate`` configuration for Ray RLlib (PPO, SAC, IMPALA, APPO).

    See Also
    --------
    ScholaCommandTemplate
    """

    @property
    def algorithm_specs(self) -> dict[str, AlgorithmSpec]:
        return RLIB_TRAIN_ALGORITHMS

    @property
    def script_args_type(self) -> type[RllibScriptSettings]:
        return RllibScriptSettings

    @property
    def main_func(self) -> Callable[[RllibScriptSettings], ExperimentAnalysis]:
        return main


app = RllibTrainCommand(app, logger).make()

if __name__ == "__main__":
    app.meta()
