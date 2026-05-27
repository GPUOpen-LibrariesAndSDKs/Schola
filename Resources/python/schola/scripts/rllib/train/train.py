# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Script to train an rllib model using Schola.
"""

import logging

from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

from schola.scripts.common.settings import (
    get_activation_function,
)
from schola.scripts.common.command_template import MetaAlgCommand

from schola.scripts.rllib.settings import (
    APPOSettings,
    PPOSettings,
    SACSettings,
    IMPALASettings,
)
from schola.scripts.rllib.train.settings import RllibScriptSettings

from cyclopts import App

# Logging setup
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
logger = logging.getLogger(__name__)


app = App(name="train", help="Train a Model using ray")
STOP_METRIC = "num_env_steps_sampled_lifetime"


def _get_restored_env_steps(checkpoint_path: Optional[Path]) -> int:
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
            "Using --timesteps as a lifetime stop target.",
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
            "Using --timesteps as a lifetime stop target.",
            env_runner_state_path,
            exc,
        )
        return 0

    if not isinstance(state, dict):
        logger.warning(
            "Unexpected RLlib env runner state in %s. "
            "Using --timesteps as a lifetime stop target.",
            env_runner_state_path,
        )
        return 0

    restored_steps = state.get("num_env_steps_sampled_lifetime", 0)
    try:
        restored_steps = int(restored_steps)
    except (TypeError, ValueError):
        logger.warning(
            "Unexpected RLlib restored timestep value %r in %s. "
            "Using --timesteps as a lifetime stop target.",
            restored_steps,
            env_runner_state_path,
        )
        return 0
    if restored_steps < 0:
        logger.warning(
            "Unexpected negative RLlib restored timestep value %s in %s. "
            "Using --timesteps as a lifetime stop target.",
            restored_steps,
            env_runner_state_path,
        )
        return 0
    return restored_steps


def _make_stop_criterion(
    timesteps: int,
    checkpoint_path: Optional[Path],
    reset_timestep: bool = False,
) -> Dict[str, int]:
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


# forward declare here for type hinting with no load
def main(args: RllibScriptSettings) -> "ray.tune.ExperimentAnalysis":
    """
    Main function for launching training with ray.

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
    import ray
    from ray import air, tune
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.policy.policy import PolicySpec
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.algorithms.sac.sac import SACConfig
    from ray.rllib.algorithms.appo.appo import APPOConfig
    from ray.rllib.algorithms.impala.impala import IMPALAConfig
    from ray.tune.registry import register_env
    from schola.rllib.export import export_onnx_from_policy
    from ray.rllib.policy.policy import Policy
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec, RLModule
    from ray.rllib.connectors.env_to_module import FlattenObservations
    from schola.rllib.env import RayVecEnv
    from schola.rllib.env_runner import ScholaEnvRunner
    from schola.core.protocols.protobuf.grpc_protocol import GrpcProtocol
    from schola.core.simulators.unreal.executable_simulator import UnrealExecutable
    from schola.core.simulators.external_simulator import ExternalSimulator
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

    sim_args = args.environment_settings.simulator_settings
    protocol_args = args.environment_settings.protocol_settings
    n_sim = sim_args.num_simulators
    # Run locally if we are only running one simulator
    num_env_runners = 0 if n_sim == 1 else n_sim

    # Space discovery: connect to a running UE instance to learn
    # observation/action shapes.  For ExternalSimulator the UE process is
    # already running; for other simulators we launch one temporarily.
    primary_sim = sim_args.make()
    tmp_env = RayVecEnv(
        protocol_args.make(),
        primary_sim,
        verbosity=args.logging_settings.schola_verbosity,
    )
    try:
        agent_ids = sorted(tmp_env.possible_agents)
        agent_types = dict(tmp_env.agent_types)
        policy_mapping_fn = tmp_env.make_policy_mapping_fn()
    finally:
        tmp_env.close()

    policies = {}
    for agent_id in agent_ids:
        policy_id = policy_mapping_fn(agent_id)
        if policy_id not in policies:
            policies[policy_id] = PolicySpec()

    typed_policy_ids = {
        agent_id: agent_type.strip()
        for agent_id, agent_type in agent_types.items()
        if agent_type.strip()
    }
    if typed_policy_ids:
        logger.info(
            "Using RLlib AgentType policy mappings: %s",
            ", ".join(
                f"{agent_id}={policy_id}"
                for agent_id, policy_id in sorted(typed_policy_ids.items())
            ),
        )

    # Clusters configure resources automatically
    if not args.resource_settings.using_cluster:
        ray.init(
            num_cpus=args.resource_settings.num_cpus,
            num_gpus=args.resource_settings.num_gpus,
        )
    else:
        if args.resource_settings.num_cpus > 1:
            logger.warning(
                "--num-cpus is a non-default value, but the script is connecting to an existing cluster. This parameter will be ignored."
            )
        if args.resource_settings.num_gpus > 0:
            logger.warning(
                "--num-gpus is a non-default value, but the script is connecting to an existing cluster. This parameter will be ignored."
            )

    # Get activation function for model config
    activation_fn = get_activation_function(
        args.network_architecture_settings.activation
    )

    # make a new variable to get typing information
    algorithm_config: AlgorithmConfig = args.algorithm_settings.rllib_config()

    # Use NEW API stack with RayEnv/RayVecEnv (new stack interface)
    # Auto-assignment: RayEnv for local runner (num_env_runners=0), RayVecEnv for remote runners
    config: Union[PPOConfig, SACConfig, APPOConfig, IMPALAConfig] = (
        algorithm_config.api_stack(
            enable_rl_module_and_learner=True,  # Enable new stack
            enable_env_runner_and_connector_v2=True,  # Enable EnvRunner
        )
        .environment(
            env_config={
                "protocol": GrpcProtocol,
                "protocol_args": {
                    "url": protocol_args.url,
                    "port": protocol_args.port,
                    "credential_mode": protocol_args.credential_mode.value,
                    "environment_start_timeout": protocol_args.environment_start_timeout,
                },
                "port_offset_mode": protocol_args.port_offset_mode.value,
                "simulator": (
                    ExternalSimulator
                    if isinstance(primary_sim, ExternalSimulator)
                    else UnrealExecutable
                ),
                "simulator_args": (
                    primary_sim.get_simulator_args()
                    if isinstance(primary_sim, ExternalSimulator)
                    else primary_sim.get_executable_args()
                ),
                "options": dict(args.environment_settings.env_options),
            },
        )
        .framework("torch")
        .env_runners(
            env_runner_cls=ScholaEnvRunner,
            num_env_runners=num_env_runners,
            env_to_module_connector=lambda env, spaces=None, device=None: FlattenObservations(
                input_observation_space=(
                    env.single_observation_space
                    if env is not None
                    else spaces["__env_single__"][0]
                ),
                input_action_space=(
                    env.single_action_space
                    if env is not None
                    else spaces["__env_single__"][1]
                ),
                multi_agent=True,
            ),
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,  # type: ignore
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
            train_batch_size=args.training_settings.train_batch_size_per_learner,
            minibatch_size=args.training_settings.minibatch_size,
            **args.algorithm_settings.get_settings_dict(),
        )
        .debugging(
            log_level=args.logging_settings.rllib_log_level,
        )
    )  # type: ignore

    # Use the new API stack metric name for stopping criterion.
    # Old stack used "timesteps_total", new stack uses "num_env_steps_sampled_lifetime".
    stop = _make_stop_criterion(
        args.training_settings.timesteps,
        args.resume_settings.resume_from,
        args.resume_settings.reset_timestep,
    )

    callbacks = []
    try:
        from ray.tune.logger import TBXLoggerCallback

        callbacks.append(TBXLoggerCallback())
    except ImportError:
        logger.warning(
            "tensorboardX is not installed; TensorBoard logging will be skipped. "
            "Install tensorboardX to enable TensorBoard logging with RLlib."
        )

    logger.info("Starting training")
    try:
        results = tune.run(
            args.algorithm_settings.name,
            config=config,  # type: ignore
            stop=stop,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=(
                    args.checkpoint_settings.save_freq
                    if args.checkpoint_settings.enable_checkpoints
                    else 0
                ),
                checkpoint_at_end=args.checkpoint_settings.save_final_policy,
            ),  # type: ignore
            restore=(
                str(args.resume_settings.resume_from)
                if args.resume_settings.resume_from
                else None
            ),
            verbose=args.logging_settings.rllib_verbosity,
            storage_path=str(args.checkpoint_settings.checkpoint_dir.resolve()),
            callbacks=callbacks,
        )
        last_checkpoint = results.get_last_checkpoint()
        logger.info("Training complete")
    finally:
        # Always shutdown ray and release the environment from training even if there is an error
        # will reraise the error unless a control flow statement is added
        if not args.resource_settings.using_cluster:
            ray.shutdown()

    if (
        args.checkpoint_settings.export_onnx
        and last_checkpoint
        and len(results.trials) > 0
        and results.trials[-1].path
    ):
        export_onnx_from_policy(
            Algorithm.from_checkpoint(last_checkpoint), Path(results.trials[-1].path)
        )
        logger.info("Models exported to ONNX at %s", results.trials[-1].path)
    return results


class RllibTrainCommand(MetaAlgCommand[RllibScriptSettings]):
    """
    ``MetaAlgCommand`` configuration for Ray RLlib (PPO, SAC, IMPALA, APPO).

    See Also
    --------
    MetaAlgCommand
    """

    @property
    def algorithm_table(self) -> Dict[str, Type[Any]]:
        return {
            "sac": SACSettings,
            "ppo": PPOSettings,
            "impala": IMPALASettings,
            "appo": APPOSettings,
        }

    @property
    def algorithm_help(self) -> Dict[str, str]:
        return {
            "sac": "Train a model using Soft Actor-Critic(SAC) with rllib.",
            "ppo": "Train a model using Proximal Policy Optimization(PPO) with rllib.",
            "impala": "Train a model using IMPALA with rllib.",
            "appo": "Train a model using Asynchronous Proximal Policy Optimization(APPO) with rllib.",
        }


app = RllibTrainCommand(app, RllibScriptSettings, main, logger).make()

if __name__ == "__main__":
    app.meta()
