# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Shared configuration and Tune lifecycle for RLlib training commands."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Collection, Literal, Protocol, cast

from schola.scripts.common.settings import get_activation_function
from schola.scripts.rllib.settings import (
    LoggingSettings,
    OfflineRllibAlgorithmSettings,
    ResourceSettings,
    RllibAlgorithmSpecificSettings,
)
from schola.scripts.rllib.train.settings import (
    NetworkArchitectureSettings,
    ResumeSettings,
    TrainingSettings,
)

if TYPE_CHECKING:
    import gymnasium as gym
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.tune import ExperimentAnalysis

    from schola.scripts.common.settings import CheckpointSettings

logger = logging.getLogger(__name__)


class TuneRunSettings(Protocol):
    """Fields consumed by the shared Tune lifecycle."""

    resource_settings: ResourceSettings
    checkpoint_settings: CheckpointSettings
    resume_settings: ResumeSettings
    logging_settings: LoggingSettings


class TrainingConfigSettings(TuneRunSettings, Protocol):
    """Fields consumed by the shared RLlib training configuration."""

    training_settings: TrainingSettings
    network_architecture_settings: NetworkArchitectureSettings
    algorithm_settings: RllibAlgorithmSpecificSettings


@dataclass(frozen=True)
class ResourcePlan:
    """Explicit Ray resource requirements for one training mode."""

    ray_cpus: int
    requested_cpus: int
    minimum_cpus: int
    description: str

    @classmethod
    def online(cls, args: TuneRunSettings) -> ResourcePlan:
        return cls(
            ray_cpus=args.resource_settings.num_cpus,
            requested_cpus=args.resource_settings.num_cpus,
            minimum_cpus=args.resource_settings.num_cpus,
            description="online environment training",
        )

    @classmethod
    def offline(
        cls, args: TuneRunSettings, settings: OfflineRllibAlgorithmSettings
    ) -> ResourcePlan:
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
    """Mode-specific values consumed by the common Tune lifecycle."""

    config: AlgorithmConfig
    trainable: type[Algorithm]
    stop: dict[str, int]
    resource_plan: ResourcePlan
    label: str
    onnx_export_source: Literal["algorithm_checkpoint", "rl_module"] = (
        "algorithm_checkpoint"
    )
    export_observation_space: gym.Space[Any] | None = None
    export_action_space: gym.Space[Any] | None = None
    restore: Path | None = None
    warm_start_rl_module_dir: Path | None = None
    warm_start_policy_ids: Collection[str] = field(default_factory=tuple)


def configure_training(
    config: AlgorithmConfig,
    args: TrainingConfigSettings,
    *,
    seed: int | None,
) -> AlgorithmConfig:
    """Apply the model, optimizer, and debugging settings shared by train modes."""
    network = args.network_architecture_settings
    training = args.training_settings
    activation_fn = get_activation_function(network.activation)

    return (
        config.rl_module(
            model_config={
                "fcnet_hiddens": network.fcnet_hiddens,
                "fcnet_activation": activation_fn,
                "use_lstm": network.use_lstm,
                "lstm_cell_size": network.lstm_cell_size,
                "max_seq_len": network.max_seq_len,
            },
        )
        .training(
            lr=training.learning_rate,
            gamma=training.gamma,
            num_epochs=training.num_epochs,
            train_batch_size_per_learner=training.train_batch_size_per_learner,
            minibatch_size=training.minibatch_size,
            **cast(Any, args.algorithm_settings.get_settings_dict()),
        )
        .debugging(
            log_level=args.logging_settings.rllib_log_level,
            seed=seed,
        )
    )


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


def run_training(args: TuneRunSettings, plan: TrainingPlan) -> ExperimentAnalysis:
    """Run mode-specific preparation through the common Tune lifecycle."""
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
    config = plan.config
    if plan.warm_start_rl_module_dir is not None:
        from schola.rllib.checkpoint import make_warm_start_callback

        config = config.callbacks(
            on_algorithm_init=make_warm_start_callback(
                plan.warm_start_rl_module_dir,
                plan.warm_start_policy_ids,
            )
        )
        logger.info(
            "Warm-starting RLModule weights from %s after Algorithm construction. "
            "Optimizer state and lifetime env steps start fresh.",
            plan.warm_start_rl_module_dir,
        )

    logger.info("Starting %s", plan.label)
    try:
        results = tune.run(
            plan.trainable,
            config=config,  # type: ignore
            stop=plan.stop,
            checkpoint_config=air.CheckpointConfig(  # pyright: ignore[reportArgumentType]
                checkpoint_frequency=(
                    checkpoint.save_freq if checkpoint.enable_checkpoints else 0
                ),
                checkpoint_at_end=(
                    checkpoint.save_final_policy or checkpoint.export_onnx
                ),
            ),
            restore=str(plan.restore) if plan.restore else None,
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
            source=plan.onnx_export_source,
            observation_space=plan.export_observation_space,
            action_space=plan.export_action_space,
        )
    return results
