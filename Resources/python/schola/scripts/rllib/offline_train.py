# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
from __future__ import annotations

"""Train RLlib's data-only algorithms (BC, MARWIL) on collected Parquet episodes."""

import logging
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, cast

from cyclopts import App, Parameter

from schola.scripts.common.command_template import ScholaCommandTemplate
from schola.scripts.common.settings import CheckpointSettings, get_activation_function
from schola.scripts.rllib.settings import (
    BCSettings,
    LoggingSettings,
    MARWILSettings,
    OfflineRllibAlgorithmSettings,
    ResourceSettings,
)
from schola.scripts.rllib.train.settings import (
    NetworkArchitectureSettings,
    ResumeSettings,
    TrainingSettings,
)
from schola.scripts.rllib.train.train import (
    OFFLINE_STOP_METRIC,
    ResourcePlan,
    TrainingPlan,
    _run_training,
)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

logger = logging.getLogger(__name__)


@dataclass
class OfflineRllibScriptSettings:
    """Shared script settings for ``schola rllib bc`` and ``schola rllib marwil``."""

    training_settings: Annotated[
        TrainingSettings, Parameter(group="Training Arguments", name="*")
    ] = field(default_factory=TrainingSettings)
    "Settings for configuring the training process."

    algorithm_settings: Annotated[
        OfflineRllibAlgorithmSettings,
        Parameter(group="Algorithm Arguments", name="*"),
    ] = field(default_factory=BCSettings)
    "Settings for the offline algorithm (BC or MARWIL)."

    logging_settings: Annotated[
        LoggingSettings, Parameter(group="Logging Arguments", name="*")
    ] = field(default_factory=LoggingSettings)
    "Settings for enabling logging and configuring the logging directory."

    resume_settings: Annotated[
        ResumeSettings, Parameter(group="Resume Arguments", name="*")
    ] = field(default_factory=ResumeSettings)
    "Settings for resuming training from a checkpoint."

    network_architecture_settings: Annotated[
        NetworkArchitectureSettings,
        Parameter(group="Network Architecture Arguments", name="*"),
    ] = field(default_factory=NetworkArchitectureSettings)
    "Settings for configuring the neural network architecture used for training."

    resource_settings: Annotated[
        ResourceSettings, Parameter(group="Resource Arguments", name="*")
    ] = field(default_factory=ResourceSettings)
    "Settings for configuring the resource allocation for the training process."

    checkpoint_settings: Annotated[
        CheckpointSettings, Parameter(group="Checkpoint Arguments", name="*")
    ] = field(default_factory=CheckpointSettings)
    "Settings for checkpoints."

    seed: int | None = None
    "Random seed passed to RLlib. Offline training does not launch a simulator."


@dataclass
class BCScriptSettings(OfflineRllibScriptSettings):
    """Script settings for Behaviour Cloning."""

    algorithm_settings: Annotated[
        BCSettings, Parameter(group="Algorithm Arguments", name="*")
    ] = field(default_factory=BCSettings)


@dataclass
class MARWILScriptSettings(OfflineRllibScriptSettings):
    """Script settings for MARWIL."""

    algorithm_settings: Annotated[
        MARWILSettings, Parameter(group="Algorithm Arguments", name="*")
    ] = field(default_factory=MARWILSettings)


def main_offline(args: OfflineRllibScriptSettings) -> Any:
    """
    Train BC or MARWIL on an RLlib Parquet dataset.

    No environment is created: observation and action spaces come from the
    dataset sidecar written by ``schola rllib collect``.
    """
    from schola.rllib.offline import load_offline_dataset

    offline = args.algorithm_settings
    if offline.input_path is None:
        raise ValueError("--input is required and must point to a collected dataset.")

    (
        data_path,
        training_observation_space,
        observation_space,
        action_space,
    ) = load_offline_dataset(offline.input_path)

    activation_fn = get_activation_function(
        args.network_architecture_settings.activation
    )

    config = (
        offline.rllib_config()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(
            observation_space=training_observation_space,
            action_space=action_space,
        )
        .framework("torch")
        .env_runners(num_env_runners=0)
        .resources(num_gpus=args.resource_settings.num_gpus)
        .learners(
            num_learners=args.resource_settings.num_learners,
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
        .offline_data(
            input_=[str(data_path)],
            input_read_episodes=True,
            input_read_batch_size=offline.input_read_batch_size,
            dataset_num_iters_per_learner=offline.dataset_num_iters_per_learner,
            input_read_method_kwargs={"num_cpus": offline.offline_read_cpus},
            map_batches_kwargs={
                "concurrency": offline.offline_data_workers,
                "num_cpus": 1,
            },
        )
        .training(
            lr=args.training_settings.learning_rate,
            gamma=args.training_settings.gamma,
            num_epochs=args.training_settings.num_epochs,
            train_batch_size_per_learner=args.training_settings.train_batch_size_per_learner,
            minibatch_size=args.training_settings.minibatch_size,
            **cast(Any, offline.get_settings_dict()),
        )
        .debugging(
            log_level=args.logging_settings.rllib_log_level,
            seed=args.seed,
        )
    )

    algo_class = config.algo_class
    if algo_class is None:
        raise RuntimeError("RLlib config did not define an algorithm class.")

    return _run_training(
        args,
        TrainingPlan(
            config=config,
            trainable=algo_class,
            stop={OFFLINE_STOP_METRIC: args.training_settings.timesteps},
            resource_plan=ResourcePlan.offline(args, offline),
            label=f"{offline.name} training on RLlib dataset '{offline.input_path}'",
            export_observation_space=observation_space,
            export_action_space=action_space,
        ),
    )


def _make_offline_command(
    name: str,
    help_text: str,
    settings_type: type[OfflineRllibScriptSettings],
) -> Any:
    app = App(name=name, help=help_text)

    class OfflineCommand(ScholaCommandTemplate[OfflineRllibScriptSettings]):
        @property
        def algorithm_table(self) -> dict[str, type[Any]]:
            return {}

        @property
        def simulator_table(self) -> dict[str, type[Any]]:
            return {}

        @property
        def script_args_type(self) -> type[OfflineRllibScriptSettings]:
            return settings_type

        @property
        def main_func(self) -> Callable[[OfflineRllibScriptSettings], Any]:
            return main_offline

    return OfflineCommand(app, logger).make()


bc_app = _make_offline_command(
    "bc",
    "Train a model using Behaviour Cloning (BC) on an RLlib Parquet dataset.",
    BCScriptSettings,
)
marwil_app = _make_offline_command(
    "marwil",
    "Train a model using MARWIL on an RLlib Parquet dataset. Set --beta 0 for plain behaviour cloning.",
    MARWILScriptSettings,
)
