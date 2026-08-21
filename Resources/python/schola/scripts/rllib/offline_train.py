# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
from __future__ import annotations

"""Train RLlib's data-only algorithms (BC, MARWIL), optionally collecting first."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Callable, cast

from cyclopts import App, Parameter, validators

from schola.scripts.common.command_template import ScholaCommandTemplate
from schola.scripts.common.settings import (
    AllSimulatorConfigs,
    CheckpointSettings,
    get_activation_function,
)
from schola.scripts.rllib.settings import (
    BCSettings,
    LoggingSettings,
    MARWILSettings,
    OfflineRllibAlgorithmSettings,
    OfflineRllibEnvironmentSettings,
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
class OfflineCollectionSettings:
    """Parameters for recording demonstrations before offline training."""

    output: Annotated[
        Path | None,
        Parameter(
            alias="-o",
            validator=validators.Path(file_okay=False, dir_okay=True),
        ),
    ] = None
    "Directory to write RLlib episode Parquet shards. Required with a simulator subcommand. Must not already exist."

    max_steps: Annotated[int | None, Parameter(validator=validators.Number(gte=1))] = (
        None
    )
    "Optional safety cap on recorded environment steps. When omitted, collection continues until the simulator process or gRPC session ends."

    episodes_per_shard: Annotated[
        int, Parameter(validator=validators.Number(gte=1))
    ] = 64
    "Maximum number of recorded episodes stored in each Parquet shard."


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

    collection_settings: Annotated[
        OfflineCollectionSettings, Parameter(group="Collection Arguments", name="*")
    ] = field(default_factory=OfflineCollectionSettings)
    "Settings for recording demonstrations when a simulator subcommand is given."

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

    environment_settings: Annotated[
        OfflineRllibEnvironmentSettings,
        Parameter(group="Environment Arguments", name="*"),
    ] = field(default_factory=OfflineRllibEnvironmentSettings)
    "Settings for the environment. A simulator is bound only when a subcommand is selected."


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


def _resolve_dataset_path(args: OfflineRllibScriptSettings) -> Path:
    """Return the Parquet directory, collecting first when a simulator is bound."""
    simulator = args.environment_settings.simulator_settings
    output_dir = args.collection_settings.output
    input_dir = args.algorithm_settings.input_path

    if simulator is not None:
        if output_dir is None:
            raise ValueError(
                "--output is required when a simulator subcommand is provided."
            )
        if input_dir is not None:
            raise ValueError(
                "--input cannot be combined with a simulator subcommand; "
                "use --output as the dataset directory."
            )
        return _collect_offline_dataset(args, output_dir, simulator)

    if output_dir is not None:
        raise ValueError("--output requires a simulator subcommand.")
    if input_dir is None:
        raise ValueError(
            "--input is required when no simulator subcommand is provided."
        )
    return input_dir


def _collect_offline_dataset(
    args: OfflineRllibScriptSettings,
    output_dir: Path,
    simulator: AllSimulatorConfigs,
) -> Path:
    """Launch a simulator and write an RLlib Parquet dataset to *output_dir*."""
    from schola.core.error_manager import ScholaErrorContextManager
    from schola.core.protocols.protobuf.offline_grpc_protocol import (
        GrpcImitationProtocol,
    )
    from schola.rllib.collector import RllibImitationCollector
    from schola.rllib.offline import write_offline_dataset

    collector: RllibImitationCollector | None = None
    try:
        with ScholaErrorContextManager():
            protocol = GrpcImitationProtocol(
                url=args.environment_settings.protocol_settings.url,
                port=args.environment_settings.protocol_settings.port,
            )
            collector = RllibImitationCollector(
                protocol=protocol,
                simulator=simulator.make(),
                seed=args.environment_settings.seed,
                options=args.environment_settings.env_options or None,
            )
            logger.info(
                "Collecting RLlib demonstrations to %s until the simulator session ends.",
                output_dir,
            )
            episodes = collector.collect_until_closed(
                max_steps=args.collection_settings.max_steps
            )
            write_offline_dataset(
                episodes,
                output_dir,
                collector.observation_space,
                collector.action_space,
                episodes_per_shard=args.collection_settings.episodes_per_shard,
            )
            logger.info("Wrote RLlib offline dataset to %s", output_dir)
            return output_dir
    finally:
        if collector is not None:
            collector.close()


def main_offline(args: OfflineRllibScriptSettings) -> Any:
    """
    Train BC or MARWIL on an RLlib Parquet dataset.

    When a simulator subcommand is given, demonstrations are recorded to
    ``--output`` first. Otherwise ``--input`` must point at an existing dataset.
    """
    from schola.rllib.offline import load_offline_dataset

    offline = args.algorithm_settings
    data_path = _resolve_dataset_path(args)
    (
        data_path,
        training_observation_space,
        observation_space,
        action_space,
    ) = load_offline_dataset(data_path)

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
            seed=args.environment_settings.seed,
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
            label=f"{offline.name} training on RLlib dataset '{data_path}'",
            onnx_export_source="rl_module",
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
        def bind_default_simulator(self) -> bool:
            return False

        @property
        def script_args_type(self) -> type[OfflineRllibScriptSettings]:
            return settings_type

        @property
        def main_func(self) -> Callable[[OfflineRllibScriptSettings], Any]:
            return main_offline

    return OfflineCommand(app, logger).make()


bc_app = _make_offline_command(
    "bc",
    "Train a model using Behaviour Cloning (BC). Pass a simulator subcommand to record demonstrations first, or --input to train on an existing RLlib Parquet dataset.",
    BCScriptSettings,
)
marwil_app = _make_offline_command(
    "marwil",
    "Train a model using MARWIL. Pass a simulator subcommand to record demonstrations first, or --input to train on an existing dataset. Set --beta 0 for plain behaviour cloning.",
    MARWILScriptSettings,
)
