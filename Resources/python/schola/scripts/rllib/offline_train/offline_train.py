# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Train RLlib's data-only algorithms from an existing offline dataset."""

from __future__ import annotations

import logging
from typing import Any, Callable

from cyclopts import App

from schola.scripts.common.command_template import ScholaCommandTemplate
from schola.scripts.rllib.offline_train.settings import OfflineRllibScriptSettings
from schola.scripts.rllib.settings import BCSettings, MARWILSettings
from schola.scripts.rllib.training import (
    ResourcePlan,
    TrainingPlan,
    configure_training,
    run_training,
)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

logger = logging.getLogger(__name__)

OFFLINE_STOP_METRIC = "learners/__all_modules__/num_env_steps_trained_lifetime"


def main(args: OfflineRllibScriptSettings) -> Any:
    """Train BC or MARWIL on an existing RLlib Parquet dataset."""
    from schola.rllib.offline import load_offline_dataset

    offline = args.algorithm_settings
    input_path = offline.input_path
    if input_path is None:
        raise ValueError("--input is required.")

    (
        data_path,
        training_observation_space,
        observation_space,
        action_space,
    ) = load_offline_dataset(input_path)

    config = configure_training(
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
        ),
        args,
        seed=args.seed,
    )

    algo_class = config.algo_class
    if algo_class is None:
        raise RuntimeError("RLlib config did not define an algorithm class.")

    return run_training(
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
            restore=args.resume_settings.resume_from,
        ),
    )


_offline_train_app = App(
    name="offline-train",
    help="Train BC or MARWIL from an existing RLlib offline dataset.",
)


class OfflineTrainCommand(ScholaCommandTemplate[OfflineRllibScriptSettings]):
    @property
    def algorithm_table(self) -> dict[str, type[Any]]:
        return {
            "bc": BCSettings,
            "marwil": MARWILSettings,
        }

    @property
    def algorithm_help(self) -> dict[str, str]:
        return {
            "bc": "Train a model using Behaviour Cloning on an existing RLlib dataset.",
            "marwil": "Train a model using MARWIL on an existing RLlib dataset.",
        }

    @property
    def simulator_table(self) -> dict[str, type[Any]]:
        return {}

    @property
    def script_args_type(self) -> type[OfflineRllibScriptSettings]:
        return OfflineRllibScriptSettings

    @property
    def main_func(self) -> Callable[[OfflineRllibScriptSettings], Any]:
        return main


app = OfflineTrainCommand(_offline_train_app, logger).make()

if __name__ == "__main__":
    app.meta()
