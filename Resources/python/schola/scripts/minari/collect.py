# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Script to collect imitation learning datasets using Minari and Schola.
"""

import logging
from typing import Any, Callable, Literal, Type
from typing_extensions import Annotated

from schola.scripts.common.command_template import ScholaCommandTemplate
from schola.scripts.common.console import configure_logging
from schola.scripts.minari.settings import MinariScriptSettings
from cyclopts import App, Parameter

configure_logging()

logger = logging.getLogger(__name__)


def main(args: MinariScriptSettings):
    """
    Main function for collecting a Minari dataset using Schola.

    Parameters
    ----------
    args : MinariScriptSettings
        The arguments for the script.

    Returns
    -------
    str
        The dataset ID of the created dataset.
    """
    import os

    import gymnasium as gym
    from schola.core.error_manager import ScholaErrorContextManager
    from schola.core.protocols.protobuf.offline_grpc_protocol import (
        GrpcImitationProtocol,
    )
    from schola.minari.datacollector import ScholaDataCollector

    cached_dataset_path = os.environ.get("MINARI_DATASETS_PATH", None)
    # Initialize so we can force closure at the end
    collector = None
    dataset = None
    try:
        # Set the Minari dataset directory if specified
        if args.collection_settings.data_path is not None:
            os.environ["MINARI_DATASETS_PATH"] = str(
                args.collection_settings.data_path.resolve()
            )
            logger.info(
                f"Setting MINARI_DATASETS_PATH to {args.collection_settings.data_path}"
            )

        # This context manager redirects GRPC errors into custom error types to help debug
        with ScholaErrorContextManager() as err_ctxt:

            # Create the protocol
            protocol = GrpcImitationProtocol(
                url=args.environment_settings.protocol_settings.url,
                port=args.environment_settings.protocol_settings.port,
            )

            # Create the simulator
            simulator = args.environment_settings.simulator_settings.make()

            # Create the data collector
            collector = ScholaDataCollector(
                protocol=protocol,
                simulator=simulator,
                seed=args.collection_settings.seed,
                record_infos=args.collection_settings.record_infos,
            )

            logger.info(
                f"Collecting {args.collection_settings.num_steps} steps for dataset '{args.collection_settings.dataset_id}'"
            )

            # Collect data
            for _ in range(args.collection_settings.num_steps):
                collector.step()

            logger.info("Data collection complete. Creating dataset...")

            if not args.collection_settings.dataset_id:
                import uuid

                args.collection_settings.dataset_id = f"schola-{uuid.uuid4()}"
                logger.info(
                    f"No dataset_id provided, generating a uuid-based dataset_id: {args.collection_settings.dataset_id}"
                )
            # Create the dataset
            dataset = collector.create_dataset(
                dataset_id=args.collection_settings.dataset_id,
                algorithm_name=args.collection_settings.algorithm_name,
                author=args.collection_settings.author,
                author_email=args.collection_settings.author_email,
                code_permalink=args.collection_settings.code_permalink,
                description=args.collection_settings.description,
            )

            logger.info(
                f"Dataset '{dataset.spec.dataset_id}' created successfully with {dataset.total_steps} steps and {dataset.total_episodes} episodes"
            )

    finally:
        if collector:
            collector.close()
        # restore the environment variable if it was set, and clear it if it was not set
        if cached_dataset_path:
            os.environ["MINARI_DATASETS_PATH"] = cached_dataset_path
        elif args.collection_settings.data_path is not None:
            os.environ.pop("MINARI_DATASETS_PATH")

    return dataset


_collect_app = App(
    name="collect", help="Collect imitation learning datasets using Minari"
)


class CollectMinariCommand(ScholaCommandTemplate[MinariScriptSettings]):

    @property
    def algorithm_table(self):
        return {}

    @property
    def script_args_type(self) -> Type[MinariScriptSettings]:
        return MinariScriptSettings

    @property
    def main_func(self) -> Callable[[MinariScriptSettings], Any]:
        return main


collect_app = CollectMinariCommand(_collect_app, logger).make()

if __name__ == "__main__":
    collect_app.meta()
