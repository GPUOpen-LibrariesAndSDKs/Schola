# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Collect imitation datasets in RLlib's offline Parquet layout."""

from __future__ import annotations

import logging
from typing import Any, Callable

from cyclopts import App

from schola.scripts.common.command_template import ScholaCommandTemplate
from schola.scripts.rllib.collect.settings import RllibCollectScriptSettings

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

logger = logging.getLogger(__name__)


def main(args: RllibCollectScriptSettings) -> Any:
    """
    Launch a simulator and record demonstrations until the session ends.

    Parameters
    ----------
    args : RllibCollectScriptSettings
        Parsed collect command settings.

    Returns
    -------
    pathlib.Path
        Directory containing the written Parquet shards and space sidecar.
    """
    from pathlib import Path

    from schola.core.error_manager import ScholaErrorContextManager
    from schola.core.protocols.protobuf.offline_grpc_protocol import (
        GrpcImitationProtocol,
    )
    from schola.rllib.collector import RllibImitationCollector
    from schola.rllib.offline import write_offline_dataset

    collector: RllibImitationCollector | None = None
    if args.collection_settings.output is None:
        raise ValueError("--output is required.")
    output_dir = Path(args.collection_settings.output)
    try:
        with ScholaErrorContextManager():
            protocol = GrpcImitationProtocol(
                url=args.environment_settings.protocol_settings.url,
                port=args.environment_settings.protocol_settings.port,
            )
            simulator = args.environment_settings.simulator_settings.make()
            collector = RllibImitationCollector(
                protocol=protocol,
                simulator=simulator,
                seed=args.collection_settings.seed,
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


_collect_app = App(
    name="collect",
    help="Collect imitation learning datasets in RLlib's offline Parquet format",
)


class CollectRllibCommand(ScholaCommandTemplate[RllibCollectScriptSettings]):
    @property
    def algorithm_table(self) -> dict[str, type[Any]]:
        return {}

    @property
    def script_args_type(self) -> type[RllibCollectScriptSettings]:
        return RllibCollectScriptSettings

    @property
    def main_func(self) -> Callable[[RllibCollectScriptSettings], Any]:
        return main


collect_app = CollectRllibCommand(_collect_app, logger).make()

if __name__ == "__main__":
    collect_app.meta()
