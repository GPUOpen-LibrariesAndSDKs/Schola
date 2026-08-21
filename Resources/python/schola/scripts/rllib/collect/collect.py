# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Collect imitation demonstrations in RLlib's offline episode format."""

from __future__ import annotations

import logging
from pathlib import Path
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


def main(args: RllibCollectScriptSettings) -> Path:
    """Collect a fixed-size RLlib Parquet dataset."""
    from schola.core.error_manager import ScholaErrorContextManager
    from schola.core.protocols.protobuf.offline_grpc_protocol import (
        GrpcImitationProtocol,
    )
    from schola.rllib.collector import RllibImitationCollector
    from schola.rllib.offline import write_offline_dataset

    output = args.collection_settings.output
    if output is None:
        raise ValueError("--output is required.")

    environment = args.environment_settings
    collector: RllibImitationCollector | None = None
    try:
        with ScholaErrorContextManager():
            protocol = GrpcImitationProtocol(
                url=environment.protocol_settings.url,
                port=environment.protocol_settings.port,
            )
            collector = RllibImitationCollector(
                protocol=protocol,
                simulator=environment.simulator_settings.make(),
                seed=environment.seed,
                options=environment.env_options or None,
            )
            logger.info(
                "Collecting %s RLlib demonstration steps to %s.",
                args.collection_settings.num_steps,
                output,
            )
            episodes = collector.collect(args.collection_settings.num_steps)
            write_offline_dataset(
                episodes,
                output,
                collector.observation_space,
                collector.action_space,
                episodes_per_shard=args.collection_settings.episodes_per_shard,
            )
            logger.info("Wrote RLlib offline dataset to %s", output)
            return output
    finally:
        if collector is not None:
            collector.close()


_collect_app = App(
    name="collect",
    help="Collect imitation demonstrations in RLlib's offline episode format.",
)


class CollectRllibCommand(ScholaCommandTemplate[RllibCollectScriptSettings]):
    @property
    def algorithm_table(self) -> dict[str, type[Any]]:
        return {}

    @property
    def script_args_type(self) -> type[RllibCollectScriptSettings]:
        return RllibCollectScriptSettings

    @property
    def main_func(self) -> Callable[[RllibCollectScriptSettings], Path]:
        return main


app = CollectRllibCommand(_collect_app, logger).make()

if __name__ == "__main__":
    app.meta()
