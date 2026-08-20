# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Cyclopts dataclasses for collecting RLlib offline datasets with Schola."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter, validators

from schola.scripts.common.settings import (
    AllSimulatorConfigs,
    EnvironmentSettings,
    ExternalSimulatorConfig,
    IgnoreParameter,
)


@dataclass
class RllibCollectionSettings:
    """Parameters for recording an RLlib offline demonstration dataset."""

    output: Annotated[
        Path | None,
        Parameter(
            alias="-o",
            required=True,
            validator=validators.Path(file_okay=False, dir_okay=True),
        ),
    ] = None
    "Directory to write RLlib episode Parquet shards and the space sidecar. Must not already exist."

    max_steps: Annotated[
        int | None, Parameter(validator=validators.Number(gte=1), alias="-t")
    ] = None
    "Optional safety cap on recorded environment steps. When omitted, collection continues until the simulator process or gRPC session ends."

    seed: int | None = None
    "Random seed forwarded to the environment on startup. If None, the environment uses its own seed."

    episodes_per_shard: Annotated[
        int, Parameter(validator=validators.Number(gte=1))
    ] = 64
    "Maximum number of recorded episodes stored in each Parquet shard."


@dataclass
class RllibCollectLoggingSettings:
    """Logging settings for RLlib data collection."""

    schola_verbosity: Annotated[
        int, Parameter(validator=validators.Number(gte=0, lte=2))
    ] = 0
    "Verbosity level for Schola-specific logging during data collection."


@dataclass
class RllibCollectEnvironmentSettings(EnvironmentSettings[AllSimulatorConfigs]):
    """Environment settings for RLlib data collection."""

    simulator_settings: Annotated[
        AllSimulatorConfigs,
        IgnoreParameter,
    ] = field(default_factory=ExternalSimulatorConfig)


@dataclass
class RllibCollectScriptSettings:
    """Top-level settings for ``schola rllib collect``."""

    collection_settings: Annotated[
        RllibCollectionSettings, Parameter(group="Collection Arguments", name="*")
    ] = field(default_factory=RllibCollectionSettings)
    "Settings for configuring the dataset collection process."

    logging_settings: Annotated[
        RllibCollectLoggingSettings, Parameter(group="Logging Arguments", name="*")
    ] = field(default_factory=RllibCollectLoggingSettings)
    "Settings for configuring logging during data collection."

    environment_settings: Annotated[
        RllibCollectEnvironmentSettings, Parameter(group="Environment Arguments", name="*")
    ] = field(default_factory=RllibCollectEnvironmentSettings)
    "Settings for configuring the environment."
