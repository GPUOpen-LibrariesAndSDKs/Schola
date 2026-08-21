# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Settings dataclasses for RLlib demonstration collection."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter, validators

from schola.scripts.rllib.settings import RllibEnvironmentSettings


@dataclass
class RllibCollectionSettings:
    """Parameters for writing an RLlib offline dataset."""

    output: Annotated[
        Path | None,
        Parameter(
            alias="-o",
            required=True,
            validator=validators.Path(file_okay=False, dir_okay=True),
        ),
    ] = None
    "Directory to create for RLlib episode Parquet shards."

    num_steps: Annotated[
        int, Parameter(alias="-t", validator=validators.Number(gte=1))
    ] = 1000
    "Number of environment steps to record."

    episodes_per_shard: Annotated[
        int, Parameter(validator=validators.Number(gte=1))
    ] = 64
    "Maximum number of recorded episodes stored in each Parquet shard."


@dataclass
class RllibCollectLoggingSettings:
    """Logging settings used while collecting demonstrations."""

    schola_verbosity: Annotated[
        int, Parameter(validator=validators.Number(gte=0, lte=2))
    ] = 0
    "Verbosity level for Schola collection logs."


@dataclass
class RllibCollectScriptSettings:
    """Top-level settings for ``schola rllib collect``."""

    collection_settings: Annotated[
        RllibCollectionSettings, Parameter(group="Collection Arguments", name="*")
    ] = field(default_factory=RllibCollectionSettings)

    logging_settings: Annotated[
        RllibCollectLoggingSettings, Parameter(group="Logging Arguments", name="*")
    ] = field(default_factory=RllibCollectLoggingSettings)

    environment_settings: Annotated[
        RllibEnvironmentSettings,
        Parameter(group="Environment Arguments", name="*"),
    ] = field(default_factory=RllibEnvironmentSettings)
