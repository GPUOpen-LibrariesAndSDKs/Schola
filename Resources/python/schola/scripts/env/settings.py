# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Shared settings dataclasses for Schola environment utility commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from cyclopts import Parameter, validators

from schola.scripts.common.settings import (
    AllSimulatorConfigs,
    EnvironmentSettings,
    ExternalSimulatorConfig,
    IgnoreParameter,
)


@dataclass
class EnvLoggingSettings:
    """
    Logging settings for Schola environment utility commands.
    """

    schola_verbosity: Annotated[
        int, Parameter(validator=validators.Number(gte=0, lte=2))
    ] = 0
    "Verbosity level for Schola-specific logging during environment inspection."


@dataclass
class EnvToolsEnvironmentSettings(EnvironmentSettings[AllSimulatorConfigs]):
    """
    Environment settings for Schola environment utility commands.
    """

    simulator_settings: Annotated[
        AllSimulatorConfigs,
        IgnoreParameter,
    ] = field(default_factory=ExternalSimulatorConfig)


@dataclass
class EnvInspectScriptSettings:
    """
    Top-level settings for ``schola env inspect``.
    """

    logging_settings: Annotated[
        EnvLoggingSettings, Parameter(group="Logging Arguments", name="*")
    ] = field(default_factory=EnvLoggingSettings)
    "Logging verbosity for Schola components."

    environment_settings: Annotated[
        EnvToolsEnvironmentSettings, Parameter(group="Environment Arguments", name="*")
    ] = field(default_factory=EnvToolsEnvironmentSettings)
    "Simulator, protocol, seed, and reset options for the environment to inspect."


@dataclass
class EnvCheckScriptSettings:
    """
    Top-level settings for ``schola env check``.
    """

    logging_settings: Annotated[
        EnvLoggingSettings, Parameter(group="Logging Arguments", name="*")
    ] = field(default_factory=EnvLoggingSettings)
    "Logging verbosity for Schola components."

    environment_settings: Annotated[
        EnvToolsEnvironmentSettings, Parameter(group="Environment Arguments", name="*")
    ] = field(default_factory=EnvToolsEnvironmentSettings)
    "Simulator, protocol, seed, and reset options for the environment to check."
