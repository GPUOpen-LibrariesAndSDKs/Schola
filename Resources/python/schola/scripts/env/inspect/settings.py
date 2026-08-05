# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Settings dataclasses for the ``schola env inspect`` command.
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
class EnvInspectLoggingSettings:
    """
    Logging settings for ``schola env inspect``.
    """

    schola_verbosity: Annotated[
        int, Parameter(validator=validators.Number(gte=0, lte=2))
    ] = 0
    "Verbosity level for Schola-specific logging during environment inspection."


@dataclass
class EnvInspectEnvironmentSettings(EnvironmentSettings[AllSimulatorConfigs]):
    """
    Environment settings for ``schola env inspect``.
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
        EnvInspectLoggingSettings, Parameter(group="Logging Arguments", name="*")
    ] = field(default_factory=EnvInspectLoggingSettings)
    "Logging verbosity for Schola components."

    environment_settings: Annotated[
        EnvInspectEnvironmentSettings,
        Parameter(group="Environment Arguments", name="*"),
    ] = field(default_factory=EnvInspectEnvironmentSettings)
    "Simulator, protocol, seed, and reset options for the environment to inspect."
