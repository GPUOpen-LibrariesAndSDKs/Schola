# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Settings dataclasses for the ``schola env inspect`` command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from cyclopts import Parameter

from schola.scripts.common.settings import (
    AllSingularSimulatorConfigs,
    EnvironmentSettings,
    IgnoreParameter,
    BaseLoggingSettings,
    SingularExternalSimulatorConfig,
)


@dataclass
class EnvInspectLoggingSettings(BaseLoggingSettings):
    """
    Logging settings for ``schola env inspect``.
    """


@dataclass
class EnvInspectEnvironmentSettings(EnvironmentSettings[AllSingularSimulatorConfigs]):
    """
    Environment settings for ``schola env inspect``.
    """

    simulator_settings: Annotated[
        AllSingularSimulatorConfigs,
        IgnoreParameter,
    ] = field(default_factory=SingularExternalSimulatorConfig)


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
