# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Settings dataclasses for the ``schola env check`` command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from cyclopts import Parameter

from schola.scripts.common.settings import (
    AllSingularSimulatorConfigs,
    EnvironmentSettings,
    SingularExternalSimulatorConfig,
    IgnoreParameter,
    BaseLoggingSettings,
)


@dataclass
class EnvCheckLoggingSettings(BaseLoggingSettings):
    """
    Logging settings for ``schola env check``.
    """


@dataclass
class EnvCheckEnvironmentSettings(EnvironmentSettings[AllSingularSimulatorConfigs]):
    """
    Environment settings for ``schola env check``.
    """

    simulator_settings: Annotated[
        AllSingularSimulatorConfigs,
        IgnoreParameter,
    ] = field(default_factory=SingularExternalSimulatorConfig)


@dataclass
class EnvCheckScriptSettings:
    """
    Top-level settings for ``schola env check``.
    """

    logging_settings: Annotated[
        EnvCheckLoggingSettings, Parameter(group="Logging Arguments", name="*")
    ] = field(default_factory=EnvCheckLoggingSettings)
    "Logging verbosity for Schola components."

    environment_settings: Annotated[
        EnvCheckEnvironmentSettings,
        Parameter(group="Environment Arguments", name="*"),
    ] = field(default_factory=EnvCheckEnvironmentSettings)
    "Simulator, protocol, seed, and reset options for the environment to check."
