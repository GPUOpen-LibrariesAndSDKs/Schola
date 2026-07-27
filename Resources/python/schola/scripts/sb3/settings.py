# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Shared SB3 script settings (logging, algorithm base types, and launcher extensions).
"""

from __future__ import annotations

from typing import Annotated, List, Optional, Type, Union, Any, Dict

from schola.scripts.common.settings import (
    ActivationFunctionEnum,
    EnvironmentSettings,
    CheckpointSettings,
    Sb3LauncherExtension,
    AllSimulatorConfigs,
    ExternalSimulatorConfig,
    IgnoreParameter,
)
from dataclasses import dataclass, field
from pathlib import Path
from cyclopts import App, Parameter, validators

from cyclopts import types


@dataclass
class Sb3BaseLoggingSettings:
    """
    Dataclass for configuring logging settings for SB3 Scripts.
    """

    schola_verbosity: Annotated[
        int, Parameter(validator=validators.Number(gte=0, lte=2))
    ] = 0
    "Verbosity level for Schola-specific logging. This controls the level of detail in the output from Schola-related components during training."

    sb3_verbosity: Annotated[
        int, Parameter(validator=validators.Number(gte=0, lte=2))
    ] = 1
    "Verbosity level for Stable Baselines3 logging. This controls the level of detail in the output from Stable Baselines3 components during training."


@dataclass
class BaseSACSettings:

    @property
    def constructor(self) -> Type["SAC"]:  # type: ignore
        from stable_baselines3 import SAC

        return SAC

    @property
    def critic_type(self) -> str:
        return "qf"

    @property
    def name(self) -> str:
        return "SAC"


@dataclass
class BasePPOSettings:

    @property
    def constructor(self) -> Type["PPO"]:  # type: ignore
        from stable_baselines3 import PPO

        return PPO

    @property
    def critic_type(self) -> str:
        return "vf"

    @property
    def name(self) -> str:
        return "PPO"

@dataclass
class Sb3EnvironmentSettings(EnvironmentSettings[AllSimulatorConfigs]):
    """
    Dataclass for configuring the environment settings for SB3 Scripts.
    """
    
    simulator_settings: Annotated[
        AllSimulatorConfigs,
        IgnoreParameter,
    ] = field(default_factory=ExternalSimulatorConfig)