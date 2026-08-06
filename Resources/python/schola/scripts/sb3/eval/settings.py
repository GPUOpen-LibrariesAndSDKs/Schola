# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Cyclopts dataclasses for evaluating a saved SB3 policy with Schola.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated
from pathlib import Path
from cyclopts import Parameter, validators
from schola.scripts.common.settings import (
    AllSimulatorConfigs,
    EnvironmentSettings,
    ExternalSimulatorConfig,
    IgnoreParameter,
)
from schola.scripts.sb3.settings import Sb3BaseLoggingSettings, Sb3EnvironmentSettings
from schola.scripts.sb3.settings import BasePPOSettings, BaseSACSettings


@dataclass
class Sb3EvalScriptSettings:
    """
    Top-level settings for evaluating a saved Stable-Baselines3 policy with Schola.
    """

    checkpoint: Annotated[
        Path | None,
        Parameter(
            group="Evaluation Arguments",
            validator=validators.Path(exists=True, file_okay=True, dir_okay=False),
            required=True,
            alias="-r",
        ),
    ] = None
    "Path to a trained policy checkpoint (``.zip``) produced by SB3 ``model.save()`` (required via ``--checkpoint``)."

    n_eval_episodes: Annotated[
        int, Parameter(validator=validators.Number(gte=1), group="Evaluation Arguments")
    ] = 10
    "Number of evaluation episodes for ``stable_baselines3.common.evaluation.evaluate_policy``."

    deterministic: bool = True
    "If True, the policy mean (or mode) is used; if False, stochastic actions are sampled."

    vecnormalize: Annotated[
        Path | None,
        Parameter(
            group="Evaluation Arguments",
            validator=validators.Path(exists=True, file_okay=True, dir_okay=False),
        ),
    ] = None
    "Optional ``VecNormalize`` statistics file (``.pkl``) saved alongside the policy."

    logging_settings: Annotated[
        Sb3BaseLoggingSettings, Parameter(group="Logging Arguments", name="*")
    ] = field(default_factory=Sb3BaseLoggingSettings)
    "Logging verbosity for Schola and SB3 components."

    environment_settings: Annotated[
        Sb3EnvironmentSettings, Parameter(group="Environment Arguments", name="*")
    ] = field(default_factory=Sb3EnvironmentSettings)
    "How to launch or attach to the Unreal simulator and gRPC protocol for the environment."

    algorithm_settings: Annotated[
        BasePPOSettings | BaseSACSettings, IgnoreParameter
    ] = field(default_factory=BasePPOSettings)
    "The algorithm used to train the checkpoint that is being evaluated."
