# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Settings dataclasses for the RLlib evaluation command.
"""

from typing import TYPE_CHECKING, Annotated
from pathlib import Path
from dataclasses import dataclass, field

from cyclopts import Parameter, validators

from schola.scripts.common.settings import (
    EnvironmentSettings,
    AllSimulatorConfigs,
    ExternalSimulatorConfig,
)

from schola.scripts.rllib.settings import (
    LoggingSettings,
    ResourceSettings,
    RllibEnvironmentSettings,
)

@dataclass
class RllibEvalScriptSettings:
    """
    Top-level settings for evaluating an RLlib checkpoint produced by Schola training.
    """

    checkpoint: Annotated[
        Path | None,
        Parameter(
            group="Evaluation Arguments",
            required=True,
            validator=validators.Path(exists=True, file_okay=True, dir_okay=True),
            alias="-r",
        ),
    ] = None
    "Path to a Ray Tune / RLlib checkpoint directory (for example ``.../checkpoint_000050``) (required)."

    n_eval_episodes: Annotated[
        int, Parameter(validator=validators.Number(gte=1), group="Evaluation Arguments")
    ] = 10
    "Number of episodes ``eval_main`` samples (each env runner may take more than one per round)."

    policy_map: Annotated[dict[str, str], Parameter(group="Evaluation Arguments")] = (
        field(default_factory=dict)
    )
    "Optional agent-to-policy overrides (for example ``--policy-map agent_0=Pawn``)."

    resource_settings: Annotated[
        ResourceSettings, Parameter(group="Resource Arguments", name="*")
    ] = field(default_factory=ResourceSettings)
    "Ray resource options for the short-lived evaluation process."

    logging_settings: Annotated[
        LoggingSettings, Parameter(group="Logging Arguments", name="*")
    ] = field(default_factory=LoggingSettings)
    "Logging verbosity for Schola and RLlib."

    environment_settings: Annotated[
        RllibEnvironmentSettings, Parameter(group="Environment Arguments", name="*")
    ] = field(default_factory=RllibEnvironmentSettings)
    "Settings for the environment to use during evaluation"
