# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Settings dataclasses for RLlib offline training."""

from typing import Annotated
from dataclasses import dataclass, field

from cyclopts import Parameter

from schola.scripts.common.settings import CheckpointSettings

from schola.scripts.rllib.settings import (
    BCSettings,
    LoggingSettings,
    MARWILSettings,
    ResourceSettings,
)
from schola.scripts.rllib.train.settings import (
    NetworkArchitectureSettings,
    ResumeSettings,
    TrainingSettings,
)


@dataclass
class OfflineRllibScriptSettings:
    """Settings shared by BC and MARWIL offline training."""

    training_settings: Annotated[
        TrainingSettings, Parameter(group="Training Arguments", name="*")
    ] = field(default_factory=TrainingSettings)
    "Settings for configuring the training process."

    algorithm_settings: Annotated[
        BCSettings | MARWILSettings,
        Parameter(show=False, parse=False),
    ] = field(default_factory=BCSettings)
    "Settings for the selected offline algorithm."

    logging_settings: Annotated[
        LoggingSettings, Parameter(group="Logging Arguments", name="*")
    ] = field(default_factory=LoggingSettings)
    "Settings for enabling logging and configuring the logging directory."

    resume_settings: Annotated[
        ResumeSettings, Parameter(group="Resume Arguments", name="*")
    ] = field(default_factory=ResumeSettings)
    "Settings for resuming training from a checkpoint."

    network_architecture_settings: Annotated[
        NetworkArchitectureSettings,
        Parameter(group="Network Architecture Arguments", name="*"),
    ] = field(default_factory=NetworkArchitectureSettings)
    "Settings for configuring the neural network architecture used for training."

    resource_settings: Annotated[
        ResourceSettings, Parameter(group="Resource Arguments", name="*")
    ] = field(default_factory=ResourceSettings)
    "Settings for configuring the resource allocation for the training process."

    checkpoint_settings: Annotated[
        CheckpointSettings, Parameter(group="Checkpoint Arguments", name="*")
    ] = field(default_factory=CheckpointSettings)
    "Settings for checkpoints."

    seed: Annotated[int | None, Parameter(group="Training Arguments")] = None
    "Random seed for RLlib's offline learner. None leaves its RNG unseeded."
