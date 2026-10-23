# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Convert pretrained LeRobot policies to ONNX for Unreal Engine."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor.normalize_processor import (
    NormalizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.utils.constants import OBS_PREFIX
from schola.core.model import ScholaModel
from schola.core.model_space_config import load_model_spaces

from .act import LeRobotACTScholaModule


def get_schola_lerobot_model(
    policy: PreTrainedPolicy,
    observation_space: gym.Space,
    action_space: gym.Space,
    normalizer: NormalizerProcessorStep | None = None,
    unnormalizer: UnnormalizerProcessorStep | None = None,
) -> ScholaModel:
    """Wrap a supported LeRobot policy for Schola ONNX export."""
    if isinstance(policy, ACTPolicy):
        return LeRobotACTScholaModule(
            policy,
            observation_space,
            action_space,
            normalizer=normalizer,
            unnormalizer=unnormalizer,
        )
    raise TypeError(
        "Schola ONNX export currently supports only LeRobot ACT "
        f"({ACTPolicy.__name__}), got {type(policy).__name__}."
    )


def check_observations_cover_features(
    observation_space: gym.spaces.Dict, input_features: Mapping[str, Any]
) -> None:
    """Raise if the spaces YAML lacks an observation for a policy input feature."""
    expected = {key.removeprefix(OBS_PREFIX) for key in input_features}
    if missing := expected - set(observation_space.keys()):
        raise ValueError(
            f"The spaces YAML is missing the policy's input features "
            f"{sorted(missing)}. Nest each observation under its feature name, "
            "for example 'observation: {state: {type: box, ...}}'."
        )


def convert_pretrained_to_onnx(
    pretrained_path: str,
    export_path: str,
    spaces_path: str,
) -> None:
    """Load a pretrained LeRobot policy and export it to ONNX for Unreal."""
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    observation_space, action_space = load_model_spaces(spaces_path)
    config = PreTrainedConfig.from_pretrained(pretrained_path)
    check_observations_cover_features(observation_space, config.input_features)
    config.device = "cpu"
    policy = get_policy_class(config.type).from_pretrained(
        pretrained_path, config=config
    )
    preprocessor, postprocessor = make_pre_post_processors(
        config, pretrained_path=pretrained_path
    )
    normalizer = next(
        (s for s in preprocessor.steps if isinstance(s, NormalizerProcessorStep)),
        None,
    )
    unnormalizer = next(
        (s for s in postprocessor.steps if isinstance(s, UnnormalizerProcessorStep)),
        None,
    )
    model = get_schola_lerobot_model(
        policy,
        observation_space,
        action_space,
        normalizer=normalizer,
        unnormalizer=unnormalizer,
    )
    model.save_as_onnx(export_path)
