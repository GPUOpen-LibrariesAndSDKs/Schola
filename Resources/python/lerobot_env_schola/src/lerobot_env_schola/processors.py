# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""LeRobot observation processing for Schola environments."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import RobotObservation
from lerobot.processor import ObservationProcessorStep, ProcessorStepRegistry
from lerobot_env_schola.feature_mapping import is_image_policy_key


@dataclass
class ScholaProcessorStep(ObservationProcessorStep):
    """Normalize policy-shaped observations emitted by the Schola vector adapter.

    The vector adapter applies source mapping before LeRobot tensorizes the
    observation. This step keeps configured policy keys and scales uint8
    images.
    """

    policy_keys: tuple[str, ...] = ()
    uint8_image_keys: tuple[str, ...] = ()

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Keep feature metadata unchanged.

        ``create_envs()`` derives the policy-facing metadata from the same
        inferred features before LeRobot creates this pipeline.
        """
        return features

    def observation(self, observation: RobotObservation) -> RobotObservation:
        """Keep policy tensors and normalize configured image values."""
        processed: RobotObservation = {}
        if "task" in observation:
            processed["task"] = observation["task"]

        uint8_images = set(self.uint8_image_keys)

        for policy_key in self.policy_keys:
            value = observation[policy_key]
            if is_image_policy_key(policy_key):
                image = value.float()
                if policy_key in uint8_images:
                    image = image / 255
                processed[policy_key] = image
            else:
                processed[policy_key] = value.float()

        return processed


ProcessorStepRegistry.register(name="schola_processor")(ScholaProcessorStep)
