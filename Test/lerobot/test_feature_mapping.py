# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

import gymnasium as gym
import numpy as np
import pytest

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION
from lerobot_env_schola.config import ScholaObservationConfig
from lerobot_env_schola.feature_mapping import (
    as_source_tuple,
    infer_features,
    is_image_policy_key,
)


def test_is_image_policy_key_matches_image_and_images_prefix():
    assert is_image_policy_key("observation.image")
    assert is_image_policy_key("observation.images.front")
    assert not is_image_policy_key("observation.state")
    assert not is_image_policy_key("observation.images")


def test_to_policy_mapping_returns_source_tuples():
    mapping = ScholaObservationConfig(
        {
            "images": {"front": "observation.sensors.front_camera"},
            "state": [
                "observation.robot.joint_positions",
                "observation.robot.joint_velocities",
            ],
            "environment_state": "observation.target",
        }
    ).to_policy_mapping()

    assert mapping == {
        "observation.images.front": ("observation.sensors.front_camera",),
        "observation.state": (
            "observation.robot.joint_positions",
            "observation.robot.joint_velocities",
        ),
        "observation.environment_state": ("observation.target",),
    }


def test_as_source_tuple_wraps_strings_and_sequences():
    assert as_source_tuple("observation.state", "observation.joints") == (
        "observation.joints",
    )
    assert as_source_tuple("observation.state", ["a", "b"]) == ("a", "b")
    assert as_source_tuple("observation.state", ("a",)) == ("a",)


def test_infer_features_uses_policy_keys_and_identity_map():
    features, features_map = infer_features(
        policy_sources={
            "observation.state": (
                "observation.robot.joints",
                "observation.robot.gripper",
            ),
            "observation.images.front": ("observation.cameras.front",),
        },
        source_spaces={
            "observation.robot.joints": gym.spaces.Box(
                -1, 1, shape=(2,), dtype=np.float32
            ),
            "observation.robot.gripper": gym.spaces.Discrete(3),
            "observation.cameras.front": gym.spaces.Box(
                0, 255, shape=(3, 8, 10), dtype=np.uint8
            ),
        },
        action_space=gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
    )

    assert features == {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(5,)),
        "observation.images.front": PolicyFeature(
            type=FeatureType.VISUAL, shape=(8, 10, 3)
        ),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
    }
    assert features_map == {
        "observation.state": "observation.state",
        "observation.images.front": "observation.images.front",
        ACTION: ACTION,
    }


def test_infer_features_warns_about_unmapped_sources(caplog):
    with caplog.at_level("WARNING", logger="lerobot_env_schola.feature_mapping"):
        infer_features(
            policy_sources={"observation.state": ("observation.joints",)},
            source_spaces={
                "observation.joints": gym.spaces.Box(
                    -1, 1, shape=(2,), dtype=np.float32
                ),
                "observation.unused": gym.spaces.Box(
                    -1, 1, shape=(1,), dtype=np.float32
                ),
            },
            action_space=gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
        )

    assert "observation.unused" in caplog.text
    assert "will be ignored" in caplog.text


def test_infer_features_warns_about_reused_sources(caplog):
    camera = gym.spaces.Box(0, 1, shape=(3, 8, 8), dtype=np.float32)
    with caplog.at_level("WARNING", logger="lerobot_env_schola.feature_mapping"):
        infer_features(
            policy_sources={
                "observation.images.front": ("observation.camera",),
                "observation.images.wrist": ("observation.camera",),
            },
            source_spaces={"observation.camera": camera},
            action_space=gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
        )

    assert "observation.camera" in caplog.text
    assert "reused by" in caplog.text


def test_infer_features_rejects_unsupported_camera_dtype():
    with pytest.raises(TypeError, match="float or uint8"):
        infer_features(
            policy_sources={"observation.images.front": ("observation.camera",)},
            source_spaces={
                "observation.camera": gym.spaces.Box(
                    0, 255, shape=(3, 8, 8), dtype=np.int32
                )
            },
            action_space=gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
        )
