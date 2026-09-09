# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.spaces import Box, Dict, Discrete

from lerobot_env_schola.feature_mapping import as_source_tuple
from lerobot_env_schola.vector_env import (
    LeRobotScholaVectorEnv,
    _build_policy_observation_space,
    _build_source_spaces,
    _normalize_info,
    _parse_success_array,
    _parse_success_string,
)
from Test.gym.testing_env import GenericTestVectorEnv


@pytest.fixture
def make_source_mapping():
    """Index a Schola observation space and expand policy sources to tuples."""

    def _make(observation_space, policy_sources):
        policy_sources = {
            key: as_source_tuple(key, sources)
            for key, sources in policy_sources.items()
        }
        source_spaces = _build_source_spaces(observation_space)
        return {
            "observation_space": observation_space,
            "policy_sources": policy_sources,
            "source_spaces": source_spaces,
            "gym_space": _build_policy_observation_space(policy_sources, source_spaces),
        }

    return _make


@pytest.fixture
def make_wrapper(make_source_mapping):
    wrappers = []

    def _make(
        *,
        num_envs=2,
        observation_space=None,
        action_space=None,
        policy_sources=None,
        success_key=None,
        render_camera=None,
        render_fps=30,
    ) -> LeRobotScholaVectorEnv:
        if observation_space is None:
            observation_space = Dict(
                {
                    "camera": Box(0, 1, shape=(3, 8, 8), dtype=np.float32),
                    "joints": Box(-1, 1, shape=(3,), dtype=np.float32),
                }
            )
        if action_space is None:
            action_space = Dict(
                {
                    "arm": Box(-1, 1, shape=(2,), dtype=np.float32),
                    "gripper": Box(-1, 1, shape=(1,), dtype=np.float32),
                }
            )
        if policy_sources is None:
            policy_sources = {
                "observation.images.front": "observation.camera",
                "observation.state": "observation.joints",
            }
        env = GenericTestVectorEnv(
            num_envs=num_envs,
            observation_space=observation_space,
            action_space=action_space,
        )
        try:
            mapping = make_source_mapping(observation_space, policy_sources)
            wrapper = LeRobotScholaVectorEnv(
                env,
                task="reach",
                task_description="Reach the target.",
                max_episode_steps=50,
                policy_sources=mapping["policy_sources"],
                success_key=success_key,
                render_fps=render_fps,
            )
            wrapper.set_render_camera(render_camera)
        except Exception:
            env.close()
            raise
        wrappers.append(wrapper)
        return wrapper

    yield _make

    for wrapper in reversed(wrappers):
        wrapper.close()


def test_schola_observation_helpers_preserve_schola_sources(
    make_wrapper, make_source_mapping
):
    mapping = make_source_mapping(
        Dict(
            {
                "camera": Box(0, 255, shape=(3, 2, 2), dtype=np.uint8),
                "robot": Dict({"mode": Discrete(3)}),
            }
        ),
        {
            "observation.images.front": "observation.camera",
            "observation.state": "observation.robot.mode",
        },
    )
    gym_space = mapping["gym_space"]

    assert set(mapping["source_spaces"]) == {
        "observation.camera",
        "observation.robot.mode",
    }
    assert set(gym_space) == {"images.front", "state"}
    assert gym_space["images.front"].shape == (3, 2, 2)
    assert gym_space["state"].shape == (3,)

    wrapper = make_wrapper(
        observation_space=mapping["observation_space"],
        policy_sources=mapping["policy_sources"],
    )
    converted = wrapper._convert_schola_observation(
        {
            "camera": np.arange(24, dtype=np.uint8).reshape(2, 3, 2, 2),
            "robot": {"mode": np.array([0, 2])},
        }
    )

    assert set(converted) == {"images.front", "state"}
    assert converted["images.front"].shape == (2, 3, 2, 2)
    np.testing.assert_array_equal(
        converted["state"],
        np.array([[1, 0, 0], [0, 0, 1]], dtype=np.int64),
    )
    assert wrapper.uint8_image_keys == ("observation.images.front",)


def test_wrapper_reports_float_image_keys_as_not_uint8(make_wrapper):
    wrapper = make_wrapper()
    assert wrapper.uint8_image_keys == ()


def test_root_box_observation_maps_directly_to_policy_key(
    make_wrapper, make_source_mapping
):
    mapping = make_source_mapping(
        Box(-1, 1, shape=(4,), dtype=np.float32),
        {"observation.state": "observation"},
    )
    gym_space = mapping["gym_space"]

    assert set(mapping["source_spaces"]) == {"observation"}
    assert set(gym_space) == {"state"}
    assert gym_space["state"].shape == (4,)

    wrapper = make_wrapper(
        observation_space=mapping["observation_space"],
        policy_sources=mapping["policy_sources"],
    )
    values = np.arange(8, dtype=np.float32).reshape(2, 4)
    converted = wrapper._convert_schola_observation(values)
    assert set(converted) == {"state"}
    np.testing.assert_array_equal(converted["state"], values)

    observation, _ = wrapper.reset()
    assert observation["state"].shape == (2, 4)
    assert wrapper.single_observation_space["state"] == Box(
        -1, 1, shape=(4,), dtype=np.float32
    )


def test_wrapper_maps_observations_actions_and_rendering(make_wrapper):
    env = make_wrapper(render_fps=24)
    assert env.unwrapped.metadata["render_fps"] == 24
    observation, _ = env.reset(options={"lerobot_new_rollout": True})
    assert set(observation) == {"images.front", "state"}
    assert observation["images.front"].shape == (2, 3, 8, 8)
    assert observation["state"].shape == (2, 3)
    assert env.single_action_space.shape == (3,)
    rendered = env.call("render")
    assert len(rendered) == 2
    assert rendered[0].shape == (8, 8, 3)

    _, reward, terminated, truncated, info = env.step(
        np.zeros((2, 3), dtype=np.float32)
    )
    np.testing.assert_array_equal(reward, np.zeros(2))
    np.testing.assert_array_equal(terminated, np.zeros(2, dtype=np.bool_))
    np.testing.assert_array_equal(truncated, np.zeros(2, dtype=np.bool_))
    assert "is_success" not in info


def test_wrapper_maps_configured_success_key(make_wrapper):
    env = make_wrapper(success_key="goal_reached")
    info = _normalize_info(
        {
            "goal_reached": np.array(["true", "false"]),
            "_goal_reached": np.ones(2, dtype=np.bool_),
        },
        env.success_key,
    )
    np.testing.assert_array_equal(info["is_success"], [True, False])
    np.testing.assert_array_equal(info["_is_success"], [True, True])


def test_wrapper_uses_info_mask_when_mapping_success(make_wrapper):
    env = make_wrapper(success_key="goal_reached")
    info = _normalize_info(
        {
            "goal_reached": np.array(["true", None], dtype=object),
            "_goal_reached": np.array([True, False]),
        },
        env.success_key,
    )
    np.testing.assert_array_equal(info["is_success"], [True, False])
    np.testing.assert_array_equal(info["_is_success"], [True, False])


def test_wrapper_does_not_reinterpret_nested_final_info(make_wrapper):
    env = make_wrapper(success_key="goal_reached")
    nested_final_info = {"goal_reached": "false"}
    info = _normalize_info(
        {
            "final_info": {
                "goal_reached": "true",
                "final_info": nested_final_info,
            }
        },
        env.success_key,
    )

    assert info["final_info"]["is_success"] is True
    assert info["final_info"]["final_info"] is nested_final_info
    assert "is_success" not in nested_final_info


def test_wrapper_does_not_require_success_key(make_wrapper):
    env = make_wrapper()
    info = _normalize_info(
        {"episode_reason": np.array(["timeout", "goal"])}, env.success_key
    )
    assert "is_success" not in info


def test_wrapper_groups_multiple_mapped_cameras(make_wrapper):
    wrapper = make_wrapper(
        num_envs=1,
        observation_space=Dict(
            {
                "front": Box(0, 1, shape=(3, 8, 8), dtype=np.float32),
                "joints": Box(-1, 1, shape=(3,), dtype=np.float32),
                "wrist": Box(0, 1, shape=(3, 4, 4), dtype=np.float32),
            }
        ),
        policy_sources={
            "observation.images.front": "observation.front",
            "observation.images.wrist": "observation.wrist",
            "observation.state": "observation.joints",
        },
        render_camera="wrist",
    )
    observation, _ = wrapper.reset()
    assert observation["images.front"].shape == (1, 3, 8, 8)
    assert observation["images.wrist"].shape == (1, 3, 4, 4)
    assert observation["state"].shape == (1, 3)
    assert wrapper.call("render")[0].shape == (4, 4, 3)


def test_wrapper_exposes_lerobot_attributes(make_wrapper):
    env = make_wrapper()
    assert env.get_attr("task") == ("reach", "reach")
    assert env.call("task_description") == ("Reach the target.", "Reach the target.")
    assert env.call("_max_episode_steps") == (50, 50)


def test_wrapper_rejects_discrete_actions(make_wrapper):
    with pytest.raises(TypeError, match="continuous actions"):
        make_wrapper(
            num_envs=2,
            observation_space=Dict(
                {"observation": Box(-1, 1, shape=(4,), dtype=np.float32)}
            ),
            action_space=Discrete(2),
            policy_sources={"observation.state": "observation.observation"},
        )


def test_wrapper_converts_policy_feature_observations(make_wrapper):
    wrapper = make_wrapper(
        num_envs=2,
        observation_space=Dict(
            {
                "front_camera": Box(0, 1, shape=(3, 8, 8), dtype=np.float32),
                "gripper": Box(-3, 3, shape=(1,), dtype=np.float32),
                "joint_positions": Box(-1, 1, shape=(2,), dtype=np.float32),
                "joint_velocities": Box(-2, 2, shape=(2,), dtype=np.float32),
                "target": Box(-1, 1, shape=(3,), dtype=np.float32),
            }
        ),
        policy_sources={
            "observation.images.front": "observation.front_camera",
            "observation.state": [
                "observation.joint_positions",
                "observation.joint_velocities",
                "observation.gripper",
            ],
            "observation.environment_state": "observation.target",
        },
    )

    observation = {
        "front_camera": np.zeros((2, 3, 8, 8), dtype=np.float32),
        "gripper": np.array([[5.0], [6.0]], dtype=np.float32),
        "joint_positions": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "joint_velocities": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "target": np.ones((2, 3), dtype=np.float32),
    }
    converted = wrapper._convert_schola_observation(observation)

    assert set(converted) == {"images.front", "state", "environment_state"}
    np.testing.assert_array_equal(
        converted["state"],
        np.array(
            [
                [1.0, 2.0, 0.1, 0.2, 5.0],
                [3.0, 4.0, 0.3, 0.4, 6.0],
            ],
            dtype=np.float32,
        ),
    )
    state_space = wrapper.single_observation_space["state"]
    assert state_space.shape == (5,)
    np.testing.assert_array_equal(state_space.low, [-1, -1, -2, -2, -3])
    np.testing.assert_array_equal(state_space.high, [1, 1, 2, 2, 3])


def test_wrapper_one_hot_encodes_discrete_observation(make_wrapper):
    wrapper = make_wrapper(
        num_envs=2,
        observation_space=Dict({"mode": Discrete(4)}),
        policy_sources={"observation.state": "observation.mode"},
    )

    converted = wrapper._convert_schola_observation({"mode": np.array([0, 2])})

    assert wrapper.single_observation_space["state"] == Box(
        0, 1, shape=(4,), dtype=np.int64
    )
    np.testing.assert_array_equal(
        converted["state"],
        np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.int64),
    )


def test_wrapper_flattens_discrete_observation_before_concatenating(make_wrapper):
    wrapper = make_wrapper(
        num_envs=2,
        observation_space=Dict(
            {
                "joints": Box(-1, 1, shape=(2,), dtype=np.float32),
                "mode": Discrete(3),
            }
        ),
        policy_sources={
            "observation.state": [
                "observation.joints",
                "observation.mode",
            ]
        },
    )

    converted = wrapper._convert_schola_observation(
        {
            "joints": np.array([[0.25, 0.5], [-0.5, -0.25]], dtype=np.float32),
            "mode": np.array([0, 2]),
        }
    )

    assert wrapper.single_observation_space["state"].shape == (5,)
    np.testing.assert_array_equal(
        converted["state"],
        np.array(
            [[0.25, 0.5, 1, 0, 0], [-0.5, -0.25, 0, 0, 1]],
            dtype=np.float64,
        ),
    )


def test_wrapper_converts_native_schola_camera_to_lerobot_format(make_wrapper):
    camera_values = np.empty((3, 4, 5), dtype=np.float32)
    camera_values[0] = 0.0
    camera_values[1] = 0.5
    camera_values[2] = 1.0
    wrapper = make_wrapper(
        num_envs=1,
        observation_space=Dict(
            {
                "camera": Box(camera_values, camera_values, dtype=np.float32),
                "joints": Box(-1, 1, shape=(2,), dtype=np.float32),
            }
        ),
        policy_sources={
            "observation.images.front": "observation.camera",
            "observation.state": "observation.joints",
        },
    )

    converted = wrapper._convert_schola_observation(
        {
            "camera": camera_values[np.newaxis, ...],
            "joints": np.zeros((1, 2), dtype=np.float32),
        }
    )
    image = converted["images.front"]

    assert wrapper.single_observation_space["images.front"] == Box(
        camera_values, camera_values, dtype=np.float32
    )
    assert image.shape == (1, 3, 4, 5)
    assert image.dtype == np.float32
    np.testing.assert_array_equal(image[0], camera_values)


def test_wrapper_maps_singular_policy_image(make_wrapper):
    wrapper = make_wrapper(
        num_envs=1,
        observation_space=Box(0, 1, shape=(3, 8, 8), dtype=np.float32),
        policy_sources={"observation.image": "observation"},
        render_camera="image",
    )

    observation, _ = wrapper.reset()
    assert observation["image"].shape == (1, 3, 8, 8)
    assert wrapper.call("render")[0].shape == (8, 8, 3)


def test_wrapper_ignores_unmapped_observations(make_wrapper):
    wrapper = make_wrapper(
        num_envs=2,
        observation_space=Dict(
            {
                "joints": Box(-1, 1, shape=(2,), dtype=np.float32),
                "unused": Box(-1, 1, shape=(1,), dtype=np.float32),
            }
        ),
        policy_sources={"observation.state": "observation.joints"},
    )

    observation, _ = wrapper.reset()
    assert set(observation) == {"state"}


def test_wrapper_allows_reused_observation_source(make_wrapper):
    wrapper = make_wrapper(
        num_envs=1,
        observation_space=Dict(
            {"camera": Box(0, 1, shape=(3, 8, 8), dtype=np.float32)}
        ),
        policy_sources={
            "observation.images.front": "observation.camera",
            "observation.images.wrist": "observation.camera",
        },
    )

    observation, _ = wrapper.reset()
    np.testing.assert_array_equal(
        observation["images.front"], observation["images.wrist"]
    )


def test_wrapper_resolves_nested_schola_sources_with_dots(make_wrapper):
    wrapper = make_wrapper(
        num_envs=1,
        observation_space=Dict(
            {
                "robot": Dict({"joints": Box(-1, 1, shape=(3,), dtype=np.float32)}),
                "sensors": Dict({"top": Box(0, 1, shape=(3, 8, 8), dtype=np.float32)}),
            }
        ),
        policy_sources={
            "observation.images.top": "observation.sensors.top",
            "observation.state": "observation.robot.joints",
        },
    )

    observation, _ = wrapper.reset()
    assert observation["images.top"].shape == (1, 3, 8, 8)
    assert observation["state"].shape == (1, 3)


def test_wrapper_rejects_literal_dots_in_schola_keys(make_wrapper):
    with pytest.raises(ValueError, match="contains '\\.'"):
        make_wrapper(
            observation_space=Dict(
                {"robot.joints": Box(-1, 1, shape=(3,), dtype=np.float32)}
            ),
            policy_sources={"observation.state": "observation.robot.joints"},
        )


def test_wrapper_maps_top_level_dict_key(make_wrapper):
    wrapper = make_wrapper(
        num_envs=1,
        observation_space=Dict(
            {"proprioception": Box(-1, 1, shape=(3,), dtype=np.float32)}
        ),
        policy_sources={"observation.state": "observation.proprioception"},
    )

    observation, _ = wrapper.reset()
    assert observation["state"].shape == (1, 3)


def test_normalize_info_maps_masked_and_final_values():
    info = _normalize_info(
        {
            "goal_reached": np.array(["true", None], dtype=object),
            "_goal_reached": np.array([True, False]),
            "final_info": {"goal_reached": "false"},
        },
        "goal_reached",
    )

    np.testing.assert_array_equal(info["is_success"], [True, False])
    np.testing.assert_array_equal(info["_is_success"], [True, False])
    assert info["final_info"]["is_success"] is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("FALSE", False), (" True ", True)],
)
def test_parse_success_accepts_scalar_values(value, expected):
    assert _parse_success_string(value) is expected


def test_parse_success_accepts_array_values():
    np.testing.assert_array_equal(
        _parse_success_array(np.array(["true", "false"])), [True, False]
    )


@pytest.mark.parametrize("value", ["yes", "1"])
def test_parse_success_rejects_non_schola_strings(value):
    with pytest.raises(ValueError, match="true' or 'false"):
        _parse_success_string(value)


def test_parse_success_rejects_bool_values():
    with pytest.raises(TypeError, match="true' or 'false' string or array"):
        _normalize_info({"goal_reached": True}, "goal_reached")
