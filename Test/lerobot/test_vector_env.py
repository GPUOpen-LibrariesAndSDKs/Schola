# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.spaces import Box, Dict, Discrete

from lerobot_env_schola.vector_env import LeRobotScholaVectorEnv, _coerce_success
from Test.gym.testing_env import GenericTestVectorEnv


@pytest.fixture
def make_wrapper():
    wrappers = []

    def _make(
        *,
        num_envs=2,
        observation_space=None,
        action_space=None,
        observation_config=None,
        success_key=None,
        render_camera=None,
        render_fps=30,
    ) -> LeRobotScholaVectorEnv:
        if observation_space is None:
            observation_space = Dict(
                {
                    "camera": Box(0, 255, shape=(8, 8, 3), dtype=np.uint8),
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
        if observation_config is None:
            observation_config = {
                "observation.images.front": "observation.camera",
                "observation.state": "observation.joints",
            }
        env = GenericTestVectorEnv(
            num_envs=num_envs,
            observation_space=observation_space,
            action_space=action_space,
        )
        try:
            wrapper = LeRobotScholaVectorEnv(
                env,
                task="reach",
                task_description="Reach the target.",
                max_episode_steps=50,
                observation_config=observation_config,
                success_key=success_key,
                render_camera=render_camera,
                render_fps=render_fps,
            )
        except Exception:
            env.close()
            raise
        wrappers.append(wrapper)
        return wrapper

    yield _make

    for wrapper in reversed(wrappers):
        wrapper.close()


def test_wrapper_maps_observations_actions_and_rendering(make_wrapper):
    env = make_wrapper(render_fps=24)
    assert env.unwrapped.metadata["render_fps"] == 24
    observation, _ = env.reset(options={"lerobot_new_rollout": True})
    assert set(observation) == {"pixels", "agent_pos"}
    assert observation["pixels"]["front"].shape == (2, 8, 8, 3)
    assert observation["agent_pos"].shape == (2, 3)
    assert env.single_action_space.shape == (3,)
    rendered = env.call("render")
    assert len(rendered) == 2
    np.testing.assert_array_equal(rendered[0], observation["pixels"]["front"][0])

    _, reward, terminated, truncated, info = env.step(
        np.zeros((2, 3), dtype=np.float32)
    )
    np.testing.assert_array_equal(reward, np.zeros(2))
    np.testing.assert_array_equal(terminated, np.zeros(2, dtype=np.bool_))
    np.testing.assert_array_equal(truncated, np.zeros(2, dtype=np.bool_))
    assert "is_success" not in info


def test_wrapper_maps_configured_success_key(make_wrapper):
    env = make_wrapper(success_key="goal_reached")
    info = env._normalize_info(
        {
            "goal_reached": np.array(["true", "false"]),
            "_goal_reached": np.ones(2, dtype=np.bool_),
        }
    )
    np.testing.assert_array_equal(info["is_success"], [True, False])
    np.testing.assert_array_equal(info["_is_success"], [True, True])


def test_wrapper_uses_info_mask_when_mapping_success(make_wrapper):
    env = make_wrapper(success_key="goal_reached")
    info = env._normalize_info(
        {
            "goal_reached": np.array(["true", None], dtype=object),
            "_goal_reached": np.array([True, False]),
        }
    )
    np.testing.assert_array_equal(info["is_success"], [True, False])
    np.testing.assert_array_equal(info["_is_success"], [True, False])


def test_wrapper_does_not_require_success_key(make_wrapper):
    env = make_wrapper()
    info = env._normalize_info({"episode_reason": np.array(["timeout", "goal"])})
    assert "is_success" not in info


def test_wrapper_groups_multiple_mapped_cameras(make_wrapper):
    wrapper = make_wrapper(
        num_envs=1,
        observation_space=Dict(
            {
                "front": Box(0, 255, shape=(8, 8, 3), dtype=np.uint8),
                "joints": Box(-1, 1, shape=(3,), dtype=np.float32),
                "wrist": Box(0, 255, shape=(4, 4, 3), dtype=np.uint8),
            }
        ),
        observation_config={
            "observation.images.front": "observation.front",
            "observation.images.wrist": "observation.wrist",
            "observation.state": "observation.joints",
        },
        render_camera="wrist",
    )
    observation, _ = wrapper.reset()
    assert set(observation["pixels"]) == {"front", "wrist"}
    assert observation["pixels"]["front"].shape == (1, 8, 8, 3)
    assert observation["pixels"]["wrist"].shape == (1, 4, 4, 3)
    assert observation["agent_pos"].shape == (1, 3)
    np.testing.assert_array_equal(
        wrapper.call("render")[0],
        observation["pixels"]["wrist"][0],
    )


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
            observation_config={"observation.state": "observation.observation"},
        )


def test_wrapper_converts_policy_feature_observations(make_wrapper):
    wrapper = make_wrapper(
        num_envs=2,
        observation_space=Dict(
            {
                "front_camera": Box(0, 255, shape=(8, 8, 3), dtype=np.uint8),
                "gripper": Box(-3, 3, shape=(1,), dtype=np.float32),
                "joint_positions": Box(-1, 1, shape=(2,), dtype=np.float32),
                "joint_velocities": Box(-2, 2, shape=(2,), dtype=np.float32),
                "target": Box(-1, 1, shape=(3,), dtype=np.float32),
            }
        ),
        observation_config={
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
        "front_camera": np.zeros((2, 8, 8, 3), dtype=np.uint8),
        "gripper": np.array([[5.0], [6.0]], dtype=np.float32),
        "joint_positions": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "joint_velocities": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "target": np.ones((2, 3), dtype=np.float32),
    }
    converted = wrapper._convert_observation(observation)

    assert set(converted) == {"pixels", "agent_pos", "environment_state"}
    assert set(converted["pixels"]) == {"front"}
    np.testing.assert_array_equal(
        converted["agent_pos"],
        np.array(
            [
                [1.0, 2.0, 0.1, 0.2, 5.0],
                [3.0, 4.0, 0.3, 0.4, 6.0],
            ],
            dtype=np.float32,
        ),
    )
    agent_pos_space = wrapper.single_observation_space["agent_pos"]
    assert agent_pos_space.shape == (5,)
    np.testing.assert_array_equal(agent_pos_space.low, [-1, -1, -2, -2, -3])
    np.testing.assert_array_equal(agent_pos_space.high, [1, 1, 2, 2, 3])


def test_wrapper_one_hot_encodes_discrete_observation(make_wrapper):
    wrapper = make_wrapper(
        num_envs=2,
        observation_space=Dict({"mode": Discrete(4)}),
        observation_config={"observation.state": "observation.mode"},
    )

    converted = wrapper._convert_observation({"mode": np.array([0, 2])})

    assert wrapper.single_observation_space["agent_pos"] == Box(
        0, 1, shape=(4,), dtype=np.int64
    )
    np.testing.assert_array_equal(
        converted["agent_pos"],
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
        observation_config={
            "observation.state": [
                "observation.joints",
                "observation.mode",
            ]
        },
    )

    converted = wrapper._convert_observation(
        {
            "joints": np.array([[0.25, 0.5], [-0.5, -0.25]], dtype=np.float32),
            "mode": np.array([0, 2]),
        }
    )

    assert wrapper.single_observation_space["agent_pos"].shape == (5,)
    np.testing.assert_array_equal(
        converted["agent_pos"],
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
        observation_config={
            "observation.images.front": "observation.camera",
            "observation.state": "observation.joints",
        },
    )

    converted = wrapper._convert_observation(
        {
            "camera": camera_values[np.newaxis, ...],
            "joints": np.zeros((1, 2), dtype=np.float32),
        }
    )
    pixels = converted["pixels"]["front"]

    assert wrapper.single_observation_space["pixels"]["front"] == Box(
        0, 255, shape=(4, 5, 3), dtype=np.uint8
    )
    assert pixels.shape == (1, 4, 5, 3)
    assert pixels.dtype == np.uint8
    np.testing.assert_array_equal(pixels[0, 0, 0], [0, 128, 255])


def test_wrapper_prefers_hwc_for_ambiguous_float_image_shape(make_wrapper):
    camera_values = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
    camera_values /= camera_values.max()
    wrapper = make_wrapper(
        num_envs=1,
        observation_space=Dict(
            {"camera": Box(0.0, 1.0, shape=(4, 5, 3), dtype=np.float32)}
        ),
        observation_config={"observation.images.front": "observation.camera"},
    )

    pixels = wrapper._convert_observation({"camera": camera_values[np.newaxis, ...]})[
        "pixels"
    ]["front"]

    assert wrapper.single_observation_space["pixels"]["front"].shape == (4, 5, 3)
    assert pixels.shape == (1, 4, 5, 3)
    np.testing.assert_array_equal(
        pixels[0],
        np.rint(camera_values * 255).astype(np.uint8),
    )


def test_wrapper_maps_singular_policy_image(make_wrapper):
    wrapper = make_wrapper(
        num_envs=1,
        observation_space=Box(0, 255, shape=(8, 8, 3), dtype=np.uint8),
        observation_config={"observation.image": "observation"},
        render_camera="image",
    )

    observation, _ = wrapper.reset()
    assert observation["pixels"].shape == (1, 8, 8, 3)
    np.testing.assert_array_equal(wrapper.call("render")[0], observation["pixels"][0])


def test_wrapper_warns_about_unaccounted_observations(make_wrapper, caplog):
    with caplog.at_level("WARNING", logger="lerobot_env_schola.vector_env"):
        wrapper = make_wrapper(
            num_envs=2,
            observation_space=Dict(
                {
                    "joints": Box(-1, 1, shape=(2,), dtype=np.float32),
                    "unused": Box(-1, 1, shape=(1,), dtype=np.float32),
                }
            ),
            observation_config={"observation.state": "observation.joints"},
        )

    observation, _ = wrapper.reset()
    assert set(observation) == {"agent_pos"}
    assert "observation.unused" in caplog.text
    assert "will be ignored" in caplog.text


def test_wrapper_warns_about_reused_observation_source(make_wrapper, caplog):
    with caplog.at_level("WARNING", logger="lerobot_env_schola.vector_env"):
        wrapper = make_wrapper(
            num_envs=1,
            observation_space=Dict(
                {"camera": Box(0, 255, shape=(8, 8, 3), dtype=np.uint8)}
            ),
            observation_config={
                "observation.images.front": "observation.camera",
                "observation.images.wrist": "observation.camera",
            },
        )

    observation, _ = wrapper.reset()
    np.testing.assert_array_equal(
        observation["pixels"]["front"], observation["pixels"]["wrist"]
    )
    assert "observation.camera" in caplog.text
    assert "device-memory costs" in caplog.text


def test_wrapper_rejects_unsupported_camera_dtype(make_wrapper):
    with pytest.raises(TypeError, match="float or uint8"):
        make_wrapper(
            num_envs=2,
            observation_space=Dict(
                {"camera": Box(0, 255, shape=(8, 8, 3), dtype=np.int32)}
            ),
            observation_config={"observation.images.front": "observation.camera"},
        )


def test_wrapper_resolves_nested_schola_sources_with_dots(make_wrapper):
    wrapper = make_wrapper(
        num_envs=1,
        observation_space=Dict(
            {
                "robot": Dict({"joints": Box(-1, 1, shape=(3,), dtype=np.float32)}),
                "sensors": Dict({"top": Box(0, 255, shape=(8, 8, 3), dtype=np.uint8)}),
            }
        ),
        observation_config={
            "observation.images.top": "observation.sensors.top",
            "observation.state": "observation.robot.joints",
        },
    )

    observation, _ = wrapper.reset()
    assert observation["pixels"]["top"].shape == (1, 8, 8, 3)
    assert observation["agent_pos"].shape == (1, 3)


def test_wrapper_maps_non_composite_observation_root(make_wrapper):
    wrapper = make_wrapper(
        num_envs=2,
        observation_space=Box(-1, 1, shape=(4,), dtype=np.float32),
        observation_config={"observation.state": "observation"},
    )

    observation, _ = wrapper.reset()
    assert observation["agent_pos"].shape == (2, 4)
    assert wrapper.single_observation_space["agent_pos"] == Box(
        -1, 1, shape=(4,), dtype=np.float32
    )


def test_wrapper_rejects_literal_dots_in_schola_keys(make_wrapper):
    with pytest.raises(ValueError, match="contains '\\.'"):
        make_wrapper(
            observation_space=Dict(
                {"robot.joints": Box(-1, 1, shape=(3,), dtype=np.float32)}
            ),
            observation_config={"observation.state": "observation.robot.joints"},
        )


def test_wrapper_maps_top_level_dict_key(make_wrapper):
    wrapper = make_wrapper(
        num_envs=1,
        observation_space=Dict(
            {"proprioception": Box(-1, 1, shape=(3,), dtype=np.float32)}
        ),
        observation_config={"observation.state": "observation.proprioception"},
    )

    observation, _ = wrapper.reset()
    assert observation["agent_pos"].shape == (1, 3)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("FALSE", False), (True, True), (np.bool_(False), False)],
)
def test_coerce_success_accepts_scalar_values(value, expected):
    assert _coerce_success(value) is expected


def test_coerce_success_accepts_array_values():
    np.testing.assert_array_equal(
        _coerce_success(np.array(["true", "false"])), [True, False]
    )


@pytest.mark.parametrize("value", ["yes", "1"])
def test_coerce_success_rejects_non_schola_strings(value):
    with pytest.raises(ValueError, match="true' or 'false"):
        _coerce_success(value)
