# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.spaces import Box, Dict, Discrete

from lerobot_env_schola.config import ScholaObservationConfig
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
            observation_config = ScholaObservationConfig(
                cameras={"front": "camera"},
                vectors={"agent_pos": ["joints"]},
            )
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
        observation_config=ScholaObservationConfig(
            cameras={"front": "front", "wrist": "wrist"},
            vectors={"agent_pos": ["joints"]},
        ),
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
            observation_config=ScholaObservationConfig(
                vectors={"agent_pos": ["observation"]}
            ),
        )


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
