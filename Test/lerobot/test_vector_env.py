from __future__ import annotations

import logging

import numpy as np
import pytest
from gymnasium.spaces import Box, Dict, Discrete

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_IMAGES, OBS_STATE
from lerobot_env_schola.config import infer_features_from_spaces
from lerobot_env_schola.vector_env import LeRobotScholaVectorEnv
from Test.gym.testing_env import GenericTestVectorEnv


@pytest.fixture
def make_wrapper():
    wrappers = []

    def _make(
        *,
        num_envs=2,
        observation_space=None,
        action_space=None,
        observation_map=None,
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
        if observation_map is None:
            observation_map = {
                "camera": "pixels",
                "joints": "agent_pos",
            }
        env = GenericTestVectorEnv(
            num_envs=num_envs,
            observation_space=observation_space,
            action_space=action_space,
        )
        wrapper = LeRobotScholaVectorEnv(
            env,
            task="reach",
            task_description="Reach the target.",
            max_episode_steps=50,
            success_key="goal_reached",
            observation_map=observation_map,
            render_camera=render_camera,
            render_fps=render_fps,
        )
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
    assert observation["pixels"].shape == (2, 8, 8, 3)
    assert observation["agent_pos"].shape == (2, 3)
    assert env.single_action_space.shape == (3,)
    rendered = env.call("render")
    assert len(rendered) == 2
    np.testing.assert_array_equal(rendered[0], observation["pixels"][0])

    _, reward, terminated, truncated, info = env.step(
        np.zeros((2, 3), dtype=np.float32)
    )
    np.testing.assert_array_equal(reward, np.zeros(2))
    np.testing.assert_array_equal(terminated, np.zeros(2, dtype=np.bool_))
    np.testing.assert_array_equal(truncated, np.zeros(2, dtype=np.bool_))
    np.testing.assert_array_equal(info["is_success"], np.zeros(2, dtype=np.bool_))


def test_wrapper_requires_a_complete_observation_map():
    env = GenericTestVectorEnv(
        num_envs=2,
        observation_space=Dict(
            {
                "camera": Box(0, 255, shape=(8, 8, 3), dtype=np.uint8),
                "joints": Box(-1, 1, shape=(3,), dtype=np.float32),
            }
        ),
        action_space=Box(-1, 1, shape=(1,), dtype=np.float32),
    )
    with pytest.raises(TypeError, match="observation_map"):
        LeRobotScholaVectorEnv(
            env,
            task="reach",
            task_description="Reach the target.",
            max_episode_steps=50,
        )
    with pytest.raises(ValueError, match="missing Schola keys"):
        LeRobotScholaVectorEnv(
            env,
            task="reach",
            task_description="Reach the target.",
            max_episode_steps=50,
            observation_map={"camera": "pixels"},
        )
    env.close()


def test_features_are_inferred_from_normalized_spaces(caplog, make_wrapper):
    env = make_wrapper()
    with caplog.at_level(logging.INFO, logger="lerobot_env_schola.config"):
        features, features_map = infer_features_from_spaces(
            env.single_observation_space,
            env.single_action_space,
        )
    assert features == {
        "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        "pixels": PolicyFeature(type=FeatureType.VISUAL, shape=(8, 8, 3)),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
    }
    assert features_map == {
        "agent_pos": OBS_STATE,
        "pixels": OBS_IMAGE,
        ACTION: ACTION,
    }
    assert "agent_pos -> observation.state" in caplog.text
    assert "pixels -> observation.image" in caplog.text
    assert "action -> action" in caplog.text


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
        observation_map={
            "front": "pixels/front",
            "joints": "agent_pos",
            "wrist": "pixels/wrist",
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
        action_space=Box(-1, 1, shape=(1,), dtype=np.float32),
        observation_map={"camera": "pixels", "joints": "agent_pos"},
    )
    observation, _ = wrapper.reset()
    pixels = observation["pixels"]
    assert wrapper.single_observation_space["pixels"] == Box(
        0, 255, shape=(4, 5, 3), dtype=np.uint8
    )
    assert pixels.shape == (1, 4, 5, 3)
    assert pixels.dtype == np.uint8
    np.testing.assert_array_equal(pixels[0, 0, 0], [0, 128, 255])


def test_multiple_camera_features_keep_camera_names():
    observation_space = Dict(
        {
            "pixels": Dict(
                {
                    "front": Box(0, 255, shape=(8, 8, 3), dtype=np.uint8),
                    "wrist": Box(0, 255, shape=(4, 4, 3), dtype=np.uint8),
                }
            )
        }
    )
    features, features_map = infer_features_from_spaces(
        observation_space,
        Box(-1, 1, shape=(2,), dtype=np.float32),
    )

    assert features["pixels/front"] == PolicyFeature(
        type=FeatureType.VISUAL, shape=(8, 8, 3)
    )
    assert features["pixels/wrist"] == PolicyFeature(
        type=FeatureType.VISUAL, shape=(4, 4, 3)
    )
    assert features_map["pixels/front"] == f"{OBS_IMAGES}.front"
    assert features_map["pixels/wrist"] == f"{OBS_IMAGES}.wrist"


def test_wrapper_exposes_lerobot_attributes(make_wrapper):
    env = make_wrapper()
    assert env.get_attr("task") == ("reach", "reach")
    assert env.call("task_description") == ("Reach the target.", "Reach the target.")
    assert env.call("_max_episode_steps") == (50, 50)


def test_wrapper_rejects_discrete_actions():
    env = GenericTestVectorEnv(
        num_envs=2,
        observation_space=Dict(
            {"observation": Box(-1, 1, shape=(4,), dtype=np.float32)}
        ),
        action_space=Discrete(2),
    )
    with pytest.raises(TypeError, match="continuous actions"):
        LeRobotScholaVectorEnv(
            env,
            task="balance",
            task_description="Balance the pole.",
            max_episode_steps=500,
            observation_map={"observation": "agent_pos"},
        )
    env.close()
