from __future__ import annotations

import logging
from functools import partial
from textwrap import dedent

import draccus
import gymnasium as gym
import numpy as np
import pytest

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.configs import EnvConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot_env_schola.config import (
    ScholaEnvConfig,
    ScholaObservationConfig,
    infer_features_from_spaces,
)
from lerobot_env_schola.vector_env import LeRobotScholaVectorEnv
from schola.gym.env import GymVectorEnv
from schola.scripts.common.settings import ExternalSimulatorConfig, GrpcProtocolConfig
from Test.gym.testing_env import GenericTestEnv
from Test.gym.vec_utils import make_env


def test_schola_config_is_registered():
    assert EnvConfig.get_choice_class("schola") is ScholaEnvConfig


def test_schola_config_does_not_use_gym_make():
    cfg = ScholaEnvConfig()
    assert cfg.gym_kwargs == {}


def test_target_oriented_observations_parse_from_yaml(tmp_path):
    config_path = tmp_path / "schola_eval.yaml"
    config_path.write_text(
        dedent(
            """
            env:
              type: schola
              observations:
                cameras:
                  front: front_camera
                vectors:
                  agent_pos:
                    - joint_positions
                    - joint_velocities
                passthrough:
                  environment_state: target
                ignore:
                  - debug
            eval:
              n_episodes: 1
              batch_size: 1
              use_async_envs: false
            """
        ),
        encoding="utf-8",
    )

    cfg = draccus.parse(EvalPipelineConfig, config_path=config_path, args=[])

    assert isinstance(cfg.env, ScholaEnvConfig)
    assert cfg.env.observations == ScholaObservationConfig(
        cameras={"front": "front_camera"},
        vectors={"agent_pos": ["joint_positions", "joint_velocities"]},
        passthrough={"environment_state": "target"},
        ignore=["debug"],
    )


def test_features_are_inferred_from_normalized_spaces(caplog):
    observation_space = gym.spaces.Dict(
        {
            "agent_pos": gym.spaces.Box(
                -1, 1, shape=(3,), dtype=np.float32
            ),
            "pixels": gym.spaces.Dict(
                {
                    "front": gym.spaces.Box(
                        0, 255, shape=(8, 8, 3), dtype=np.uint8
                    ),
                    "wrist": gym.spaces.Box(
                        0, 255, shape=(4, 4, 3), dtype=np.uint8
                    ),
                }
            ),
        }
    )
    action_space = gym.spaces.Box(
        -1, 1, shape=(2,), dtype=np.float32
    )

    with caplog.at_level(logging.INFO, logger="lerobot_env_schola.config"):
        features, features_map = infer_features_from_spaces(
            observation_space,
            action_space,
        )

    assert features == {
        "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        "pixels/front": PolicyFeature(
            type=FeatureType.VISUAL, shape=(8, 8, 3)
        ),
        "pixels/wrist": PolicyFeature(
            type=FeatureType.VISUAL, shape=(4, 4, 3)
        ),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }
    assert features_map == {
        "agent_pos": OBS_STATE,
        "pixels/front": f"{OBS_IMAGES}.front",
        "pixels/wrist": f"{OBS_IMAGES}.wrist",
        ACTION: ACTION,
    }
    assert "agent_pos -> observation.state" in caplog.text
    assert "pixels/front -> observation.images.front" in caplog.text
    assert "pixels/wrist -> observation.images.wrist" in caplog.text
    assert "action -> action" in caplog.text


def test_create_envs_builds_schola_vector_env(make_vec_env_server):
    observation_space = gym.spaces.Dict(
        {"joints": gym.spaces.Box(-1, 1, shape=(3,), dtype=float)}
    )
    action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=float)
    port = make_vec_env_server(
        [
            partial(
                GenericTestEnv,
                observation_space=observation_space,
                action_space=action_space,
            )
            for _ in range(2)
        ]
    )
    cfg = ScholaEnvConfig(
        task="swing_up",
        task_description="Swing the pendulum upright.",
        episode_length=200,
        render_fps=24,
        observations=ScholaObservationConfig(
            vectors={"agent_pos": ["joints"]}
        ),
        simulator=ExternalSimulatorConfig(),
        protocol=GrpcProtocolConfig(url="localhost", port=port),
    )

    env = cfg.create_envs(n_envs=2)["schola"][0]
    try:
        assert isinstance(env, LeRobotScholaVectorEnv)
        assert isinstance(env.env, GymVectorEnv)
        assert env.num_envs == 2
        assert env.single_observation_space["agent_pos"] == observation_space["joints"]
        assert env.single_action_space == action_space
        assert env.unwrapped.metadata["render_fps"] == 24
        assert env.call("task") == ("swing_up", "swing_up")
        assert env.call("task_description") == (
            "Swing the pendulum upright.",
            "Swing the pendulum upright.",
        )
        assert env.call("_max_episode_steps") == (200, 200)
        assert cfg.features == {
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
        }
        assert cfg.features_map == {"agent_pos": OBS_STATE, ACTION: ACTION}

        observations, _ = env.reset(options={"lerobot_new_rollout": True})
        assert observations["agent_pos"].shape == (2, 3)

        next_observations, rewards, terminated, truncated, infos = env.step(
            env.action_space.sample()
        )
        assert next_observations["agent_pos"].shape == (2, 3)
        assert rewards.shape == terminated.shape == truncated.shape == (2,)
        assert isinstance(infos, dict)
    finally:
        env.close()


def test_create_envs_rejects_mismatched_schola_vector_size(make_vec_env_server):
    port = make_vec_env_server([make_env("CartPole-v1", i) for i in range(2)])
    cfg = ScholaEnvConfig(
        observations=ScholaObservationConfig(
            vectors={"agent_pos": ["observation"]}
        ),
        simulator=ExternalSimulatorConfig(),
        protocol=GrpcProtocolConfig(url="localhost", port=port),
    )

    with pytest.raises(ValueError, match="Schola exposed 2"):
        cfg.create_envs(n_envs=1)


def test_create_envs_rejects_lerobot_async_vectorization():
    with pytest.raises(ValueError, match="async environment wrapping"):
        ScholaEnvConfig().create_envs(n_envs=1, use_async_envs=True)


def test_create_envs_requires_observation_configuration():
    with pytest.raises(ValueError, match="requires observations"):
        ScholaEnvConfig().create_envs(n_envs=1)
