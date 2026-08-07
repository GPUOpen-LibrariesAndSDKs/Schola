from __future__ import annotations

from functools import partial

import gymnasium as gym
import pytest

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot_env_schola.config import ScholaEnvConfig
from lerobot_env_schola.vector_env import LeRobotScholaVectorEnv
from schola.gym.env import GymVectorEnv
from schola.scripts.common.settings import ExternalSimulatorConfig, GrpcProtocolConfig
from Test.gym.testing_env import GenericTestEnv
from Test.gym.vec_utils import make_env


def test_schola_config_is_registered():
    assert EnvConfig.get_choice_class("schola") is ScholaEnvConfig


def test_schola_config_does_not_use_gym_make():
    assert ScholaEnvConfig().gym_kwargs == {}


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
        observation_map={"joints": "agent_pos"},
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
        observation_map={"observation": "agent_pos"},
        simulator=ExternalSimulatorConfig(),
        protocol=GrpcProtocolConfig(url="localhost", port=port),
    )

    with pytest.raises(ValueError, match="Schola exposed 2"):
        cfg.create_envs(n_envs=1)


def test_create_envs_rejects_lerobot_async_vectorization():
    with pytest.raises(ValueError, match="async environment wrapping"):
        ScholaEnvConfig().create_envs(n_envs=1, use_async_envs=True)


def test_create_envs_requires_observation_map():
    with pytest.raises(ValueError, match="requires observation_map"):
        ScholaEnvConfig().create_envs(n_envs=1)
