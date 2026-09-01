# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

from __future__ import annotations

import logging
from functools import partial
from textwrap import dedent, indent

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
    ScholaExecutableEnvConfig,
    ScholaExternalEnvConfig,
    ScholaProjectEnvConfig,
    infer_features_from_spaces,
)
from lerobot_env_schola.vector_env import LeRobotScholaVectorEnv, _policy_output
from schola.gym.env import GymVectorEnv
from schola.scripts.common.settings import (
    GrpcProtocolConfig,
    SingularExecutableSimulatorConfig,
    SingularExternalSimulatorConfig,
    SingularProjectSimulatorConfig,
)
from Test.gym.testing_env import GenericTestEnv


@pytest.fixture
def make_eval_config(tmp_path):
    def _make(env_yaml: str, args: list[str] | None = None) -> EvalPipelineConfig:
        config_path = tmp_path / "schola_eval.yaml"
        env_block = indent(dedent(env_yaml).strip(), "  ")
        config_path.write_text(
            f"""env:
{env_block}
eval:
  n_episodes: 1
  batch_size: 1
""",
            encoding="utf-8",
        )
        return draccus.parse(
            EvalPipelineConfig, config_path=config_path, args=args or []
        )

    return _make


def test_schola_config_is_registered():
    assert EnvConfig.get_choice_class("schola") is ScholaEnvConfig
    assert EnvConfig.get_choice_class("schola-external") is ScholaExternalEnvConfig
    assert EnvConfig.get_choice_class("schola-project") is ScholaProjectEnvConfig
    assert EnvConfig.get_choice_class("schola-executable") is ScholaExecutableEnvConfig


def test_schola_config_does_not_use_gym_make():
    cfg = ScholaEnvConfig()
    assert cfg.gym_kwargs == {}


def test_schola_external_alias_parses_from_yaml(make_eval_config):
    cfg = make_eval_config("""
        type: schola-external
        observations:
          state: observation.state
        """)

    assert isinstance(cfg.env, ScholaExternalEnvConfig)
    assert isinstance(cfg.env.simulator, SingularExternalSimulatorConfig)


@pytest.mark.parametrize("field", ["num_simulators", "num_environments"])
def test_schola_yaml_rejects_environment_counts(make_eval_config, field):
    with pytest.raises(draccus.utils.DecodingError, match=field):
        make_eval_config(f"""
            type: schola
            simulator:
              {field}: 5
            observations:
              state: observation.state
            """)


def test_schola_cli_rejects_use_async_envs(make_eval_config):
    with pytest.raises(SystemExit):
        make_eval_config(
            """
            type: schola
            observations:
              state: observation.state
            """,
            args=["--env.use_async_envs=true"],
        )


@pytest.mark.parametrize(
    ("env_type", "config_class", "simulator_class", "path_field", "file_name"),
    [
        (
            "schola-project",
            ScholaProjectEnvConfig,
            SingularProjectSimulatorConfig,
            "uproject_path",
            "RobotLab.uproject",
        ),
        (
            "schola-executable",
            ScholaExecutableEnvConfig,
            SingularExecutableSimulatorConfig,
            "executable_path",
            "RobotLab.exe",
        ),
    ],
)
def test_launched_simulator_configs_parse_from_yaml(
    tmp_path,
    make_eval_config,
    env_type,
    config_class,
    simulator_class,
    path_field,
    file_name,
):
    simulator_path = tmp_path / file_name
    simulator_path.touch()
    cfg = make_eval_config(f"""
        type: {env_type}
        simulator:
          {path_field}: {simulator_path.as_posix()}
        observations:
          state: observation.state
        """)

    assert isinstance(cfg.env, config_class)
    assert isinstance(cfg.env.simulator, simulator_class)
    assert getattr(cfg.env.simulator, path_field) == simulator_path


def test_policy_feature_observations_parse_from_yaml(make_eval_config):
    cfg = make_eval_config("""
        type: schola
        observations:
          images:
            front: observation.sensors.front_camera
          state:
            - observation.robot.joint_positions
            - observation.robot.joint_velocities
          environment_state: observation.target
        """)

    assert isinstance(cfg.env, ScholaEnvConfig)
    assert cfg.env.observations == {
        "images": {"front": "observation.sensors.front_camera"},
        "state": [
            "observation.robot.joint_positions",
            "observation.robot.joint_velocities",
        ],
        "environment_state": "observation.target",
    }
    assert cfg.env.observations.to_policy_mapping() == {
        "observation.images.front": "observation.sensors.front_camera",
        "observation.state": [
            "observation.robot.joint_positions",
            "observation.robot.joint_velocities",
        ],
        "observation.environment_state": "observation.target",
    }


def test_features_are_inferred_from_normalized_spaces(caplog):
    observation_space = gym.spaces.Dict(
        {
            "agent_pos": gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
            "pixels": gym.spaces.Dict(
                {
                    "front": gym.spaces.Box(0, 255, shape=(8, 8, 3), dtype=np.uint8),
                    "wrist": gym.spaces.Box(0, 255, shape=(4, 4, 3), dtype=np.uint8),
                }
            ),
        }
    )
    action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

    with caplog.at_level(logging.INFO, logger="lerobot_env_schola.config"):
        features, features_map = infer_features_from_spaces(
            observation_space,
            action_space,
        )

    assert features == {
        "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        "pixels/front": PolicyFeature(type=FeatureType.VISUAL, shape=(8, 8, 3)),
        "pixels/wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(4, 4, 3)),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }
    assert features_map == {
        "agent_pos": OBS_STATE,
        "pixels/front": f"{OBS_IMAGES}.front",
        "pixels/wrist": f"{OBS_IMAGES}.wrist",
        ACTION: ACTION,
    }
    assert "policy feature observation.state" in caplog.text
    assert "policy feature observation.images.front" in caplog.text
    assert "policy feature observation.images.wrist" in caplog.text
    assert "policy feature action" in caplog.text


def test_feature_inference_rejects_non_dict_observation_space():
    observation_space = gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

    with pytest.raises(TypeError, match="mapped observation space.*Gymnasium Dict"):
        infer_features_from_spaces(observation_space, action_space)


@pytest.mark.parametrize(
    ("policy_key", "expected"),
    [
        ("observation.images.front", ("camera", "front")),
        ("observation.image", ("single_image", "pixels")),
        ("observation.state", ("value", "agent_pos")),
        ("observation.environment_state", ("value", "environment_state")),
        ("observation.velocity", ("value", "velocity")),
    ],
)
def test_policy_output_rules(policy_key, expected):
    assert _policy_output(policy_key) == expected


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
        observations={"state": "observation.joints"},
        simulator=SingularExternalSimulatorConfig(),
        protocol=GrpcProtocolConfig(url="localhost", port=port),
    )

    env = cfg.create_envs(n_envs=2, use_async_envs=True)["schola"][0]
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


def test_create_envs_uses_schola_vector_size_on_mismatch(make_vec_env_server, caplog):
    observation_space = gym.spaces.Dict(
        {"observation": gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32)}
    )
    action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
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
        observations={"state": "observation.observation"},
        simulator=SingularExternalSimulatorConfig(),
        protocol=GrpcProtocolConfig(url="localhost", port=port),
    )

    with caplog.at_level(logging.WARNING, logger="lerobot_env_schola.config"):
        env = cfg.create_envs(n_envs=1)["schola"][0]
    try:
        assert env.num_envs == 2
        assert "using Schola's native vector size" in caplog.text
        observations, _ = env.reset()
        assert observations["agent_pos"].shape == (2, 4)
    finally:
        env.close()


def test_create_envs_requires_observation_configuration():
    with pytest.raises(ValueError, match="requires observations"):
        ScholaEnvConfig().create_envs(n_envs=1)
