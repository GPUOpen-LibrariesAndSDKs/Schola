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
from lerobot.utils.constants import ACTION
from lerobot_env_schola.config import (
    ScholaEnvConfig,
    ScholaExecutableEnvConfig,
    ScholaExternalEnvConfig,
    ScholaProjectEnvConfig,
)
from lerobot_env_schola.vector_env import (
    LeRobotScholaVectorEnv,
    _build_source_spaces,
)
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


@pytest.fixture
def make_schola_spaces():
    """Build nested Schola Gym spaces and the matching policy source mapping."""

    def _make(
        *,
        state_shape=(3,),
        cameras=None,
        action_shape=(2,),
    ):
        if cameras is None:
            cameras = {"front": (3, 8, 8), "wrist": (3, 4, 4)}

        observation_spaces: dict[str, gym.Space] = {
            "joints": gym.spaces.Box(-1, 1, shape=state_shape, dtype=np.float32),
        }
        if cameras:
            observation_spaces["cameras"] = gym.spaces.Dict(
                {
                    name: gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)
                    for name, shape in cameras.items()
                }
            )
        observation_space = gym.spaces.Dict(observation_spaces)
        action_space = gym.spaces.Box(-1, 1, shape=action_shape, dtype=np.float32)
        policy_sources = {"observation.state": ("observation.joints",)}
        for name in cameras:
            policy_sources[f"observation.images.{name}"] = (
                f"observation.cameras.{name}",
            )
        return {
            "observation_space": observation_space,
            "action_space": action_space,
            "policy_sources": policy_sources,
            "source_spaces": _build_source_spaces(observation_space),
        }

    return _make


@pytest.fixture
def make_created_env(make_vec_env_server):
    """Start a fake Schola server and return the env from ``create_envs()``."""
    created = []

    def _make(
        *,
        observation_space: gym.Space,
        action_space: gym.Space,
        observations: dict[str, object],
        num_server_envs: int = 2,
        n_envs: int | None = None,
        use_async_envs: bool = False,
        **config_kwargs,
    ) -> tuple[ScholaEnvConfig, LeRobotScholaVectorEnv]:
        port = make_vec_env_server(
            [
                partial(
                    GenericTestEnv,
                    observation_space=observation_space,
                    action_space=action_space,
                )
                for _ in range(num_server_envs)
            ]
        )
        cfg = ScholaEnvConfig(
            observations=observations,
            simulator=SingularExternalSimulatorConfig(),
            protocol=GrpcProtocolConfig(url="localhost", port=port),
            **config_kwargs,
        )
        env = cfg.create_envs(
            n_envs=num_server_envs if n_envs is None else n_envs,
            use_async_envs=use_async_envs,
        )["schola"][0]
        created.append(env)
        return cfg, env

    yield _make

    for env in reversed(created):
        env.close()


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


def test_homogeneous_multi_agent_definition_becomes_vector_slots(
    mock_protocol_and_simulator,
):
    protocol, simulator = mock_protocol_and_simulator
    agent_ids = ["agent_0", "agent_1"]
    observation_space = gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
    protocol.get_definition.return_value = (
        [agent_ids],
        [{agent_id: "" for agent_id in agent_ids}],
        {0: {agent_id: observation_space for agent_id in agent_ids}},
        {0: {agent_id: action_space for agent_id in agent_ids}},
    )
    protocol.send_reset_msg.return_value = (
        [
            {
                "agent_0": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "agent_1": np.array([4.0, 5.0, 6.0], dtype=np.float32),
            }
        ],
        [{agent_id: {} for agent_id in agent_ids}],
    )

    schola_env = GymVectorEnv(simulator, protocol)
    env = LeRobotScholaVectorEnv(
        schola_env,
        task="multi_agent",
        task_description="Homogeneous multi-agent test.",
        max_episode_steps=10,
        policy_sources={"observation.state": ("observation",)},
    )
    try:
        observations, _ = env.reset()
        assert env.num_envs == 2
        np.testing.assert_array_equal(
            observations["state"],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        )
    finally:
        env.close()


@pytest.mark.parametrize(
    ("mismatch", "error_pattern"),
    [
        ("observation", "Observation Space Mismatch"),
        ("action", "Action Space Mismatch"),
    ],
)
def test_heterogeneous_multi_agent_definition_fails_and_cleans_up(
    mock_protocol_and_simulator,
    mismatch,
    error_pattern,
):
    protocol, simulator = mock_protocol_and_simulator
    observation_spaces = {
        "agent_0": gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
        "agent_1": gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
    }
    action_spaces = {
        "agent_0": gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
        "agent_1": gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
    }
    if mismatch == "observation":
        observation_spaces["agent_1"] = gym.spaces.Box(
            -2, 2, shape=(3,), dtype=np.float32
        )
    else:
        action_spaces["agent_1"] = gym.spaces.Box(-2, 2, shape=(2,), dtype=np.float32)
    protocol.get_definition.return_value = (
        [["agent_0", "agent_1"]],
        [{"agent_0": "", "agent_1": ""}],
        {0: observation_spaces},
        {0: action_spaces},
    )

    with pytest.raises(AssertionError, match=error_pattern):
        GymVectorEnv(simulator, protocol)

    assert protocol.close.call_count >= 1
    assert simulator.stop.call_count >= 1


def test_create_envs_builds_schola_vector_env(make_created_env, make_schola_spaces):
    spaces = make_schola_spaces(cameras={}, action_shape=(1,))
    observation_space = spaces["observation_space"]
    action_space = spaces["action_space"]
    cfg, env = make_created_env(
        observation_space=observation_space,
        action_space=action_space,
        observations={"state": "observation.joints"},
        use_async_envs=True,
        task="swing_up",
        task_description="Swing the pendulum upright.",
        episode_length=200,
        render_fps=24,
    )

    assert isinstance(env, LeRobotScholaVectorEnv)
    assert isinstance(env.env, GymVectorEnv)
    assert env.num_envs == 2
    assert env.single_observation_space["state"] == observation_space["joints"]
    assert env.single_action_space == action_space
    assert env.unwrapped.metadata["render_fps"] == 24
    assert env.call("task") == ("swing_up", "swing_up")
    assert env.call("task_description") == (
        "Swing the pendulum upright.",
        "Swing the pendulum upright.",
    )
    assert env.call("_max_episode_steps") == (200, 200)
    assert cfg.features == {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
    }
    assert cfg.features_map == {
        "observation.state": "observation.state",
        ACTION: ACTION,
    }

    observations, _ = env.reset(options={"lerobot_new_rollout": True})
    assert observations["state"].shape == (2, 3)

    next_observations, rewards, terminated, truncated, infos = env.step(
        env.action_space.sample()
    )
    assert next_observations["state"].shape == (2, 3)
    assert rewards.shape == terminated.shape == truncated.shape == (2,)
    assert isinstance(infos, dict)


def test_create_envs_uses_schola_vector_size_on_mismatch(make_created_env, caplog):
    observation_space = gym.spaces.Dict(
        {"observation": gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32)}
    )
    action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

    with caplog.at_level(logging.WARNING, logger="lerobot_env_schola.config"):
        _, env = make_created_env(
            observation_space=observation_space,
            action_space=action_space,
            observations={"state": "observation.observation"},
            n_envs=1,
        )

    assert env.num_envs == 2
    assert "using Schola's native vector size" in caplog.text
    observations, _ = env.reset()
    assert observations["state"].shape == (2, 4)


def test_create_envs_requires_observation_configuration():
    with pytest.raises(ValueError, match="requires observations"):
        ScholaEnvConfig().create_envs(n_envs=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "features": {
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,))
            }
        },
        {"features_map": {"observation.state": "observation.state"}},
        {
            "features": {
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,))
            },
            "features_map": {"observation.state": "observation.state"},
        },
    ],
)
def test_create_envs_rejects_configured_features(kwargs):
    cfg = ScholaEnvConfig(observations={"state": "observation"}, **kwargs)
    with pytest.raises(ValueError, match="infers features and features_map"):
        cfg.create_envs(n_envs=1)
