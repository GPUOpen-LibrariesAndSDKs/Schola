# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""Multi-agent RLlib tests backed by Schola ``RayEnv``."""

import pytest
import numpy as np
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from schola.core.simulators.unreal.editor_simulator import UnrealEditor
from schola.core.protocols.protobuf.grpc_protocol import GrpcProtocol
from schola.rllib.env import RayEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gymnasium as gym
from schola.rllib.export import export_onnx_from_policy
from Test.conftest import (
    check_onnx_metadata,
    check_onnx_model_loads,
    check_onnx_runtime,
)

# Check if PettingZoo is available
try:
    import pettingzoo

    PETTINGZOO_AVAILABLE = True
except ImportError:
    PETTINGZOO_AVAILABLE = False

# Skip all tests in this file if PettingZoo is not available
pytestmark = pytest.mark.skipif(
    not PETTINGZOO_AVAILABLE, reason="PettingZoo not installed"
)


def test_rayenv_pettingzoo_create(make_rllib_pettingzoo_env):
    """Test creating RayEnv with a PettingZoo environment."""
    env = make_rllib_pettingzoo_env("simple_spread_v3")

    assert env.num_envs == 1
    assert (
        env.single_observation_space is not None
    ), f"Single observation space should not be None"
    assert (
        env.single_action_space is not None
    ), f"Single action space should not be None"
    assert (
        env.single_observation_spaces is not None
    ), f"Single observation spaces should not be None"
    assert (
        env.single_action_spaces is not None
    ), f"Single action spaces should not be None"

    # PettingZoo pistonball_v6 should have 20 agents (piston_0 through piston_19)
    assert (
        len(env.possible_agents) == 3
    ), "PettingZoo simple_spread_v3 should have 3 agents"

    # Verify observation spaces match simple_spread_v3 spec
    # Observation Shape: (18,), Values: (-inf, inf)
    for agent_id in env.possible_agents:
        obs_space = env.single_observation_spaces[agent_id]
        assert obs_space == gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,)
        ), f"Agent {agent_id} observation space should be Box(low=-np.inf, high=np.inf, shape=(18,)), got {obs_space}"

    # Verify action spaces match simple_spread_v3 spec
    # Action Shape: (5,), Values: [0, 4]
    for agent_id in env.possible_agents:
        action_space = env.single_action_spaces[agent_id]
        assert isinstance(
            action_space, gym.spaces.Discrete
        ), f"Agent {agent_id} action space should be Discrete, got {type(action_space)}"
        assert action_space == gym.spaces.Discrete(
            5
        ), f"Agent {agent_id} action space should be Discrete(5), got {action_space}"

    env.close()


def test_rayenv_pettingzoo_reset(make_rllib_pettingzoo_env):
    """Test RayEnv reset with PettingZoo environment."""
    env = make_rllib_pettingzoo_env("simple_spread_v3")

    observations, infos = env.reset()

    # Should return dict format (not list)
    assert isinstance(observations, dict)
    assert isinstance(infos, dict)

    # Should have multiple agents
    assert len(observations) == 3, "simple_spread_v3 environment should have 3 agents"

    # All agents should be in possible_agents
    for agent_id in observations.keys():
        assert agent_id in env.possible_agents

    # Check observations are in observation space
    for agent_id, obs in observations.items():
        assert (
            obs in env.single_observation_spaces[agent_id]
        ), f"Observation {obs} should be in agent {agent_id} observation space {env.single_observation_spaces[agent_id]}"

    env.close()


def test_rayenv_pettingzoo_step(make_rllib_pettingzoo_env, make_pettingzoo_env):
    """Test RayEnv step with PettingZoo environment by comparing with local environment."""
    # Create remote environment (via Schola)
    remote_env = make_rllib_pettingzoo_env("pistonball_v6")

    # Create local PettingZoo environment for comparison
    local_env = make_pettingzoo_env("pistonball_v6")

    # Set seed for reproducibility
    seed = 42

    # Reset both environments with the same seed
    remote_obs, remote_infos = remote_env.reset(seed=seed)
    local_obs, local_infos = local_env.reset(seed=seed)

    # Verify observations match
    assert set(remote_obs.keys()) == set(
        local_obs.keys()
    ), "Agent keys should match between remote and local"
    for agent_id in remote_obs.keys():
        np.testing.assert_array_almost_equal(
            remote_obs[agent_id],
            local_obs[agent_id],
            err_msg=f"Initial observations for agent {agent_id} should match",
        )

    # Create the same actions for both environments
    actions = {}
    for agent_id in remote_obs.keys():
        # Use a seeded random generator for reproducible actions
        rng = np.random.RandomState(seed + hash(agent_id) % 2**31)
        action_space = remote_env.single_action_spaces[agent_id]
        # Sample action manually for reproducibility
        actions[agent_id] = rng.uniform(
            action_space.low, action_space.high, action_space.shape
        ).astype(action_space.dtype)

    # Step both environments with the same actions
    remote_obs, remote_rewards, remote_terminateds, remote_truncateds, remote_infos = (
        remote_env.step(actions)
    )
    local_obs, local_rewards, local_terminateds, local_truncateds, local_infos = (
        local_env.step(actions)
    )

    # Check return types for remote environment
    assert isinstance(remote_obs, dict)
    assert isinstance(remote_rewards, dict)
    assert isinstance(remote_terminateds, dict)
    assert isinstance(remote_truncateds, dict)
    assert isinstance(remote_infos, dict)

    # Check for __all__ key in remote environment (RLlib convention)
    assert "__all__" in remote_terminateds
    assert "__all__" in remote_truncateds

    # Compare observations
    assert set(remote_obs.keys()) == set(
        local_obs.keys()
    ), "Agent keys should match after step"
    for agent_id in remote_obs.keys():
        np.testing.assert_array_almost_equal(
            remote_obs[agent_id],
            local_obs[agent_id],
            err_msg=f"Observations for agent {agent_id} should match",
        )

    # Compare rewards
    for agent_id in remote_rewards.keys():
        if agent_id != "__all__":  # Skip RLlib's __all__ key
            assert (
                agent_id in local_rewards
            ), f"Agent {agent_id} should have reward in local env"
            assert remote_rewards[agent_id] == pytest.approx(
                local_rewards[agent_id], rel=1e-6
            ), f"Rewards for agent {agent_id} should match: remote={remote_rewards[agent_id]}, local={local_rewards[agent_id]}"

    # Verify __all__ flag is correctly set based on individual agent flags
    assert remote_terminateds["__all__"] == all(
        local_terminateds.values()
    ), "__all__ terminated flag should be True if all agents are terminated"
    assert remote_truncateds["__all__"] == all(
        local_truncateds.values()
    ), "__all__ truncated flag should be True if all agents are truncated"

    remote_env.close()
    local_env.close()


def test_rayenv_pettingzoo_multiple_agents(make_rllib_pettingzoo_env):
    """Test that PettingZoo multi-agent functionality works correctly."""
    env = make_rllib_pettingzoo_env("simple_spread_v3")

    observations, infos = env.reset()

    # pistonball_v6 has 20 agents by default
    assert len(observations) == 3, f"Expected 3 agents, got {len(observations)}"

    # Step through environment
    for _ in range(10):
        actions = {
            agent_id: env.single_action_spaces[agent_id].sample()
            for agent_id in observations.keys()
        }
        observations, rewards, terminateds, truncateds, infos = env.step(actions)

        # Verify multi-agent consistency
        assert len(rewards) == 3, "Should have rewards for all agents"

        # All active agents should have observations
        for agent_id in observations.keys():
            if agent_id != "__all__":
                assert agent_id in env.possible_agents

    env.close()


def test_rayenv_pettingzoo_inheritance(make_rllib_pettingzoo_env):
    """Test that RayEnv with PettingZoo inherits from correct base classes."""
    env = make_rllib_pettingzoo_env("simple_spread_v3")

    # Check inheritance
    assert isinstance(env, MultiAgentEnv), "Should inherit from MultiAgentEnv"

    # Check that required properties exist
    assert hasattr(env, "possible_agents")
    assert hasattr(env, "num_agents")
    assert hasattr(env, "observation_space")
    assert hasattr(env, "action_space")

    env.close()


from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.connectors.env_to_module import FlattenObservations

# import ray registry
from ray.tune.registry import register_env


@pytest.mark.xdist_group(name="ray-cluster")
def test_multiagent_training(
    make_pettingzoo_env_server,
    make_pettingzoo_env,
    ray_cluster,
    tmp_path,
):
    env_server_port = make_pettingzoo_env_server(
        make_pettingzoo_env("simple_spread_v3")
    )

    def make(*args, **kwargs):
        simulator = UnrealEditor()
        protocol = GrpcProtocol(url="localhost", port=env_server_port)
        return RayEnv(protocol, simulator)

    register_env("simple_spread_v3", make)

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .training(
            num_epochs=2,
            train_batch_size=128,
            minibatch_size=32,
        )
        .environment(env="simple_spread_v3")
        .framework("torch")
        .env_runners(
            num_env_runners=0,  # Use local runner only (no remote workers)
            env_to_module_connector=lambda env, spaces=None, device=None: FlattenObservations(
                input_observation_space=env.single_observation_space,
                input_action_space=env.single_action_space,
                multi_agent=True,
            ),
        )
        .multi_agent(
            policies={"policy_0": PolicySpec(), "policy_1": PolicySpec()},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: f"policy_{int(agent_id[6:]) % 2}",  # type: ignore
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "policy_0": RLModuleSpec(
                        model_config=DefaultModelConfig(
                            fcnet_hiddens=[32, 32], vf_share_layers=True
                        )
                    ),
                    "policy_1": RLModuleSpec(
                        model_config=DefaultModelConfig(
                            fcnet_hiddens=[32, 32], vf_share_layers=True
                        )
                    ),
                },
            ),
        )
    )
    algo = config.build_algo()
    # Just run a few iterations to verify training works
    result = algo.train()
    assert result is not None
    algo.stop()

    export_onnx_from_policy(algo, tmp_path)
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(18,))
    action_space = gym.spaces.Discrete(5)
    for policy_name in ("policy_0", "policy_1"):
        onnx_model = check_onnx_model_loads(tmp_path / f"{policy_name}.onnx")
        check_onnx_metadata(onnx_model, observation_space, action_space)
        check_onnx_runtime(onnx_model, observation_space, action_space)
