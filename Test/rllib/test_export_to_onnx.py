# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for exporting RLlib policies to ONNX"""

import copy
from dataclasses import dataclass, field
import functools
from typing import Any, Dict, Literal, Optional, Tuple
import pytest
import gymnasium as gym
import numpy as np
import onnx
import os
from ray.rllib import Policy
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.env import PettingZooEnv
from schola.rllib.export import export_onnx_from_policy
from schola.scripts.rllib.rllib_to_onnx import export_onnx_app
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.sac import SAC, SACConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
from schola.core.model import StateMetadata
from pettingzoo.utils.env import ParallelEnv

# Define test spaces
box_obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
discrete_obs_space = gym.spaces.Discrete(2)
binary_obs_space = gym.spaces.MultiBinary(2)
multi_discrete_obs_space = gym.spaces.MultiDiscrete([2, 3])
dict_obs_space = gym.spaces.Dict(
    {
        "box": box_obs_space,
        "discrete": discrete_obs_space,
        "binary": binary_obs_space,
        "multi_discrete": multi_discrete_obs_space,
    }
)
# we use slightly altered action spaces, so that no-ops can't work by accident.
box_action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
discrete_action_space = gym.spaces.Discrete(3)
binary_action_space = gym.spaces.MultiBinary(3)
multi_discrete_action_space = gym.spaces.MultiDiscrete([3, 4])
dict_action_space = gym.spaces.Dict(
    {
        "o_box": box_action_space,
        "o_discrete": discrete_action_space,
        "o_multi_discrete": multi_discrete_action_space,
    }
)


import ray


def params_to_str(params):
    return f"{type(params[0][0]).__name__}-{type(params[0][1]).__name__}-{params[1]}"


# ideally this would be frozen, but we need to be able to modify the num_policies after initialization
@dataclass()
class ExportParams:
    algo_name: str
    observation_space: gym.Space
    action_space: gym.Space
    use_lstm: bool = False  # only works on the new API Stack
    new_stack: bool = True
    num_layers: int = (
        1  # This is fixed to 1 as of the latest Ray Release (see rllib.core.models.catalog._get_encoder_config)
    )
    bidirectional: bool = False
    share_layers: bool = False
    use_attention: bool = False  # only works on the old API Stack
    num_agents: int = 1
    num_policies: Optional[int] = None

    def __hash__(self):
        lstm_settings = (
            (True, self.num_layers, self.bidirectional) if self.use_lstm else (False,)
        )
        return hash(
            (
                self.algo_name,
                str(self.observation_space),
                str(self.action_space),
                self.use_lstm,
                self.new_stack,
                *lstm_settings,
                self.num_agents,
                self.num_policies,
            )
        )

    def __post_init__(self):
        if self.num_policies is None:
            self.num_policies = self.num_agents
        if self.num_policies > self.num_agents:
            raise ValueError(
                f"num_policies must be less than or equal to num_agents, but got {self.num_policies} > {self.num_agents}"
            )

    @property
    def max_seq_len(self) -> int:
        return 10

    @property
    def state_shapes(self) -> Dict[str, Tuple[int, ...]]:
        layer_dim = self.num_layers * (2 if self.bidirectional else 1)
        hidden_size = 64
        if self.use_lstm:
            # RLlib's new API stack exports actor state as (1, hidden) tensors.
            if self.new_stack:
                return {
                    "actor_h": (1, hidden_size),
                    "actor_c": (1, hidden_size),
                }
            return {
                "actor_h": (layer_dim, hidden_size),
                "actor_c": (layer_dim, hidden_size),
            }
        elif self.use_attention:
            # layer index is represented as the key in the state dict
            return {"0": (self.max_seq_len, 32)}
        else:
            return {}

    @property
    def metadata(self) -> Dict[str, str]:
        state_metadata = {}
        if self.use_lstm:
            state_metadata["state_in_actor_h"] = StateMetadata(
                has_seq_dim=False
            ).to_dict()
            state_metadata["state_in_actor_c"] = StateMetadata(
                has_seq_dim=False
            ).to_dict()
        elif self.use_attention:
            state_metadata["state_in_0"] = StateMetadata(
                has_seq_dim=True, seq_dim=1, max_seq_len=self.max_seq_len
            ).to_dict()

        return state_metadata

    def get_config(self, env_name: str):
        if self.algo_name == "ppo":
            base_config = PPOConfig()
            base_config.train_batch_size = 256
            base_config.sgd_minibatch_size = 64
            base_config.num_epochs = 10

        elif self.algo_name == "sac":
            base_config = SACConfig()
            if not self.new_stack:
                base_config = base_config.training(
                    replay_buffer_config={
                        "type": "MultiAgentPrioritizedReplayBuffer",
                        "capacity": 10000,
                    }
                )
        else:
            raise ValueError(f"Unknown algorithm: {self.algo_name}")

        base_config = base_config

        config = (
            base_config.framework("torch")
            .environment(env=env_name)
            .env_runners(
                num_env_runners=0,  # Use local runner only for testing
                env_to_module_connector=lambda env, spaces=None, device=None: FlattenObservations(
                    input_observation_space=env.single_observation_space,
                    input_action_space=env.single_action_space,
                    multi_agent=self.num_agents > 1,
                ),
            )
            .api_stack(
                enable_rl_module_and_learner=self.new_stack,  # toggled
                enable_env_runner_and_connector_v2=self.new_stack,
            )
            .learners(num_learners=0)
            .rl_module(
                model_config={
                    "use_lstm": self.use_lstm,  # Note this arg isn't supported on the old API Stack
                    "lstm_cell_size": 64,
                    "fcnet_hiddens": [64, 64],
                    "bidirectional": self.bidirectional,
                    "lstm_num_layers": self.num_layers,
                }
            )
            .debugging(
                log_level="DEBUG",
            )
        )
        if not self.new_stack:
            config = config.training(
                model={
                    "use_attention": self.use_attention,  # note this arg is not supported on the new API Stack
                    "attention_dim": 32,
                    "max_seq_len": 10,
                }
            )
        if self.num_agents > 1:
            config = config.multi_agent(
                policies={f"policy_{i}": PolicySpec() for i in range(self.num_policies)},  # type: ignore
                policy_mapping_fn=lambda agent_id, *args, **kwargs: f"policy_{int(agent_id[6:]) % self.num_policies}",  # type: ignore
            )
        else:
            config = config.multi_agent(
                policies={"default_policy": PolicySpec()},
                policy_mapping_fn=lambda agent_id, *args, **kwargs: "default_policy",
            )

        return config

    def __str__(self):
        params_str = f"{self.algo_name.upper()}({type(self.observation_space).__name__},{type(self.action_space).__name__}"
        if self.share_layers:
            params_str += ",share_layers=True"
        if self.use_lstm:
            params_str += ",use_lstm=True"
            if self.bidirectional:
                params_str += ",bidirectional=True"
            if self.num_layers > 1:
                params_str += f",num_layers={self.num_layers}"
        if self.use_attention:
            params_str += ",use_attention=True"
        if self.num_agents > 1:
            params_str += f",num_agents={self.num_agents}"
        if self.num_policies != self.num_agents:
            params_str += f",num_policies={self.num_policies}"
        params_str += ")"
        return params_str

    def get_env_class(self):
        obs_space = self.observation_space
        action_space = self.action_space
        # test env is a gym env with a dictionary observation space and dictionary action space
        if self.num_agents > 1:
            num_agents = 2

            class PettingZooTestEnv(ParallelEnv):
                def __init__(self):
                    self.observation_spaces = {
                        f"agent_{i}": copy.deepcopy(obs_space)
                        for i in range(num_agents)
                    }
                    self.action_spaces = {
                        f"agent_{i}": copy.deepcopy(action_space)
                        for i in range(num_agents)
                    }
                    self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
                    self.agents = self.possible_agents

                def reset(self, seed=None, options=None):
                    observations = {
                        f"agent_{i}": self.observation_spaces[f"agent_{i}"].sample()
                        for i in range(num_agents)
                    }
                    infos = {}
                    return observations, infos

                # define these to remove the warning about overriding the observation_space and action_space methods
                def observation_space(self, agent: str):
                    return self.observation_spaces[agent]

                def action_space(self, agent: str):
                    return self.action_spaces[agent]

                def step(self, actions):
                    observations = {
                        f"agent_{i}": self.observation_spaces[f"agent_{i}"].sample()
                        for i in range(num_agents)
                    }
                    rewards = {f"agent_{i}": 0 for i in range(num_agents)}
                    terminateds = {f"agent_{i}": False for i in range(num_agents)}
                    truncateds = {f"agent_{i}": False for i in range(num_agents)}
                    infos = {f"agent_{i}": {} for i in range(num_agents)}
                    return observations, rewards, terminateds, truncateds, infos

            # apply the PettingZooEnv wrapper to the test env
            class TestEnv(PettingZooEnv):
                def __init__(self):
                    super().__init__(PettingZooTestEnv())

        else:  # test with a gym environment

            class TestEnv(gym.Env):
                def __init__(self):
                    self.observation_space = copy.deepcopy(obs_space)
                    self.action_space = copy.deepcopy(action_space)
                    super().__init__()

                def reset(
                    self,
                    *,
                    seed: Optional[int] = None,
                    options: Optional[Dict[str, Any]] = None,
                ):
                    super().reset(seed=seed)
                    observation = self.observation_space.sample()
                    info = {}
                    return observation, info

                def step(self, action):
                    observation = self.observation_space.sample()
                    reward = 0
                    terminated = False
                    truncated = False
                    info = {}
                    return observation, reward, terminated, truncated, info

        return TestEnv


@pytest.fixture(scope="module")
def make_algo_config():
    cached_configs = {}

    def _make_algo_config(params: ExportParams):
        if params in cached_configs:
            return cached_configs[params]

        env_class = params.get_env_class()
        env_name = f"test_env_{id(env_class)}"
        register_env(env_name, lambda config: env_class())
        config = params.get_config(env_name)
        cached_configs[params] = config
        return config

    yield _make_algo_config
    cached_configs.clear()


@pytest.fixture(scope="module")
def make_algo(make_algo_config):
    cached_algos = {}

    def _make_algo(params):
        if params in cached_algos:
            return cached_algos[params]
        algo = make_algo_config(params).build_algo()
        cached_algos[params] = algo
        return algo

    yield _make_algo
    cached_algos.clear()


@pytest.mark.xdist_group(name="ray-cluster")
class TestExportFunctions:
    # Test exporting RLlib policies to ONNX
    @pytest.mark.parametrize(
        "alg_params",
        [
            # Check different observation spaces with box action space
            ExportParams("ppo", box_obs_space, box_action_space, new_stack=False),
            pytest.param(
                ExportParams(
                    "ppo",
                    box_obs_space,
                    box_action_space,
                    use_attention=True,
                    new_stack=False,
                ),
                marks=pytest.mark.xfail(
                    reason="Old-stack attention export is not yet compatible with torch.onnx dynamo",
                    strict=False,
                ),
            ),
            pytest.param(
                ExportParams(
                    "ppo",
                    dict_obs_space,
                    dict_action_space,
                    use_attention=True,
                    new_stack=False,
                ),
                marks=pytest.mark.xfail(
                    reason="Old-stack attention export is not yet compatible with torch.onnx dynamo",
                    strict=False,
                ),
            ),
            ExportParams("ppo", discrete_obs_space, box_action_space, new_stack=False),
            ExportParams("ppo", dict_obs_space, dict_action_space, new_stack=False),
            ExportParams(
                "ppo", multi_discrete_obs_space, box_action_space, new_stack=False
            ),
            ExportParams("sac", box_obs_space, box_action_space, new_stack=False),
        ],
        ids=str,
    )
    def test_export_rllib_policy_to_onnx(
        self, tmp_path, make_algo, alg_params, onnx_model_checker, ray_cluster
    ):
        """Test exporting RLlib policies to ONNX format."""

        # Get the policy
        policy: Policy = make_algo(alg_params).get_policy("default_policy")
        onnx_file = tmp_path / "test_policy" / "model.onnx"

        # Export to ONNX
        export_onnx_from_policy(
            policy, onnx_file, observation_space=alg_params.observation_space
        )

        # Verify the ONNX file was created
        onnx_model_checker(
            onnx_file,
            alg_params.observation_space,
            alg_params.action_space,
            alg_params.state_shapes,
            alg_params.metadata,
        )

    # test exporting just the RLModule to ONNX (direct creation without Ray cluster)
    @pytest.mark.parametrize(
        "alg_params",
        [
            # Check different action spaces with dict observation space
            ExportParams("ppo", box_obs_space, box_action_space),
            ExportParams("ppo", discrete_obs_space, box_action_space),
            ExportParams("ppo", box_obs_space, dict_action_space),
            # Skip MultiBinary - not supported by RLlib
            ExportParams("ppo", multi_discrete_obs_space, box_action_space),
            # Test ppo with LSTM enabled
            ExportParams("ppo", dict_obs_space, dict_action_space, use_lstm=True),
            ExportParams("ppo", box_obs_space, box_action_space, use_lstm=True),
            ExportParams(
                "ppo",
                box_obs_space,
                box_action_space,
                use_lstm=True,
                num_layers=2,
            ),
            ExportParams(
                "ppo",
                box_obs_space,
                box_action_space,
                use_lstm=True,
                bidirectional=True,
            ),
            ExportParams("ppo", dict_obs_space, dict_action_space),
            # SAC - only supports Box action spaces (no Dict wrapping)
            ExportParams("sac", box_obs_space, box_action_space),
        ],
        ids=str,
    )
    def test_export_rllib_rl_module_to_onnx(
        self, tmp_path, make_algo, alg_params, onnx_model_checker, ray_cluster
    ):
        """Test exporting RLlib RLModule to ONNX format (no Ray cluster spawning)."""
        algo = make_algo(alg_params)
        onnx_file = tmp_path / "model.onnx"
        # Export to ONNX - uses the Algorithm export path which calls get_module()
        export_onnx_from_policy(
            algo.get_module(), onnx_file, observation_space=alg_params.observation_space
        )
        # TODO handle exporting vision models
        # Verify the ONNX file was created
        onnx_model_checker(
            onnx_file,
            alg_params.observation_space,
            alg_params.action_space,
            alg_params.state_shapes,
            alg_params.metadata,
        )

    # test exporting using the entire algorithm object
    @pytest.mark.parametrize(
        "alg_params",
        [
            # Check different action spaces with dict observation space
            # Test with LSTM enabled
            ExportParams("ppo", box_obs_space, box_action_space, use_lstm=True),
            ExportParams("sac", dict_obs_space, box_action_space),
        ],
        ids=str,
    )
    def test_export_rllib_algorithm_to_onnx(
        self, tmp_path, make_algo, alg_params, onnx_model_checker, ray_cluster
    ):
        """Test exporting RLlib algorithms to ONNX format from checkpoint using tune.run()."""
        algo = make_algo(alg_params)
        onnx_file = tmp_path / "model.onnx"
        # Export the RLModule from the algorithm object
        export_onnx_from_policy(algo, onnx_file)

        # Verify the ONNX file was created
        onnx_model_checker(
            onnx_file,
            alg_params.observation_space,
            alg_params.action_space,
            alg_params.state_shapes,
            alg_params.metadata,
        )

    @pytest.mark.parametrize(
        "alg_params",
        [
            ExportParams("ppo", box_obs_space, box_action_space, num_agents=2),
            ExportParams("ppo", dict_obs_space, dict_action_space, num_agents=2),
            ExportParams(
                "ppo", box_obs_space, box_action_space, num_agents=2, num_policies=1
            ),
        ],
        ids=str,
    )
    def test_export_rllib_multiagent_to_onnx(
        self, tmp_path, make_algo, alg_params, onnx_model_checker, ray_cluster
    ):
        """Test exporting RLlib multiagent policies to ONNX format."""
        algo = make_algo(alg_params)
        export_onnx_from_policy(algo, tmp_path)

        for i in range(alg_params.num_policies):
            onnx_model_checker(
                tmp_path / f"policy_{i}.onnx",
                alg_params.observation_space,
                alg_params.action_space,
                alg_params.state_shapes,
                alg_params.metadata,
            )


@pytest.mark.xdist_group(name="ray-cluster")
class TestExportScript:
    """End-to-end tests for the rllib-to-onnx CLI script."""

    @pytest.mark.parametrize(
        "alg_params",
        [
            ExportParams("ppo", box_obs_space, box_action_space),
            ExportParams("ppo", dict_obs_space, dict_action_space, use_lstm=True),
            ExportParams("sac", box_obs_space, box_action_space),
        ],
        ids=str,
    )
    def test_rllib_to_onnx_script(
        self, tmp_path, make_algo, alg_params, onnx_model_checker, ray_cluster
    ):
        """Test the rllib-to-onnx script: save checkpoint, run script, verify ONNX output."""
        algo: Algorithm = make_algo(alg_params)
        checkpoint_path = tmp_path / "checkpoint"
        export_path = tmp_path / "model.onnx"
        algo.save(checkpoint_path)
        export_onnx_app(
            [str(checkpoint_path), str(export_path)],
            result_action="return_value",
        )
        onnx_model_checker(
            export_path,
            alg_params.observation_space,
            alg_params.action_space,
            state_shapes=alg_params.state_shapes,
            metadata=alg_params.metadata,
        )


class TestExportOnnxFromPolicyEdgeCases:
    """Tests for edge cases in export_onnx_from_policy"""

    def test_export_fails_with_unsupported_type(self, tmp_path):
        """Test that export raises TypeError for unsupported argument types"""
        path = tmp_path / "output.onnx"

        # Try with an integer (unsupported type)
        with pytest.raises(
            TypeError, match="Cannot export ONNX from Policy/Checkpoint"
        ):
            export_onnx_from_policy(123, path)

    def test_export_fails_with_list(self, tmp_path):
        """Test that export raises TypeError for list argument"""
        path = tmp_path / "output.onnx"

        with pytest.raises(
            TypeError, match="Cannot export ONNX from Policy/Checkpoint"
        ):
            export_onnx_from_policy([1, 2, 3], path)

    def test_export_fails_with_none(self, tmp_path):
        """Test that export raises TypeError for None argument"""
        path = tmp_path / "output.onnx"

        with pytest.raises(
            TypeError, match="Cannot export ONNX from Policy/Checkpoint"
        ):
            export_onnx_from_policy(None, path)
