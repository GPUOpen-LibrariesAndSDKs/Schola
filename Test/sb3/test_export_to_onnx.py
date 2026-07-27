# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for exporting SB3 policies to ONNX"""

import copy
import functools
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Tuple
import pytest
import gymnasium as gym
import numpy as np
from schola.sb3.export import save_model_as_onnx
import stable_baselines3 as sb3
from schola.scripts.sb3.sb3_to_onnx import export_onnx_app
from schola.sb3.utils import VecMergeDictActionWrapper
import onnx

# Test Exporting

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_util import make_vec_env

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
box_dict_action_space = gym.spaces.Dict(
    {
        "o_box": box_action_space,
        "o_box_2": box_action_space,
    }
)
binary_dict_action_space = gym.spaces.Dict(
    {
        "o_binary": binary_action_space,
        "o_binary_2": binary_action_space,
    }
)
discrete_dict_action_space = gym.spaces.Dict(
    {
        "o_discrete": discrete_action_space,
        "o_discrete_2": discrete_action_space,
    }
)
multi_discrete_dict_action_space = gym.spaces.Dict(
    {
        "o_multi_discrete": multi_discrete_action_space,
        "o_multi_discrete_2": multi_discrete_action_space,
    }
)


@dataclass(frozen=True)
class ONNXExportParams:
    """Dataclass for handling different SB3 export configurations."""

    algo_name: str
    observation_space: gym.Space
    action_space: gym.Space
    buffer_size: int = 10000

    def __hash__(self):
        return hash(
            (
                self.algo_name,
                str(self.observation_space),
                str(self.action_space),
            )
        )

    @property
    def state_shapes(self) -> Optional[Dict[str, Tuple[int, ...]]]:
        """SB3 models are not stateful, so state_shapes is None."""
        return None

    @property
    def metadata(self) -> None:
        """SB3 models are not stateful, so metadata is None."""
        return None

    @property
    def policy_type(self) -> str:
        """Returns the policy type string based on observation space."""
        if isinstance(self.observation_space, gym.spaces.Dict):
            return "MultiInputPolicy"
        else:
            return "MlpPolicy"

    def __str__(self):
        # If the action space is a dict add the name of the action space inside it (all will be the same so just one is enough)
        if isinstance(self.action_space, gym.spaces.Dict):
            action_space_subtype = type(
                list(self.action_space.spaces.values())[0]
            ).__name__
            return f"{self.algo_name.upper()}({type(self.observation_space).__name__},{type(self.action_space).__name__}-{action_space_subtype})"
        return f"{self.algo_name.upper()}({type(self.observation_space).__name__},{type(self.action_space).__name__})"

    def get_env_class(self):
        obs_space = copy.deepcopy(self.observation_space)
        action_space = copy.deepcopy(self.action_space)

        class TestEnv(gym.Env):
            def __init__(self):
                self.observation_space = obs_space
                self.action_space = action_space

                super().__init__()

            def reset(self, seed=None, options=None):
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

    def make_algo(self):
        if self.algo_name == "ppo":
            return sb3.PPO
        elif self.algo_name == "sac":
            return lambda *args, **kwargs: sb3.SAC(
                *args, **kwargs, buffer_size=self.buffer_size
            )
        elif self.algo_name == "td3":
            return lambda *args, **kwargs: sb3.TD3(
                *args, **kwargs, buffer_size=self.buffer_size
            )
        elif self.algo_name == "ddpg":
            return lambda *args, **kwargs: sb3.DDPG(
                *args, **kwargs, buffer_size=self.buffer_size
            )
        elif self.algo_name == "a2c":
            return lambda *args, **kwargs: sb3.A2C(*args, **kwargs)
        elif self.algo_name == "dqn":
            return lambda *args, **kwargs: sb3.DQN(
                *args, **kwargs, buffer_size=self.buffer_size
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algo_name}")


def make_all_combinations(alg_name):
    """Helper function to generate all combinations for an algorithm."""
    return [
        # Check different action spaces
        ONNXExportParams(alg_name, dict_obs_space, box_action_space),
        ONNXExportParams(alg_name, dict_obs_space, discrete_action_space),
        ONNXExportParams(alg_name, dict_obs_space, binary_action_space),
        ONNXExportParams(alg_name, dict_obs_space, multi_discrete_action_space),
        # Check different observation spaces
        ONNXExportParams(alg_name, box_obs_space, box_action_space),
        ONNXExportParams(alg_name, discrete_obs_space, box_action_space),
        ONNXExportParams(alg_name, binary_obs_space, box_action_space),
        ONNXExportParams(alg_name, multi_discrete_obs_space, box_action_space),
    ]


class TestExportFunction:
    # Test exporting SB3 policies to ONNX, tests with every algorithm from SB3
    @pytest.mark.parametrize(
        "export_params",
        [
            # Testing for merged dict action spaces
            ONNXExportParams("ppo", dict_obs_space, box_dict_action_space),
            ONNXExportParams("ppo", dict_obs_space, discrete_dict_action_space),
            ONNXExportParams("ppo", dict_obs_space, binary_dict_action_space),
            ONNXExportParams("ppo", dict_obs_space, multi_discrete_dict_action_space),
            # PPO
            *make_all_combinations("ppo"),
            # SAC
            # try all obs spaces but only Box action spaces are supported
            ONNXExportParams("sac", box_obs_space, box_action_space),
            ONNXExportParams("sac", discrete_obs_space, box_action_space),
            ONNXExportParams("sac", binary_obs_space, box_action_space),
            ONNXExportParams("sac", multi_discrete_obs_space, box_action_space),
            ONNXExportParams("sac", dict_obs_space, box_action_space),
            # TD3
            ONNXExportParams("td3", box_obs_space, box_action_space),
            ONNXExportParams("td3", discrete_obs_space, box_action_space),
            ONNXExportParams("td3", binary_obs_space, box_action_space),
            ONNXExportParams("td3", multi_discrete_obs_space, box_action_space),
            ONNXExportParams("td3", dict_obs_space, box_action_space),
            # DDPG
            ONNXExportParams("ddpg", box_obs_space, box_action_space),
            ONNXExportParams("ddpg", discrete_obs_space, box_action_space),
            ONNXExportParams("ddpg", binary_obs_space, box_action_space),
            ONNXExportParams("ddpg", multi_discrete_obs_space, box_action_space),
            ONNXExportParams("ddpg", dict_obs_space, box_action_space),
            # A2C
            *make_all_combinations("a2c"),
            # DQN
            ONNXExportParams("dqn", box_obs_space, discrete_action_space),
            ONNXExportParams("dqn", binary_obs_space, discrete_action_space),
            ONNXExportParams("dqn", dict_obs_space, discrete_action_space),
            ONNXExportParams("dqn", discrete_obs_space, discrete_action_space),
            ONNXExportParams("dqn", multi_discrete_obs_space, discrete_action_space),
        ],
        ids=str,
    )
    def test_export_sb3_policy_to_onnx(
        self, tmp_path, export_params, onnx_model_checker
    ):
        params: ONNXExportParams = export_params

        env = make_vec_env(
            params.get_env_class(), 2, seed=1
        )  # prevents an error with automatic seeding
        # Create a dummy environment and model
        if isinstance(params.action_space, gym.spaces.Dict):
            env = VecMergeDictActionWrapper(env)

        model: BaseAlgorithm = params.make_algo()(params.policy_type, env, verbose=1)

        save_model_as_onnx(
            model, tmp_path / "test.onnx", action_space=params.action_space
        )

        onnx_model_checker(
            tmp_path / "test.onnx",
            params.observation_space,
            params.action_space,
            state_shapes=params.state_shapes,
            metadata=params.metadata,
        )


class TestExportScript:

    @pytest.mark.parametrize(
        "export_params",
        [
            # Testing for merged dict action spaces
            ONNXExportParams("ppo", dict_obs_space, box_action_space),
            ONNXExportParams("sac", dict_obs_space, box_action_space),
            ONNXExportParams("td3", dict_obs_space, box_action_space),
            ONNXExportParams("ddpg", dict_obs_space, box_action_space),
            ONNXExportParams("a2c", dict_obs_space, box_action_space),
            ONNXExportParams("dqn", dict_obs_space, discrete_action_space),
        ],
        ids=str,
    )
    def test_algorithms(self, tmp_path, export_params, onnx_model_checker):
        params: ONNXExportParams = export_params
        env = make_vec_env(params.get_env_class(), 2, seed=1)
        model: BaseAlgorithm = params.make_algo()(params.policy_type, env, verbose=1)
        model.save(tmp_path / "test.zip")
        export_onnx_app(
            [
                f"{tmp_path}/test.zip",
                f"{tmp_path}/test.onnx",
                "--algorithm",
                params.algo_name.upper(),
            ],
            result_action="return_value",
            exit_on_error=False,
        )

        onnx_model_checker(
            tmp_path / "test.onnx",
            params.observation_space,
            params.action_space,
            state_shapes=params.state_shapes,
            metadata=params.metadata,
        )

    @pytest.mark.parametrize(
        "export_params",
        [
            ONNXExportParams("ppo", dict_obs_space, box_dict_action_space),
            ONNXExportParams("ppo", dict_obs_space, discrete_dict_action_space),
            ONNXExportParams("ppo", dict_obs_space, binary_dict_action_space),
            ONNXExportParams("ppo", dict_obs_space, multi_discrete_dict_action_space),
        ],
        ids=str,
    )
    def test_merged_dict_action_space(
        self, tmp_path, export_params, onnx_model_checker
    ):
        params: ONNXExportParams = export_params

        env = make_vec_env(params.get_env_class(), 2, seed=1)
        env = VecMergeDictActionWrapper(env)

        model: BaseAlgorithm = params.make_algo()(params.policy_type, env, verbose=1)
        setattr(model, "__unmerged_action_space", env.unmerged_action_space)
        model.save(tmp_path / "test.zip")

        export_onnx_app(
            [
                f"{tmp_path}/test.zip",
                f"{tmp_path}/test.onnx",
                "--algorithm",
                params.algo_name.upper(),
            ],
            result_action="return_value",
            exit_on_error=False,
        )
        onnx_model_checker(
            tmp_path / "test.onnx",
            params.observation_space,
            params.action_space,
            state_shapes=params.state_shapes,
            metadata=params.metadata,
        )

    @pytest.mark.parametrize(
        "export_params",
        [
            ONNXExportParams("ppo", dict_obs_space, box_dict_action_space),
        ],
        ids=str,
    )
    def test_merge_with_no_hidden_state(
        self, tmp_path, export_params, onnx_model_checker
    ):
        params: ONNXExportParams = export_params
        env = make_vec_env(params.get_env_class(), 2, seed=1)
        env = VecMergeDictActionWrapper(env)
        model: BaseAlgorithm = params.make_algo()(params.policy_type, env, verbose=1)
        model.save(tmp_path / "test.zip")
        export_onnx_app(
            [
                f"{tmp_path}/test.zip",
                f"{tmp_path}/test.onnx",
                "--algorithm",
                params.algo_name.upper(),
            ],
            result_action="return_value",
            exit_on_error=False,
        )
        # Check that the action space is the same as the merged action space, since the original shape wasn't provided
        onnx_model_checker(
            tmp_path / "test.onnx",
            params.observation_space,
            env.action_space,
            state_shapes=params.state_shapes,
            metadata=params.metadata,
        )
