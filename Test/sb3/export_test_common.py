# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Spaces, cases, and helpers shared by the SB3 ONNX and IREE export tests."""

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3

from schola.sb3.utils import split_value
from Test.export_test_common import (
    binary_action_space,
    binary_obs_space,
    box_action_space,
    box_obs_space,
    dict_obs_space,
    discrete_action_space,
    discrete_obs_space,
    multi_discrete_action_space,
    multi_discrete_obs_space,
    offset_box_action_space,
)

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
class ExportParams:
    """Dataclass for handling different SB3 export configurations."""

    algo_name: str
    observation_space: gym.Space
    action_space: gym.Space
    buffer_size: int = 10000
    custom_id: str | None = None

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
        if self.custom_id:
            return self.custom_id
        # If the action space is a dict add the name of the action space inside it (all will be the same so just one is enough)
        if isinstance(self.action_space, gym.spaces.Dict):
            action_space_subtype = type(
                list(self.action_space.spaces.values())[0]
            ).__name__
            return f"{self.algo_name.upper()}({type(self.observation_space).__name__},{type(self.action_space).__name__}-{action_space_subtype})"
        return f"{self.algo_name.upper()}({type(self.observation_space).__name__},{type(self.action_space).__name__})"

    def get_env_class(self, record_actions: bool = False, episode_length: int = 30):
        obs_space = copy.deepcopy(self.observation_space)
        action_space = copy.deepcopy(self.action_space)

        class TestEnv(gym.Env):
            def __init__(self):
                self.observation_space = obs_space
                self.action_space = action_space
                self.record_actions = record_actions
                self.recorded_actions: list[Any] = []
                self.episode_length = episode_length
                self.current_step = 0

                super().__init__()

            def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
                # Fixed seed so the SB3 and exported-model evaluations sample the same observations.
                # The env RNG is not what ``observation_space.sample()`` uses.
                super().reset(seed=1)
                self.observation_space.seed(1)
                self.current_step = 0
                return self.observation_space.sample(), {}

            def step(self, action):
                if self.record_actions:
                    self.recorded_actions.append(copy.deepcopy(action))
                observation = self.observation_space.sample()
                reward = 0
                terminated = False
                truncated = False
                info = {}
                self.current_step += 1
                if self.current_step >= self.episode_length:
                    terminated = True
                    truncated = True
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


def action_in_exported_space(action: Any, exported_action_space: gym.Space) -> Any:
    """Return ``action`` as a point in ``exported_action_space``.

    SB3 policies trained on a merged dictionary action space emit one flat array.
    The exported model emits the original dictionary. ``split_value`` restores
    that dictionary so both sides can be flattened the same way.
    """
    if isinstance(exported_action_space, gym.spaces.Dict) and not isinstance(action, dict):
        return split_value(np.asarray(action), exported_action_space)
    return action


def make_all_combinations(alg_name):
    """Helper function to generate all combinations for an algorithm."""
    return [
        # Check different action spaces
        ExportParams(alg_name, dict_obs_space, box_action_space),
        ExportParams(alg_name, dict_obs_space, offset_box_action_space, custom_id=f"{alg_name.upper()}(Dict, BoxWithOffset)"),
        ExportParams(alg_name, dict_obs_space, discrete_action_space),
        ExportParams(alg_name, dict_obs_space, binary_action_space),
        ExportParams(alg_name, dict_obs_space, multi_discrete_action_space),
        # Check different observation spaces
        ExportParams(alg_name, box_obs_space, box_action_space),
        ExportParams(alg_name, discrete_obs_space, box_action_space),
        ExportParams(alg_name, binary_obs_space, box_action_space),
        ExportParams(alg_name, multi_discrete_obs_space, box_action_space),
    ]


export_function_cases = [
    # Testing for merged dict action spaces
    ExportParams("ppo", dict_obs_space, box_dict_action_space),
    ExportParams("ppo", dict_obs_space, discrete_dict_action_space),
    ExportParams("ppo", dict_obs_space, binary_dict_action_space),
    ExportParams("ppo", dict_obs_space, multi_discrete_dict_action_space),
    # PPO
    *make_all_combinations("ppo"),
    # SAC
    # try all obs spaces but only Box action spaces are supported
    ExportParams("sac", box_obs_space, box_action_space),
    ExportParams("sac", discrete_obs_space, box_action_space),
    ExportParams("sac", binary_obs_space, box_action_space),
    ExportParams("sac", multi_discrete_obs_space, box_action_space),
    ExportParams("sac", dict_obs_space, box_action_space),
    ExportParams(
        "sac",
        dict_obs_space,
        offset_box_action_space,
        custom_id="SAC(Dict, BoxWithOffset)",
    ),
    # TD3
    ExportParams("td3", box_obs_space, box_action_space),
    ExportParams("td3", discrete_obs_space, box_action_space),
    ExportParams("td3", binary_obs_space, box_action_space),
    ExportParams("td3", multi_discrete_obs_space, box_action_space),
    ExportParams("td3", dict_obs_space, box_action_space),
    ExportParams(
        "td3",
        dict_obs_space,
        offset_box_action_space,
        custom_id="TD3(Dict, BoxWithOffset)",
    ),
    # DDPG
    ExportParams("ddpg", box_obs_space, box_action_space),
    ExportParams("ddpg", discrete_obs_space, box_action_space),
    ExportParams("ddpg", binary_obs_space, box_action_space),
    ExportParams("ddpg", multi_discrete_obs_space, box_action_space),
    ExportParams("ddpg", dict_obs_space, box_action_space),
    ExportParams(
        "ddpg",
        dict_obs_space,
        offset_box_action_space,
        custom_id="DDPG(Dict, BoxWithOffset)",
    ),
    # A2C
    *make_all_combinations("a2c"),
    # DQN
    ExportParams("dqn", box_obs_space, discrete_action_space),
    ExportParams("dqn", binary_obs_space, discrete_action_space),
    ExportParams("dqn", dict_obs_space, discrete_action_space),
    ExportParams("dqn", discrete_obs_space, discrete_action_space),
    ExportParams("dqn", multi_discrete_obs_space, discrete_action_space),
]
