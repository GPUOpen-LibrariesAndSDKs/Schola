# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for exporting SB3 policies to ONNX"""

import copy
import functools
from dataclasses import dataclass
from typing import Any, Literal, Optional, Dict, Tuple
import onnx
import onnxruntime as ort
import pytest
import gymnasium as gym
import numpy as np
from schola.sb3.export import save_model_as_onnx
import stable_baselines3 as sb3
from schola.scripts.sb3.sb3_to_onnx import export_onnx_app
from schola.sb3.utils import VecMergeDictActionWrapper, split_value
from stable_baselines3.common.evaluation import evaluate_policy
from Test.conftest import (
    check_onnx_metadata,
    check_onnx_model_loads,
    check_onnx_runtime,
)

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
# Bounds away from zero so an unsquashed, unscaled actor mean fails contains().
offset_box_action_space = gym.spaces.Box(low=12, high=14, shape=(3,), dtype=np.float32)
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
                # Fixed seed so the SB3 and ONNX evaluations sample the same observations.
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


def _onnx_inputs(observation: Any, observation_space: gym.Space) -> dict[str, np.ndarray]:
    """Batch one environment observation the way the exported graph expects it."""
    if isinstance(observation_space, gym.spaces.Dict):
        return {
            key: np.expand_dims(np.asarray(observation[key]), axis=0)
            for key in observation_space.spaces
        }
    return {"obs": np.expand_dims(np.asarray(observation), axis=0)}


def _unbatch_onnx_output(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.shape[:1] == (1,):
        return array[0]
    return array


def _action_from_onnx(
    outputs: Any,
    output_names: list[str],
    action_space: gym.Space,
) -> Any:
    output_map = {
        name: value
        for name, value in zip(output_names, outputs)
        if not name.startswith("state_out_")
    }
    if isinstance(action_space, gym.spaces.Dict):
        return {
            key: _unbatch_onnx_output(output_map[key]) for key in action_space.spaces
        }
    return _unbatch_onnx_output(output_map["action"])


def evaluate_onnx_policy(
    model: onnx.ModelProto,
    env: gym.Env,
    action_space: gym.Space,
    n_eval_episodes: int = 1,
) -> None:
    """Evaluate an exported ONNX policy by stepping ``env`` with its actions.

    This is the ONNX counterpart of
    ``stable_baselines3.common.evaluation.evaluate_policy`` for one
    non-vectorized environment. ``env.step`` records each action.
    """
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    output_names = [output.name for output in model.graph.output]
    episodes_completed = 0
    observation, _ = env.reset()
    while episodes_completed < n_eval_episodes:
        outputs = session.run(None, _onnx_inputs(observation, env.observation_space))
        action = _action_from_onnx(outputs, output_names, action_space)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            episodes_completed += 1
            if episodes_completed < n_eval_episodes:
                observation, _ = env.reset()

def _action_in_exported_space(action: Any, exported_action_space: gym.Space) -> Any:
    """Return ``action`` as a point in ``exported_action_space``.

    SB3 policies trained on a merged dictionary action space emit one flat array.
    The exported graph emits the original dictionary. ``split_value`` restores
    that dictionary so both sides can be flattened the same way.
    """
    if isinstance(exported_action_space, gym.spaces.Dict) and not isinstance(action, dict):
        return split_value(np.asarray(action), exported_action_space)
    return action


def check_onnx_matches_sb3(
    model: BaseAlgorithm,
    onnx_model: onnx.ModelProto,
    params: ONNXExportParams,
    action_space: gym.Space | None = None,
    episode_length: int = 32,
) -> None:
    """Fixed-seed episodes should make the exported policy match SB3.

    ``action_space`` is the space the ONNX graph emits. It defaults to
    ``params.action_space``. Pass the merged space when the export collapsed a
    dictionary action space and did not keep the original keys.
    """
    exported_action_space = (
        params.action_space if action_space is None else action_space
    )
    env_cls = params.get_env_class(
        record_actions=True, episode_length=episode_length
    )
    eval_env = make_vec_env(env_cls, 1)
    evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=1,
        deterministic=True,
        warn=False,
    )
    sb3_actions = eval_env.get_attr("recorded_actions")[0]

    onnx_eval_env = env_cls()
    evaluate_onnx_policy(
        onnx_model,
        onnx_eval_env,
        action_space=exported_action_space,
        n_eval_episodes=1,
    )
    assert len(onnx_eval_env.recorded_actions) == episode_length
    _assert_recorded_actions_match(
        sb3_actions,
        onnx_eval_env.recorded_actions,
        exported_action_space,
    )


def _assert_recorded_actions_match(
    sb3_actions: list[Any],
    onnx_actions: list[Any],
    exported_action_space: gym.Space,
) -> None:
    assert len(sb3_actions) == len(onnx_actions), (
        "SB3 and ONNX evaluations recorded different numbers of actions: "
        f"{len(sb3_actions)} != {len(onnx_actions)}"
    )
    for step, (sb3_action, onnx_action) in enumerate(zip(sb3_actions, onnx_actions)):
        flat_sb3_action = gym.spaces.flatten(
            exported_action_space,
            _action_in_exported_space(sb3_action, exported_action_space),
        )
        flat_onnx_action = gym.spaces.flatten(
            exported_action_space,
           onnx_action,
        )
        np.testing.assert_allclose(
            np.asarray(flat_sb3_action, dtype=np.float64),
            np.asarray(flat_onnx_action, dtype=np.float64),
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"Action mismatch at step {step}",
        )
        
def make_all_combinations(alg_name):
    """Helper function to generate all combinations for an algorithm."""
    return [
        # Check different action spaces
        ONNXExportParams(alg_name, dict_obs_space, box_action_space),
        ONNXExportParams(alg_name, dict_obs_space, offset_box_action_space, custom_id=f"{alg_name.upper()}(Dict, BoxWithOffset)"),
        ONNXExportParams(alg_name, dict_obs_space, discrete_action_space),
        ONNXExportParams(alg_name, dict_obs_space, binary_action_space),
        ONNXExportParams(alg_name, dict_obs_space, multi_discrete_action_space),
        # Check different observation spaces
        ONNXExportParams(alg_name, box_obs_space, box_action_space),
        ONNXExportParams(alg_name, discrete_obs_space, box_action_space),
        ONNXExportParams(alg_name, binary_obs_space, box_action_space),
        ONNXExportParams(alg_name, multi_discrete_obs_space, box_action_space),
    ]


export_function_cases = [
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
    ONNXExportParams(
        "sac",
        dict_obs_space,
        offset_box_action_space,
        custom_id="SAC(Dict, BoxWithOffset)",
    ),
    # TD3
    ONNXExportParams("td3", box_obs_space, box_action_space),
    ONNXExportParams("td3", discrete_obs_space, box_action_space),
    ONNXExportParams("td3", binary_obs_space, box_action_space),
    ONNXExportParams("td3", multi_discrete_obs_space, box_action_space),
    ONNXExportParams("td3", dict_obs_space, box_action_space),
    ONNXExportParams(
        "td3",
        dict_obs_space,
        offset_box_action_space,
        custom_id="TD3(Dict, BoxWithOffset)",
    ),
    # DDPG
    ONNXExportParams("ddpg", box_obs_space, box_action_space),
    ONNXExportParams("ddpg", discrete_obs_space, box_action_space),
    ONNXExportParams("ddpg", binary_obs_space, box_action_space),
    ONNXExportParams("ddpg", multi_discrete_obs_space, box_action_space),
    ONNXExportParams("ddpg", dict_obs_space, box_action_space),
    ONNXExportParams(
        "ddpg",
        dict_obs_space,
        offset_box_action_space,
        custom_id="DDPG(Dict, BoxWithOffset)",
    ),
    # A2C
    *make_all_combinations("a2c"),
    # DQN
    ONNXExportParams("dqn", box_obs_space, discrete_action_space),
    ONNXExportParams("dqn", binary_obs_space, discrete_action_space),
    ONNXExportParams("dqn", dict_obs_space, discrete_action_space),
    ONNXExportParams("dqn", discrete_obs_space, discrete_action_space),
    ONNXExportParams("dqn", multi_discrete_obs_space, discrete_action_space),
]


class TestExportFunction:
    # Test exporting SB3 policies to ONNX, tests with every algorithm from SB3
    @pytest.mark.parametrize("export_params", export_function_cases, ids=str)
    def test_export_sb3_policy_to_onnx(self, tmp_path, export_params):
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

        onnx_model = check_onnx_model_loads(tmp_path / "test.onnx")
        check_onnx_metadata(
            onnx_model,
            params.observation_space,
            params.action_space,
            state_shapes=params.state_shapes,
            metadata=params.metadata,
        )
        check_onnx_runtime(
            onnx_model,
            params.observation_space,
            params.action_space,
            state_shapes=params.state_shapes,
        )
        check_onnx_matches_sb3(model, onnx_model, params)


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
    def test_algorithms(self, tmp_path, export_params):
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

        onnx_model = check_onnx_model_loads(tmp_path / "test.onnx")
        check_onnx_metadata(
            onnx_model,
            params.observation_space,
            params.action_space,
            state_shapes=params.state_shapes,
            metadata=params.metadata,
        )
        check_onnx_runtime(
            onnx_model,
            params.observation_space,
            params.action_space,
            state_shapes=params.state_shapes,
        )
        check_onnx_matches_sb3(model, onnx_model, params)

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
    def test_merged_dict_action_space(self, tmp_path, export_params):
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
        onnx_model = check_onnx_model_loads(tmp_path / "test.onnx")
        check_onnx_metadata(
            onnx_model,
            params.observation_space,
            params.action_space,
            state_shapes=params.state_shapes,
            metadata=params.metadata,
        )
        check_onnx_runtime(
            onnx_model,
            params.observation_space,
            params.action_space,
            state_shapes=params.state_shapes,
        )
        check_onnx_matches_sb3(model, onnx_model, params)

    @pytest.mark.parametrize(
        "export_params",
        [
            ONNXExportParams("ppo", dict_obs_space, box_dict_action_space),
        ],
        ids=str,
    )
    def test_merge_with_no_hidden_state(self, tmp_path, export_params):
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
        onnx_model = check_onnx_model_loads(tmp_path / "test.onnx")
        check_onnx_metadata(
            onnx_model,
            params.observation_space,
            env.action_space,
            state_shapes=params.state_shapes,
            metadata=params.metadata,
        )
        check_onnx_runtime(
            onnx_model,
            params.observation_space,
            env.action_space,
            state_shapes=params.state_shapes,
        )
        check_onnx_matches_sb3(
            model, onnx_model, params, action_space=env.action_space
        )
