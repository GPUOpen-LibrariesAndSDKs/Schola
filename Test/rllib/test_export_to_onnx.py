# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for exporting RLlib policies to ONNX"""

import copy
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import functools
from typing import Any, Dict, Literal, Optional, Tuple
import pytest
import gymnasium as gym
import numpy as np
import onnx
import os
import torch as th
from ray.rllib import Policy
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.env import PettingZooEnv
from schola.rllib.export import (
    _require_policy_spaces,
    _select_policies,
    _spaces_by_policy,
    export_onnx_from_policy,
)
from schola.scripts.rllib.rllib_to_onnx import export_onnx_app
from Test.conftest import (
    check_onnx_metadata,
    check_onnx_model_loads,
    check_onnx_runtime,
)
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
# Bounds away from zero so an unsquashed, unscaled actor mean fails contains().
offset_box_action_space = gym.spaces.Box(low=12, high=14, shape=(3,), dtype=np.float32)
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
    custom_id: Optional[str] = None

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
        if self.use_lstm:
            return {"actor_h": (layer_dim, 64), "actor_c": (layer_dim, 64)}
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
        if self.custom_id:
            return self.custom_id
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
                    self.record_actions = False
                    self.recorded_actions: list[Any] = []
                    # None keeps the env from ending, which is what the export tests use.
                    self.episode_length: int | None = None
                    self.current_step = 0
                    super().__init__()

                def reset(
                    self,
                    *,
                    seed: Optional[int] = None,
                    options: Optional[Dict[str, Any]] = None,
                ):
                    # Fixed seed so RLlib and ONNX sample the same observations.
                    # ``observation_space.sample()`` uses the space RNG, not the env RNG.
                    super().reset(seed=1)
                    self.observation_space.seed(1)
                    self.current_step = 0
                    return self.observation_space.sample(), {}

                def step(self, action):
                    if self.record_actions:
                        self.recorded_actions.append(copy.deepcopy(action))
                    self.current_step += 1
                    terminated = (
                        self.episode_length is not None
                        and self.current_step >= self.episode_length
                    )
                    return (
                        self.observation_space.sample(),
                        0.0,
                        terminated,
                        terminated,
                        {},
                    )

        return TestEnv


def _recording_sub_env(vector_env):
    """Return the gym env inside RLlib's vector env."""
    base = getattr(vector_env, "unwrapped", vector_env)
    envs = getattr(base, "envs", None)
    if not envs:
        raise AssertionError(f"Expected a vector env with sub-envs, got {type(vector_env)}")
    env = envs[0]
    return env.unwrapped if hasattr(env, "unwrapped") else env


def _onnx_inputs(observation: Any, observation_space: gym.Space) -> dict[str, np.ndarray]:
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


def _initial_onnx_state(state_shapes: Dict[str, Tuple[int, ...]]) -> dict[str, np.ndarray]:
    return {
        f"state_in_{name}": np.zeros((1, *shape), dtype=np.float32)
        for name, shape in state_shapes.items()
    }


def evaluate_onnx_policy(
    model: onnx.ModelProto,
    env: gym.Env,
    action_space: gym.Space,
    state_shapes: Dict[str, Tuple[int, ...]] | None = None,
    n_eval_episodes: int = 1,
) -> None:
    """Step ``env`` with an exported ONNX policy, carrying recurrent state when present."""
    import onnxruntime as ort

    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    output_names = [output.name for output in model.graph.output]
    state = _initial_onnx_state(state_shapes or {})
    episodes_completed = 0
    observation, _ = env.reset()
    while episodes_completed < n_eval_episodes:
        outputs = session.run(
            None,
            _onnx_inputs(observation, env.observation_space) | state,
        )
        output_map = dict(zip(output_names, outputs))
        for name in state:
            state[name] = np.asarray(output_map["state_out_" + name[len("state_in_") :]])
        if isinstance(action_space, gym.spaces.Dict):
            action = {
                key: _unbatch_onnx_output(output_map[key]) for key in action_space.spaces
            }
        else:
            action = _unbatch_onnx_output(output_map["action"])
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            episodes_completed += 1
            if episodes_completed < n_eval_episodes:
                observation, _ = env.reset()
                state = _initial_onnx_state(state_shapes or {})


def _assert_recorded_actions_match(
    rllib_actions: list[Any],
    onnx_actions: list[Any],
    action_space: gym.Space,
) -> None:
    assert len(rllib_actions) == len(onnx_actions), (
        "RLlib and ONNX evaluations recorded different numbers of actions: "
        f"{len(rllib_actions)} != {len(onnx_actions)}"
    )
    for step, (rllib_action, onnx_action) in enumerate(zip(rllib_actions, onnx_actions)):
        np.testing.assert_allclose(
            np.asarray(gym.spaces.flatten(action_space, rllib_action), dtype=np.float64),
            np.asarray(gym.spaces.flatten(action_space, onnx_action), dtype=np.float64),
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"Action mismatch at step {step}",
        )


def _as_numpy(value: dict[str, Any] | np.ndarray | th.Tensor) -> dict[str, Any] | np.ndarray:
    if isinstance(value, dict):
        return {key: _as_numpy(item) for key, item in value.items()}
    if isinstance(value, th.Tensor):
        value = value.detach().cpu().numpy()
    return value


def _record_new_api_actions(
    algo: Algorithm,
    action_space: gym.Space,
    episode_length: int,
) -> tuple[list[Any], type]:
    """Record one deterministic episode from the new-API env runner."""
    assert algo.env_runner is not None, "Unexpected Error, env_runner was None during test"
    sub_env = _recording_sub_env(algo.env_runner.env)
    env_cls = type(sub_env)
    sub_env.record_actions = True
    sub_env.episode_length = episode_length
    sub_env.recorded_actions.clear()
    logp_context = (
        _summable_multi_action_logp()
        if isinstance(action_space, gym.spaces.Dict)
        else nullcontext()
    )
    try:
        with logp_context:
            algo.env_runner.sample(num_episodes=1, explore=False)
        return copy.deepcopy(sub_env.recorded_actions), env_cls
    finally:
        # The algorithm is cached, so leave its env non-terminating.
        sub_env.record_actions = False
        sub_env.episode_length = None
        sub_env.recorded_actions.clear()


def _record_old_api_actions(
    algo: Algorithm,
    episode_length: int,
) -> tuple[list[Any], type]:
    """Record one deterministic episode from the old-API policy.

    ``Algorithm.compute_single_action`` applies the observation preprocessor and
    unsquashes normalized actions. Calling the policy directly skips both, so
    its outputs do not match the exported graph.
    """
    assert algo.env_runner is not None, "Unexpected Error, env_runner was None during test"
    sub_env = algo.env_runner.env
    sub_env = getattr(sub_env, "unwrapped", sub_env)
    env_cls = type(sub_env)
    policy: Policy = algo.get_policy("default_policy")
    sub_env.record_actions = True
    sub_env.episode_length = episode_length
    sub_env.recorded_actions.clear()
    try:
        observation, _ = sub_env.reset()
        state = policy.get_initial_state()
        while len(sub_env.recorded_actions) < episode_length:
            result = algo.compute_single_action(
                observation,
                state=state or None,
                explore=False,
            )
            if state:
                action, state, _ = result
            else:
                action = result
            observation, _, terminated, truncated, _ = sub_env.step(_as_numpy(action))
            if terminated or truncated:
                break
        return copy.deepcopy(sub_env.recorded_actions), env_cls
    finally:
        sub_env.record_actions = False
        sub_env.episode_length = None
        sub_env.recorded_actions.clear()


def check_onnx_matches_rllib(
    algo: Algorithm,
    model: onnx.ModelProto,
    action_space: gym.Space,
    state_shapes: Dict[str, Tuple[int, ...]] | None = None,
    episode_length: int = 32,
) -> None:
    """Fixed-seed episodes should make the exported policy match RLlib."""
    assert algo.config is not None, "Unexpected Error, config was None during test"
    if algo.config.enable_env_runner_and_connector_v2:
        rllib_actions, env_cls = _record_new_api_actions(
            algo, action_space, episode_length
        )
    else:
        rllib_actions, env_cls = _record_old_api_actions(algo, episode_length)

    onnx_env = env_cls()
    onnx_env.record_actions = True
    onnx_env.episode_length = episode_length
    evaluate_onnx_policy(
        model,
        onnx_env,
        action_space,
        state_shapes=state_shapes,
    )
    assert len(onnx_env.recorded_actions) == episode_length
    _assert_recorded_actions_match(
        rllib_actions,
        onnx_env.recorded_actions,
        action_space,
    )


@contextmanager
def _summable_multi_action_logp():
    """Let deterministic dict-action log-probs add during ``EnvRunner.sample``.

    ``GetActions`` always requests ``logp`` after it samples. With
    ``explore=False``, each child is a ``TorchDeterministic`` whose ``logp`` is
    ``zeros_like(loc)`` and still has the action shape. ``TorchMultiDistribution``
    then adds a width-3 Box to a width-2 MultiDiscrete and sampling stops before
    the environment is stepped. Summing that action axis makes each child
    log-probability one value per timestep, which is what the stochastic
    ``logp`` methods already return. The action itself is sampled before
    ``logp`` and is unchanged.

    Single-step inference gives a stateful batch shape ``(batch, time=1, ...)``.
    A Discrete index is already ``(batch,)`` or ``(batch, 1)``. A trailing axis
    wider than that time axis is the action axis and is the one summed here.
    """
    from ray.rllib.core.distribution.torch.torch_distribution import TorchDeterministic

    original = TorchDeterministic.logp

    def logp(self, value, **kwargs):
        import torch

        zeros = torch.zeros_like(self.loc)
        if zeros.ndim >= 2 and zeros.shape[-1] != 1:
            zeros = zeros.sum(dim=-1)
        return zeros

    TorchDeterministic.logp = logp
    try:
        yield
    finally:
        TorchDeterministic.logp = original


# Same single-agent cases as the RLModule export test.
rl_module_export_cases = [
    ExportParams("ppo", box_obs_space, box_action_space),
    ExportParams(
        "ppo", box_obs_space, offset_box_action_space, custom_id="PPO(Box, BoxWithOffset)"
    ),
    ExportParams("ppo", discrete_obs_space, box_action_space),
    ExportParams("ppo", box_obs_space, dict_action_space),
    ExportParams("ppo", multi_discrete_obs_space, box_action_space),
    ExportParams("ppo", dict_obs_space, dict_action_space, use_lstm=True),
    ExportParams("ppo", box_obs_space, box_action_space, use_lstm=True),
    ExportParams("ppo", dict_obs_space, dict_action_space),
    ExportParams("sac", box_obs_space, box_action_space),
    ExportParams(
        "sac",
        box_obs_space,
        offset_box_action_space,
        custom_id="SAC(Box, BoxWithOffset)",
    ),
]


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
            # TestExportParams("ppo", box_obs_space, box_action_space, use_attention=True, new_stack=False),
            # TestExportParams("ppo", dict_obs_space, dict_action_space, use_attention=True, new_stack=False),
            ExportParams("ppo", discrete_obs_space, box_action_space, new_stack=False),
            ExportParams("ppo", dict_obs_space, dict_action_space, new_stack=False),
            ExportParams(
                "ppo", multi_discrete_obs_space, box_action_space, new_stack=False
            ),
            ExportParams("sac", box_obs_space, box_action_space, new_stack=False),
            ExportParams(
                "sac",
                box_obs_space,
                offset_box_action_space,
                new_stack=False,
                custom_id="SAC(Box, BoxWithOffset, old)",
            ),
        ],
        ids=str,
    )
    def test_export_rllib_policy_to_onnx(
        self, tmp_path, make_algo, alg_params, ray_cluster
    ):
        """Test exporting RLlib policies to ONNX format."""

        # Get the policy
        algo = make_algo(alg_params)
        policy: Policy = algo.get_policy("default_policy")
        onnx_file = tmp_path / "test_policy" / "model.onnx"

        # Export to ONNX
        export_onnx_from_policy(
            policy, onnx_file, observation_space=alg_params.observation_space
        )

        # Verify the ONNX file was created
        onnx_model = check_onnx_model_loads(onnx_file)
        check_onnx_metadata(
            onnx_model,
            alg_params.observation_space,
            alg_params.action_space,
            state_shapes=alg_params.state_shapes,
            metadata=alg_params.metadata,
        )
        check_onnx_runtime(
            onnx_model,
            alg_params.observation_space,
            alg_params.action_space,
            state_shapes=alg_params.state_shapes,
        )
        check_onnx_matches_rllib(
            algo,
            onnx_model,
            alg_params.action_space,
            state_shapes=alg_params.state_shapes,
        )

    # test exporting just the RLModule to ONNX (direct creation without Ray cluster)
    @pytest.mark.parametrize("alg_params", rl_module_export_cases, ids=str)
    def test_export_rllib_rl_module_to_onnx(
        self, tmp_path, make_algo, alg_params, ray_cluster
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
        onnx_model = check_onnx_model_loads(onnx_file)
        check_onnx_metadata(
            onnx_model,
            alg_params.observation_space,
            alg_params.action_space,
            state_shapes=alg_params.state_shapes,
            metadata=alg_params.metadata,
        )
        check_onnx_runtime(
            onnx_model,
            alg_params.observation_space,
            alg_params.action_space,
            state_shapes=alg_params.state_shapes,
        )
        check_onnx_matches_rllib(
            algo,
            onnx_model,
            alg_params.action_space,
            state_shapes=alg_params.state_shapes,
        )

    # test exporting using the entire algorithm object
    @pytest.mark.parametrize(
        "alg_params",
        [
            # Check different action spaces with dict observation space
            # Test with LSTM enabled
            ExportParams("ppo", box_obs_space, box_action_space, use_lstm=True),
            ExportParams("sac", dict_obs_space, box_action_space),
            ExportParams(
                "sac",
                dict_obs_space,
                offset_box_action_space,
                custom_id="SAC(Dict, BoxWithOffset)",
            ),
        ],
        ids=str,
    )
    def test_export_rllib_algorithm_to_onnx(
        self, tmp_path, make_algo, alg_params, ray_cluster
    ):
        """Test exporting RLlib algorithms to ONNX format from checkpoint using tune.run()."""
        algo = make_algo(alg_params)
        onnx_file = tmp_path / "model.onnx"
        # Export the RLModule from the algorithm object
        export_onnx_from_policy(algo, onnx_file)

        # Verify the ONNX file was created
        onnx_model = check_onnx_model_loads(onnx_file)
        check_onnx_metadata(
            onnx_model,
            alg_params.observation_space,
            alg_params.action_space,
            state_shapes=alg_params.state_shapes,
            metadata=alg_params.metadata,
        )
        check_onnx_runtime(
            onnx_model,
            alg_params.observation_space,
            alg_params.action_space,
            state_shapes=alg_params.state_shapes,
        )
        check_onnx_matches_rllib(
            algo,
            onnx_model,
            alg_params.action_space,
            state_shapes=alg_params.state_shapes,
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
        self, tmp_path, make_algo, alg_params, ray_cluster
    ):
        """Test exporting RLlib multiagent policies to ONNX format."""
        algo = make_algo(alg_params)
        export_onnx_from_policy(algo, tmp_path)

        for i in range(alg_params.num_policies):
            onnx_model = check_onnx_model_loads(tmp_path / f"policy_{i}.onnx")
            check_onnx_metadata(
                onnx_model,
                alg_params.observation_space,
                alg_params.action_space,
                state_shapes=alg_params.state_shapes,
                metadata=alg_params.metadata,
            )
            check_onnx_runtime(
                onnx_model,
                alg_params.observation_space,
                alg_params.action_space,
                state_shapes=alg_params.state_shapes,
            )


@pytest.mark.xdist_group(name="ray-cluster")
class TestExportScript:
    """End-to-end tests for the rllib-to-onnx CLI script."""

    @pytest.mark.parametrize(
        "spaces_yaml",
        [
            pytest.param(
                """
                observation:
                  type: box
                  shape: [4]
                  dtype: float32
                action:
                  type: box
                  shape: [3]
                  dtype: float32
                """,
                id="shared",
            ),
            pytest.param(
                """
                default_policy:
                  observation:
                    type: box
                    shape: [4]
                    dtype: float32
                  action:
                    type: box
                    shape: [3]
                    dtype: float32
                """,
                id="per_policy",
            ),
        ],
    )
    def test_spaces_yaml_overrides_checkpoint_interface(
        self,
        tmp_path,
        make_algo,
        make_spaces_yaml,
        onnx_model_checker,
        ray_cluster,
        spaces_yaml,
    ):
        params = ExportParams("ppo", box_obs_space, box_action_space)
        algo: Algorithm = make_algo(params)
        checkpoint_path = tmp_path / "checkpoint"
        export_path = tmp_path / "model.onnx"
        spaces_path = make_spaces_yaml(spaces_yaml)
        algo.save(checkpoint_path)

        export_onnx_app(
            [
                str(checkpoint_path),
                str(export_path),
                "--spaces-path",
                str(spaces_path),
            ],
            result_action="return_value",
            exit_on_error=False,
        )

        onnx_model_checker(
            export_path,
            box_obs_space,
            box_action_space,
            state_shapes=params.state_shapes,
            metadata=params.metadata,
        )

    @pytest.mark.parametrize(
        "alg_params",
        [
            ExportParams("ppo", box_obs_space, box_action_space),
            ExportParams("ppo", dict_obs_space, dict_action_space, use_lstm=True),
            ExportParams("sac", box_obs_space, box_action_space),
            ExportParams(
                "sac",
                box_obs_space,
                offset_box_action_space,
                custom_id="SAC(Box, BoxWithOffset)",
            ),
        ],
        ids=str,
    )
    def test_rllib_to_onnx_script(
        self, tmp_path, make_algo, alg_params, ray_cluster
    ):
        """Test the rllib-to-onnx script: save checkpoint, run script, verify ONNX output."""
        algo: Algorithm = make_algo(alg_params)
        checkpoint_path = tmp_path / "checkpoint"
        export_path = tmp_path / "model.onnx"
        algo.save(checkpoint_path)
        export_onnx_app(
            [str(checkpoint_path), str(export_path)],
            result_action="return_value",
            exit_on_error=False,
        )
        onnx_model = check_onnx_model_loads(export_path)
        check_onnx_metadata(
            onnx_model,
            alg_params.observation_space,
            alg_params.action_space,
            state_shapes=alg_params.state_shapes,
            metadata=alg_params.metadata,
        )
        check_onnx_runtime(
            onnx_model,
            alg_params.observation_space,
            alg_params.action_space,
            state_shapes=alg_params.state_shapes,
        )
        check_onnx_matches_rllib(
            algo,
            onnx_model,
            alg_params.action_space,
            state_shapes=alg_params.state_shapes,
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


def test_spaces_by_policy_shares_space_or_looks_up_policy_id():
    a_space = gym.spaces.Box(0, 1, (2,))
    b_space = gym.spaces.Box(0, 1, (3,))
    assert _spaces_by_policy(a_space, ["a", "b"]) == {"a": a_space, "b": a_space}
    assert _spaces_by_policy({"a": a_space, "b": b_space}, ["a"]) == {"a": a_space}


def test_select_policies_keeps_only_listed_policies(caplog):
    with caplog.at_level("WARNING", logger="schola.rllib.export"):
        assert _select_policies({"a": 1, "b": 2}, ["a", "typo"]) == {"a": 1}
    assert "['typo']" in caplog.text
    assert "missing from the spaces config: ['b']" in caplog.text


def test_select_policies_rejects_when_nothing_matches():
    with pytest.raises(ValueError, match="None of the policy IDs"):
        _select_policies({"a": 1}, ["typo"])


def test_require_policy_spaces_defaults_every_policy_to_none():
    assert _require_policy_spaces(None, ["a", "b"], "observation") == {
        "a": None,
        "b": None,
    }


@pytest.mark.parametrize("mapping_type", [dict, gym.spaces.Dict], ids=str)
def test_require_policy_spaces_rejects_missing_policy_ids(mapping_type):
    spaces = mapping_type({"a": gym.spaces.Box(0, 1, (2,))})
    assert _require_policy_spaces(spaces, ["a"], "action") is spaces
    with pytest.raises(KeyError, match=r"No action space given for policy IDs \['b'\]"):
        _require_policy_spaces(spaces, ["a", "b"], "action")
