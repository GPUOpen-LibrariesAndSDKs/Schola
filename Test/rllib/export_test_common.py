# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Spaces, cases, fixtures, and helpers shared by the RLlib ONNX and IREE export tests."""

import copy
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from pettingzoo.utils.env import ParallelEnv
from ray.rllib import Policy
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.env import PettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

from schola.core.model import StateMetadata
from Test.export_test_common import (
    box_action_space,
    box_obs_space,
    dict_obs_space,
    discrete_action_space,
    discrete_obs_space,
    multi_discrete_action_space,
    multi_discrete_obs_space,
    offset_box_action_space,
)

dict_action_space = gym.spaces.Dict(
    {
        "o_box": box_action_space,
        "o_discrete": discrete_action_space,
        "o_multi_discrete": multi_discrete_action_space,
    }
)


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
                    # Fixed seed so RLlib and the exported model sample the same observations.
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


def initial_state(state_shapes: Dict[str, Tuple[int, ...]]) -> dict[str, np.ndarray]:
    return {
        f"state_in_{name}": np.zeros((1, *shape), dtype=np.float32)
        for name, shape in state_shapes.items()
    }


def _as_numpy(value: dict[str, Any] | np.ndarray | th.Tensor) -> dict[str, Any] | np.ndarray:
    if isinstance(value, dict):
        return {key: _as_numpy(item) for key, item in value.items()}
    if isinstance(value, th.Tensor):
        value = value.detach().cpu().numpy()
    return value


def record_new_api_actions(
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


def record_old_api_actions(
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
