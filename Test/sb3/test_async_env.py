# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for the SB3 AsyncVecEnv environment (AsyncGrpcProtocol)."""

import functools
import sys
from typing import Optional

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize

from schola.core.protocols.protobuf.async_grpc_protocol import AsyncGrpcProtocol
from schola.core.protocols.protobuf.grpc_protocol import GrpcProtocol
from schola.core.simulators.base_simulator import UnsupportedProtocolException
from schola.core.simulators.unreal.editor_simulator import UnrealEditor
from schola.sb3.async_env import AsyncVecEnv

_ASYNC_VECENV_PY311_REASON = "AsyncVecEnv requires Python 3.11+ (asyncio.TaskGroup)"

pytestmark = pytest.mark.xfail(
    sys.version_info < (3, 11),
    reason=_ASYNC_VECENV_PY311_REASON,
    raises=RuntimeError,
)


def wrap(env, wrappers):
    if wrappers:
        for wrapper in wrappers:
            env = wrapper(env)
    return env


@pytest.fixture(scope="function")
def sb3_and_async_schola_env(gym_id_and_wrappers, make_env_server):
    if sys.version_info < (3, 11):
        pytest.xfail(_ASYNC_VECENV_PY311_REASON)

    gym_id, wrappers = gym_id_and_wrappers
    sb3_env = make_vec_env(gym_id, n_envs=1, wrapper_class=lambda x: wrap(x, wrappers))

    env_server_port = make_env_server(gym_id, wrappers)
    simulator = UnrealEditor()
    protocol = AsyncGrpcProtocol(url="localhost", port=env_server_port)
    schola_env = AsyncVecEnv(simulator, protocol)
    yield sb3_env, schola_env

    sb3_env.close()
    schola_env.close()


def test_async_sb3_env_action_space(sb3_and_async_schola_env):
    sb3_env, schola_env = sb3_and_async_schola_env
    assert (
        schola_env.action_space == sb3_env.action_space
    ), f"Expected action space: {sb3_env.action_space} Got: {schola_env.action_space}"


def test_async_sb3_env_observation_space(sb3_and_async_schola_env):
    sb3_env, schola_env = sb3_and_async_schola_env
    assert schola_env.observation_space == sb3_env.observation_space, (
        f"Expected observation space: {sb3_env.observation_space} "
        f"Got: {schola_env.observation_space}"
    )


def test_async_vecenv_rejects_sync_protocol():
    """AsyncVecEnv validates protocol against supported_async_protocols, not supported_protocols."""
    simulator = UnrealEditor()
    sync_protocol = GrpcProtocol("localhost", None)
    with pytest.raises(UnsupportedProtocolException) as exc_info:
        AsyncVecEnv(simulator, sync_protocol)
    assert "not supported" in str(exc_info.value).lower()


def test_async_vecenv_list_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        AsyncVecEnv(
            [UnrealEditor(), UnrealEditor()],
            [AsyncGrpcProtocol(url="localhost", port=12345)],
        )


def test_async_vecenv_two_servers(make_env_server):
    """Two simulators / protocols merged into one VecEnv (one event loop)."""
    port0 = make_env_server("CartPole-v1")
    port1 = make_env_server("CartPole-v1")
    sb3_env = make_vec_env("CartPole-v1", n_envs=2)
    schola_env = AsyncVecEnv(
        [UnrealEditor(), UnrealEditor()],
        [
            AsyncGrpcProtocol(url="localhost", port=port0),
            AsyncGrpcProtocol(url="localhost", port=port1),
        ],
    )
    try:
        assert schola_env.num_envs == 2
        assert schola_env.action_space == sb3_env.action_space
        assert schola_env.observation_space == sb3_env.observation_space
        acts = np.zeros((2,), dtype=np.int64)
        schola_env.reset()
        obs, rew, done, infos = schola_env.step(acts)
        assert obs.shape[0] == 2
        assert rew.shape == (2,)
        assert done.shape == (2,)
        assert len(infos) == 2
    finally:
        sb3_env.close()
        schola_env.close()


def test_async_vecenv_merged_segments_step(make_vec_env_server):
    """Server A: 2 envs, server B: 3 envs → 5 flat envs, step_async/step_wait merge."""
    port0 = make_vec_env_server([lambda: gym.make("CartPole-v1") for _ in range(2)])
    port1 = make_vec_env_server([lambda: gym.make("CartPole-v1") for _ in range(3)])
    sb3_env = make_vec_env("CartPole-v1", n_envs=5)
    schola_env = AsyncVecEnv(
        [UnrealEditor(), UnrealEditor()],
        [
            AsyncGrpcProtocol(url="localhost", port=port0),
            AsyncGrpcProtocol(url="localhost", port=port1),
        ],
    )
    try:
        assert schola_env.num_envs == 5
        assert schola_env.observation_space == sb3_env.observation_space
        acts = np.zeros((5,), dtype=np.int64)
        schola_env.reset()
        for _ in range(3):
            obs, rew, done, infos = schola_env.step(acts)
            assert obs.shape[0] == 5
            assert rew.shape == (5,)
            assert len(infos) == 5
    finally:
        sb3_env.close()
        schola_env.close()


@pytest.mark.skip()
def test_async_sb3_env_close(make_env_server):
    gym_id = "CartPole-v1"
    env_server_port = make_env_server(gym_id)
    simulator = UnrealEditor()
    protocol = AsyncGrpcProtocol(url="localhost", port=env_server_port)
    env = AsyncVecEnv(simulator, protocol)
    env.close()


# The below StepEnv and terminal-obs test are adapted from stable-baselines3 tests
# (see Test/sb3/test_env.py for license header).


class StepEnv(gym.Env):
    def __init__(self, max_steps):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            np.array([0]), np.array([999]), dtype="int"
        )
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.current_step = 0
        return np.array([self.current_step], dtype="int"), {}

    def step(self, action):
        prev_step = self.current_step
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps
        return np.array([prev_step], dtype="int"), 0.0, terminated, truncated, {}


VEC_ENV_WRAPPERS = [None, VecNormalize, VecFrameStack]
N_ENVS = 3


@pytest.mark.parametrize("vec_env_wrapper", VEC_ENV_WRAPPERS)
def test_async_vecenv_terminal_obs(make_vec_env_server, vec_env_wrapper):
    """Test that 'terminal_observation' gets added to info dict upon termination."""

    step_nums = [i + 5 for i in range(N_ENVS)]
    env_funcs = [functools.partial(StepEnv, n) for n in step_nums]
    env_server_port = make_vec_env_server(env_funcs)
    simulator = UnrealEditor()
    protocol = AsyncGrpcProtocol(url="localhost", port=env_server_port)
    schola_env = AsyncVecEnv(simulator, protocol)

    if vec_env_wrapper is not None:
        if vec_env_wrapper == VecFrameStack:
            schola_env = vec_env_wrapper(schola_env, n_stack=2)
        else:
            schola_env = vec_env_wrapper(schola_env)

    zero_acts = np.zeros((N_ENVS,), dtype="int")
    prev_obs_b = schola_env.reset()
    for step_num in range(1, max(step_nums) + 1):
        obs_b, _, done_b, info_b = schola_env.step(zero_acts)
        assert len(obs_b) == N_ENVS
        assert len(done_b) == N_ENVS
        assert len(info_b) == N_ENVS
        env_iter = zip(prev_obs_b, obs_b, done_b, info_b, step_nums)
        for prev_obs, obs, done, info, final_step_num in env_iter:
            assert done == (step_num == final_step_num)
            if not done:
                assert "terminal_observation" not in info
            else:
                terminal_obs = info["terminal_observation"]
                assert np.all(prev_obs < terminal_obs)
                assert np.all(obs < prev_obs)
                if not isinstance(schola_env, VecNormalize):
                    assert np.all(prev_obs + 1 == terminal_obs)
                    assert np.all(obs == 0)
        prev_obs_b = obs_b

    schola_env.close()
