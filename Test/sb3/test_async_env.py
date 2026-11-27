# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for the SB3 AsyncVecEnv environment (AsyncGrpcProtocol)."""

import asyncio
import functools
from collections.abc import Mapping
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize

from schola.core.protocols.async_base_protocol import AsyncBaseRLProtocol
from schola.core.protocols.base_protocol import AutoResetType
from schola.core.protocols.protobuf.async_grpc_protocol import AsyncGrpcProtocol
from schola.core.protocols.protobuf.grpc_protocol import GrpcProtocol
from schola.core.simulators.base_simulator import (
    BaseSimulator,
    UnsupportedProtocolException,
)
from schola.core.simulators.unreal.editor_simulator import UnrealEditor
from schola.sb3.async_env import AsyncVecEnv

def wrap(env, wrappers):
    if wrappers:
        for wrapper in wrappers:
            env = wrapper(env)
    return env


@pytest.fixture(scope="function")
def sb3_and_async_schola_env(gym_id_and_wrappers, make_env_server):
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


# --- AsyncVecEnv cleanup tests (fake protocols/simulators) ---


class FakeAsyncProtocol(AsyncBaseRLProtocol):
    """Fake AsyncBaseRLProtocol that implements only what the tests exercise."""

    def __init__(
        self,
        close_error: Exception | None = None,
        reset_marker: float = 0.0,
        reset_barrier: asyncio.Barrier | None = None,
    ):
        self.close_error = close_error
        self.close_calls = 0
        self.closed = False
        self.reset_marker = reset_marker
        self.reset_barrier = reset_barrier
        self.reset_calls: list[tuple[list[Any] | None, list[Any] | None]] = []

    async def start(self) -> None:
        pass

    async def send_startup_msg(
        self,
        auto_reset_type: AutoResetType = AutoResetType.SAME_STEP,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        pass

    async def get_definition(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[
        list[list[str]],
        list[dict[str, str]],
        dict[int, dict[str, gym.Space[Any]]],
        dict[int, dict[str, gym.Space[Any]]],
    ]:
        obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        act_space = gym.spaces.Discrete(2)
        return (
            [["agent0"]],
            [{"agent0": "default"}],
            {0: {"agent0": obs_space}},
            {0: {"agent0": act_space}},
        )

    async def send_reset_msg(
        self,
        seeds: list[Any] | None = None,
        options: list[Any] | None = None,
    ) -> tuple[list[dict[str, dict[str, Any]]], list[dict[str, dict[str, str]]]]:
        self.reset_calls.append((seeds, options))
        if self.reset_barrier is not None:
            # Only completes if every participating reset is in flight at once.
            await asyncio.wait_for(self.reset_barrier.wait(), timeout=2.0)
        obs = np.full((2,), self.reset_marker, dtype=np.float32)
        return [{"agent0": obs}], [{"agent0": {}}]

    async def send_action_msg(
        self,
        actions: Mapping[int, Mapping[str, Any]],
        action_space: Mapping[str, gym.Space[Any]],
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, float]],
        list[dict[str, bool]],
        list[dict[str, bool]],
        list[dict[str, dict[str, str]]],
        dict[int, dict[str, dict[str, Any]]],
        dict[int, dict[str, dict[str, str]]],
    ]:
        raise NotImplementedError

    def __bool__(self) -> bool:
        return True

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error
        self.closed = True


class FakeSimulator(BaseSimulator):
    """Fake BaseSimulator that counts start/stop calls."""

    @property
    def supported_async_protocols(self) -> tuple[type[AsyncBaseRLProtocol], ...]:
        return (FakeAsyncProtocol,)

    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0

    def start(self, _protocol_properties: dict[str, object]) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1


def _make_env(
    num_pairs: int = 2,
    failing: tuple[int, ...] = (),
    reset_barrier: asyncio.Barrier | None = None,
):
    sims = [FakeSimulator() for _ in range(num_pairs)]
    protos = [
        FakeAsyncProtocol(
            close_error=RuntimeError(f"close failed {i}") if i in failing else None,
            reset_marker=float(i),
            reset_barrier=reset_barrier,
        )
        for i in range(num_pairs)
    ]
    env = AsyncVecEnv(sims, protos)
    return env, sims, protos


def test_close_closes_all_protocols_and_stops_loop_thread():
    env, sims, protos = _make_env(num_pairs=3)
    thread, loop = env._thread, env._loop
    assert thread is not None and thread.is_alive()
    assert loop is not None
    env.close()

    assert all(p.close_calls == 1 and p.closed for p in protos)
    assert not thread.is_alive()
    assert not loop.is_running()


def test_close_continues_when_one_protocol_close_fails():
    env, sims, protos = _make_env(num_pairs=3, failing=(1,))

    env.close()  # must not raise

    assert [p.close_calls for p in protos] == [1, 1, 1]
    assert protos[0].closed and protos[2].closed
    assert not protos[1].closed


def test_close_stops_all_simulators_after_protocol_failure():
    env, sims, protos = _make_env(num_pairs=3, failing=(0, 2))

    env.close()

    assert [s.stop_calls for s in sims] == [1, 1, 1]
    assert env._thread is not None and not env._thread.is_alive()


# --- AsyncVecEnv reset tests (fake protocols/simulators) ---


def test_reset_runs_protocol_resets_concurrently():
    # Each reset blocks on the barrier, so this only finishes if all three
    # send_reset_msg calls are in flight at the same time.
    env, _, _ = _make_env(num_pairs=3, reset_barrier=asyncio.Barrier(3))
    try:
        env.reset()
    finally:
        env.close()


def test_reset_preserves_protocol_order_in_results():
    env, _, _ = _make_env(num_pairs=3)
    try:
        obs = env.reset()
    finally:
        env.close()

    assert isinstance(obs, np.ndarray)
    assert obs[:, 0].tolist() == [0.0, 1.0, 2.0]


def test_reset_passes_each_protocol_its_own_options_slice():
    env, _, protos = _make_env(num_pairs=3)
    try:
        env.set_options([{"i": 0}, {"i": 1}, {"i": 2}])
        env.reset()
    finally:
        env.close()

    for i, proto in enumerate(protos):
        assert len(proto.reset_calls) == 1
        _, options = proto.reset_calls[0]
        assert options == [{"i": i}]
