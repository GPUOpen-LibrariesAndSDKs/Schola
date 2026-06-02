# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for the SB3 VecEnv environment"""

import pytest
from schola.core.protocols.protobuf.grpc_protocol import GrpcProtocol
import gymnasium as gym
import numpy as np
from typing import Optional
import functools
from unittest.mock import MagicMock

from schola.sb3.env import VecEnv
from schola.core.simulators.unreal.editor_simulator import UnrealEditor
from stable_baselines3.common.env_util import make_vec_env


def wrap(env, wrappers):
    if wrappers:
        for wrapper in wrappers:
            env = wrapper(env)
    return env


@pytest.fixture(scope="function")
def sb3_and_schola_env(gym_id_and_wrappers, make_env_server):
    gym_id, wrappers = gym_id_and_wrappers
    sb3_env = make_vec_env(gym_id, n_envs=1, wrapper_class=lambda x: wrap(x, wrappers))

    env_server_port = make_env_server(gym_id, wrappers)
    simulator = UnrealEditor()
    protocol = GrpcProtocol(url="localhost", port=env_server_port)
    schola_env = VecEnv(simulator, protocol)
    yield sb3_env, schola_env

    sb3_env.close()
    schola_env.close()


@pytest.fixture(scope="function")
def schola_env(make_env_server, gym_id_and_wrappers):
    gym_id, wrappers = gym_id_and_wrappers
    env_server_port = make_env_server(gym_id, wrappers)
    simulator = UnrealEditor()
    protocol = GrpcProtocol(url="localhost", port=env_server_port)
    return VecEnv(simulator, protocol)


def test_sb3_env_action_space(sb3_and_schola_env):
    sb3_env, schola_env = sb3_and_schola_env
    assert (
        schola_env.action_space == sb3_env.action_space
    ), f"Expected action space: {sb3_env.action_space} Got: {schola_env.action_space}"


def test_sb3_env_observation_space(sb3_and_schola_env):
    sb3_env, schola_env = sb3_and_schola_env

    assert (
        schola_env.observation_space == sb3_env.observation_space
    ), f"Expected observation space: {sb3_env.observation_space} Got: {schola_env.observation_space}"


@pytest.mark.skip()
def test_sb3_env_close(make_env_server):
    gym_id = "CartPole-v1"
    env_server_port = make_env_server(gym_id)
    simulator = UnrealEditor()
    protocol = GrpcProtocol(url="localhost", port=env_server_port)
    env = VecEnv(simulator, protocol)

    env.close()


@pytest.fixture(scope="function")
def make_stub_vec_env(stub_protocol_class, stub_simulator_class):
    """Factory for a real ``VecEnv`` built against stub protocol/simulator
    instances, so the full ``VecEnv.__init__`` flow runs without subclassing.

    ``_process_reset`` is replaced with a sentinel mock so set_options/reset
    tests can assert on protocol traffic without caring about the obs/info
    shape ``send_reset_msg`` returns.
    """

    def _factory() -> VecEnv:
        protocol = stub_protocol_class()
        simulator = stub_simulator_class()
        env = VecEnv(simulator, protocol)
        env._process_reset = MagicMock(return_value="sentinel_obs")
        return env

    return _factory


def test_set_options_then_reset_forwards_to_protocol(make_stub_vec_env):
    """After ``set_options(opts)``, the next ``reset()`` forwards the broadcast
    options list to ``protocol.send_reset_msg(options=...)``."""
    env = make_stub_vec_env()
    opts = {"level": "67", "curriculum": "sorta_hard"}

    env.set_options(opts)
    result = env.reset()

    env.protocol.send_reset_msg.assert_called_once_with(
        seeds=[None] * env.num_envs,
        options=[opts] * env.num_envs,
    )
    assert result == "sentinel_obs"


def test_reset_without_set_options_forwards_empty_per_env_options(make_stub_vec_env):
    """If ``set_options`` is never called, ``reset()`` should forward the
    default per-sub-env empty-dict options to the protocol (not None, not a
    bare dict)."""
    env = make_stub_vec_env()

    env.reset()

    env.protocol.send_reset_msg.assert_called_once_with(
        seeds=[None] * env.num_envs,
        options=[{}] * env.num_envs,
    )


def test_options_are_consumed_after_one_reset(make_stub_vec_env):
    """``set_options`` is one-shot: a second ``reset()`` after the first must
    not re-send the previously supplied options."""
    env = make_stub_vec_env()
    env.set_options({"level": "67"})

    env.reset()
    env.protocol.send_reset_msg.reset_mock()

    env.reset()

    env.protocol.send_reset_msg.assert_called_once_with(
        seeds=[None] * env.num_envs,
        options=[{}] * env.num_envs,
    )


def test_set_options_survive_vec_monitor_wrap(make_stub_vec_env):
    """Regression guard: wrapping with ``VecMonitor`` AFTER ``set_options`` must
    not clobber the options that were set on the underlying env.

    ``VecEnvWrapper.__init__`` invokes ``VecEnv.__init__`` on the wrapper, which
    creates a fresh ``_options`` list -- but only on the wrapper instance, not on
    the wrapped venv. ``VecMonitor.reset()`` then delegates to ``self.venv.reset()``,
    which reads the wrapped env's preserved ``_options``. This test pins that
    invariant so a future SB3 change (or local refactor) can't silently swallow
    ``env_options`` in ``eval.main`` / ``train.main``.
    """
    from stable_baselines3.common.vec_env import VecMonitor

    env = make_stub_vec_env()
    opts = {"level": "67", "curriculum": "sorta_hard"}

    env.set_options(options=opts)
    monitored = VecMonitor(env)
    monitored.reset()

    env.protocol.send_reset_msg.assert_called_once_with(
        seeds=[None] * env.num_envs,
        options=[opts] * env.num_envs,
    )


# The below code is adapted from https://github.com/DLR-RM/stable-baselines3/blob/master/tests/test_vec_envs.py

"""
The MIT License

Copyright (c) 2019 Antonin Raffin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

HE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.


class StepEnv(gym.Env):
    def __init__(self, max_steps):
        """Gym environment for testing that terminal observation is inserted
        correctly."""
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


from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize

VEC_ENV_WRAPPERS = [None, VecNormalize, VecFrameStack]
N_ENVS = 3


@pytest.mark.parametrize("vec_env_wrapper", VEC_ENV_WRAPPERS)
def test_vecenv_terminal_obs(make_vec_env_server, vec_env_wrapper):
    """Test that 'terminal_observation' gets added to info dict upon
    termination."""

    step_nums = [i + 5 for i in range(N_ENVS)]
    env_funcs = [functools.partial(StepEnv, n) for n in step_nums]
    env_server_port = make_vec_env_server(env_funcs)
    simulator = UnrealEditor()
    protocol = GrpcProtocol(url="localhost", port=env_server_port)
    schola_env = VecEnv(simulator, protocol)  # just test it can be created and closed

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

                # do some rough ordering checks that should work for all
                # wrappers, including VecNormalize
                assert np.all(prev_obs < terminal_obs)
                assert np.all(obs < prev_obs)

                if not isinstance(schola_env, VecNormalize):
                    # more precise tests that we can't do with VecNormalize
                    # (which changes observation values)
                    assert np.all(prev_obs + 1 == terminal_obs)
                    assert np.all(obs == 0)

        prev_obs_b = obs_b

    schola_env.close()
