# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""Tests for ``GymEnv`` and ``GymVectorEnv`` against local gRPC test servers."""

import warnings

import pytest

import gymnasium as gym
from schola.gym.env import GymEnv, GymVectorEnv
from schola.core.protocols.protobuf.grpc_protocol import GrpcProtocol
from schola.core.simulators.unreal.editor_simulator import UnrealEditor
import minigrid


def wrap(env, wrappers):
    if wrappers:
        for wrapper in wrappers:
            env = wrapper(env)
    return env


@pytest.fixture(scope="function")
def schola_env(make_env_server, gym_id_and_wrappers):
    gym_id, wrappers = gym_id_and_wrappers
    env_server_port = make_env_server(gym_id, wrappers)
    simulator = UnrealEditor()
    protocol = GrpcProtocol(url="localhost", port=env_server_port)
    return GymEnv(simulator, protocol)


@pytest.fixture(scope="function")
def vec_schola_env(make_vec_env_server, gym_id_and_wrappers):
    gym_id, wrappers = gym_id_and_wrappers
    env_server_port = make_vec_env_server(gym_id, wrappers)
    simulator = UnrealEditor()
    protocol = GrpcProtocol(url="localhost", port=env_server_port)
    return GymVectorEnv(simulator, protocol)


@pytest.fixture(scope="function")
def gym_and_schola_env(gym_id_and_wrappers, make_env_server):
    gym_id, wrappers = gym_id_and_wrappers
    gym_env = wrap(gym.make(gym_id), wrappers)
    env_server_port = make_env_server(gym_id, wrappers)
    simulator = UnrealEditor()
    protocol = GrpcProtocol(url="localhost", port=env_server_port)
    schola_env = GymEnv(simulator, protocol)
    yield gym_env, schola_env

    # schola env closing is handled by make_env_server fixture, but we close it again here anyway for safety
    schola_env.close()
    gym_env.close()


def test_gym_env_action_space(gym_and_schola_env):
    gym_env, schola_env = gym_and_schola_env

    assert (
        schola_env.action_space == gym_env.action_space
    ), f"Expected action space: {gym_env.action_space} Got: {schola_env.action_space}"


def test_gym_env_observation_space(gym_and_schola_env):
    gym_env, schola_env = gym_and_schola_env

    assert (
        schola_env.observation_space == gym_env.observation_space
    ), f"Expected observation space: {gym_env.observation_space} Got: {schola_env.observation_space}"


def test_gym_env_env_checker(schola_env):
    obs, info = schola_env.reset()
    from gymnasium.utils.env_checker import check_env

    # CartPole-v1 uses unbounded velocity dimensions in observation space; Gymnasium warns by design.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=r".*[Ii]nfinit.*",
        )
        check_env(schola_env, skip_render_check=True)


@pytest.mark.skip(reason="Test not implemented yet")
def test_env_close(make_env_server): ...


@pytest.mark.skip(reason="Test not implemented yet")
def test_gym_vec_env_close(make_env_server, schola_env): ...
