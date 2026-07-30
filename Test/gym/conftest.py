# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""Pytest fixtures for Schola Gymnasium integration tests."""

import pytest

import gymnasium as gym
from schola.gym.env import GymEnv
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
