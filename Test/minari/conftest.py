# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""Pytest fixtures for Minari data collection tests."""

import pytest


@pytest.fixture(
    scope="function",
    params=[("CartPole-v1", None), ("MountainCar-v0", None)],
    ids=lambda x: x[0],
)
def imitation_id_and_wrappers(request):
    env_id, wrappers = request.param
    return env_id, wrappers
