# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Tests for gym servicer seed handling."""

from schola.core.simulators.gym.servicer import _seed_from_proto
from schola.generated.StateUpdates_pb2 import EnvironmentSettings


def test_unset_seed_returns_none():
    assert _seed_from_proto(EnvironmentSettings()) is None


def test_zero_seed_returns_none():
    assert _seed_from_proto(EnvironmentSettings(seed=0)) is None


def test_nonzero_seed_returns_value():
    assert _seed_from_proto(EnvironmentSettings(seed=42)) == 42
