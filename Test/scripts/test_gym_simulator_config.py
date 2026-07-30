# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for ``GymSimulatorConfig`` simulator vs environment counts."""

from __future__ import annotations

from schola.core.simulators.gym.simulator import GymSimulator
from schola.scripts.common.settings import GymSimulatorConfig


def test_make_uses_num_environments_not_num_simulators():
    cfg = GymSimulatorConfig(
        env_id="CartPole-v1",
        num_simulators=4,
        num_environments=3,
    )
    sim = cfg.make()
    assert isinstance(sim, GymSimulator)
    assert sim.num_envs == 3
    assert sim.env_id == "CartPole-v1"


def test_make_n_creates_n_distinct_simulators_with_shared_vector_size():
    cfg = GymSimulatorConfig(env_id="CartPole-v1", num_environments=2)
    sims = cfg.make_n(3)
    assert len(sims) == 3
    assert all(sim.num_envs == 2 for sim in sims)
    assert all(sim.env_id == "CartPole-v1" for sim in sims)
    assert len({id(sim) for sim in sims}) == 3

def test_make_n_shares_thread_pool_across_simulators():
    cfg = GymSimulatorConfig(env_id="CartPole-v1", num_environments=2)
    sims = cfg.make_n(3)
    assert all(isinstance(sim, GymSimulator) for sim in sims), "Expected all simulators to be GymSimulator instances"
    unique_thread_pools = {id(sim._thread_pool) for sim in sims}
    assert len(unique_thread_pools) == 1, f"Expected 1 unique thread pool, got {len(unique_thread_pools)}"