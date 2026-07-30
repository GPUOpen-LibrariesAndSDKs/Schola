# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Tests for gym servicer seed handling and autoreset."""

from functools import partial

import pytest
from gymnasium.spaces import Discrete

from schola.core.protocols.protobuf.deserialize import from_proto
from schola.core.protocols.protobuf.serialize import fill_generic, to_proto
from schola.core.simulators.gym.servicer import GymToGymServiceServicer, _seed_from_proto
from schola.generated.GymConnector_pb2 import AutoResetType, GymConnectorStartRequest
from schola.generated.StateUpdates_pb2 import Reset, StateUpdate, Step
from Test.gym.testing_env import GenericTestEnv
from schola.generated.StateUpdates_pb2 import EnvironmentSettings


def count_reset(
    self: GenericTestEnv, seed: int | None = None, options: dict | None = None
):
    super(GenericTestEnv, self).reset(seed=seed)
    self.count = seed if seed is not None else 0
    return self.count, {}


def make_count_step(max_count):
    def count_step(self: GenericTestEnv, action):
        self.count += 1
        return self.count, action, self.count == max_count, False, {}

    return count_step


def _make_env_factory(max_count: int = 2):
    return partial(
        GenericTestEnv,
        action_space=Discrete(5),
        observation_space=Discrete(5),
        reset_func=count_reset,
        step_func=make_count_step(max_count),
    )


def _start_servicer(servicer: GymToGymServiceServicer, autoreset_type: AutoResetType):
    servicer.StartGymConnector(
        GymConnectorStartRequest(autoreset_type=autoreset_type), None
    )


def _reset(servicer: GymToGymServiceServicer):
    return servicer.UpdateState(StateUpdate(reset=Reset()), None)


def _step(servicer: GymToGymServiceServicer, action: int):
    state_update = StateUpdate(step=Step())
    env_update = state_update.step.environments.add()
    fill_generic(
        to_proto(servicer.env.action_space, action),
        env_update.updates["single_agent"],
    )
    return servicer.UpdateState(state_update, None)


def _agent_state_values(state):
    agent = state.training_state.environment_states[0].agent_states["single_agent"]
    obs, reward, terminated, truncated, _info = from_proto(agent)
    return obs, reward, terminated, truncated


def _initial_obs_value(state):
    agent = state.initial_state.environment_states[0].agent_states["single_agent"]
    obs, _info = from_proto(agent)
    return obs


class TestSeeding:

    def test_unset_seed_returns_none(self):
        
        assert _seed_from_proto(EnvironmentSettings()) is None

    def test_zero_seed_returns_zero(self):

        assert _seed_from_proto(EnvironmentSettings(seed=0)) == 0, (
            "Explicitly set seed of 0 should return 0"
        )

    def test_nonzero_seed_returns_value(self):

        assert _seed_from_proto(EnvironmentSettings(seed=42)) == 42


class TestAutoreset:

    @pytest.mark.parametrize(
        "autoreset_type",
        [
            AutoResetType.NEXT_STEP,
            AutoResetType.SAME_STEP,
        ],
    )
    def test_autoreset_next_and_same_step(self, autoreset_type):
        servicer = GymToGymServiceServicer(_make_env_factory(max_count=2))
        _start_servicer(servicer, autoreset_type)

        reset_state = _reset(servicer)
        assert _initial_obs_value(reset_state) == 0

        state = _step(servicer, 1)
        obs, reward, terminated, _truncated = _agent_state_values(state)
        assert obs == 1
        assert reward == 1.0
        assert terminated is False

        state = _step(servicer, 1)
        obs, reward, terminated, _truncated = _agent_state_values(state)
        assert obs == 2
        assert reward == 1.0
        assert terminated is True

        if autoreset_type == AutoResetType.NEXT_STEP:
            state = _step(servicer, 1)
            obs, reward, terminated, _truncated = _agent_state_values(state)
            assert obs == 0
            assert reward == 0.0
            assert terminated is False
        else:
            assert _initial_obs_value(state) == 0

            state = _step(servicer, 1)
            obs, reward, terminated, _truncated = _agent_state_values(state)
            assert obs == 1
            assert reward == 1.0
            assert terminated is False

    def test_autoreset_disabled_blocks_step_after_done(self):
        servicer = GymToGymServiceServicer(_make_env_factory(max_count=1))
        _start_servicer(servicer, AutoResetType.DISABLED)
        _reset(servicer)

        _step(servicer, 1)
        with pytest.raises(Exception, match="already terminated or truncated"):
            _step(servicer, 1)

    def test_explicit_reset_clears_completed_flag(self):
        servicer = GymToGymServiceServicer(_make_env_factory(max_count=2))
        _start_servicer(servicer, AutoResetType.DISABLED)
        _reset(servicer)
        _step(servicer, 1)
        _step(servicer, 1)

        _reset(servicer)
        state = _step(servicer, 1)
        obs, _reward, terminated, _truncated = _agent_state_values(state)
        assert obs == 1
        assert terminated is False
