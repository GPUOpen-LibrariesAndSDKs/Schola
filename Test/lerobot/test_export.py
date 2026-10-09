# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch as th
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
from lerobot_env_schola.export import LeRobotACTScholaModule, get_spaces


@pytest.fixture
def make_spaces_yaml(tmp_path: Path):
    """Write a spaces YAML under ``tmp_path`` and return its path."""

    def _make(yaml_text: str, name: str = "spaces.yaml") -> Path:
        path = tmp_path / name
        path.write_text(dedent(yaml_text).strip() + "\n", encoding="utf-8")
        return path

    return _make


@pytest.fixture
def make_act_module():
    def _make(
        *,
        state_dim: int,
        action_dim: int,
        chunk_size: int = 4,
        temporal_ensemble_coeff: float | None = None,
        chunk: th.Tensor | None = None,
        observation_space: Dict | None = None,
        action_space: Dict | None = None,
    ) -> tuple[LeRobotACTScholaModule, MagicMock]:
        if observation_space is None:
            observation_space = Dict(
                {"observation.state": Box(-1, 1, shape=(state_dim,), dtype=np.float32)}
            )
        if action_space is None:
            action_space = Dict(
                {"action": Box(-1, 1, shape=(action_dim,), dtype=np.float32)}
            )
        policy = MagicMock()
        policy.config.chunk_size = chunk_size
        policy.config.temporal_ensemble_coeff = temporal_ensemble_coeff
        policy.eval.return_value = policy
        if chunk is not None:
            policy.predict_action_chunk.return_value = chunk
        return (
            LeRobotACTScholaModule(policy, observation_space, action_space),
            policy,
        )

    return _make


@pytest.fixture
def make_ensemble_compare(make_act_module):
    def _make(
        *,
        coeff: float,
        chunk_size: int,
        action_dim: int,
        batch_size: int,
    ) -> tuple[LeRobotACTScholaModule, ACTTemporalEnsembler, th.Tensor, th.Tensor]:
        model, _ = make_act_module(
            state_dim=1,
            action_dim=action_dim,
            chunk_size=chunk_size,
            temporal_ensemble_coeff=coeff,
        )
        leftover_actions = th.zeros(batch_size, chunk_size - 1, action_dim)
        leftover_counts = th.zeros(batch_size, chunk_size - 1, 1, dtype=th.long)
        return (
            model,
            ACTTemporalEnsembler(coeff, chunk_size),
            leftover_actions,
            leftover_counts,
        )

    return _make


def test_get_spaces_loads_nested_observation_leaves(make_spaces_yaml):
    path = make_spaces_yaml("""
        observation:
          state:
            type: box
            shape: [7]
            dtype: float32
          images:
            wrist:
              type: box
              shape: [3, 128, 128]
              dtype: uint8
            front:
              type: box
              shape: [3, 128, 128]
              dtype: float32
          mode:
            type: discrete
            shape: [4]
          tools:
            type: multi_discrete
            shape: [2, 5, 3]
          flags:
            type: multi_binary
            shape: [8]
        action:
          type: box
          shape: [7]
          dtype: float32
        """)
    observation_space, action_space = get_spaces(str(path))

    assert isinstance(observation_space, Dict)
    assert set(observation_space.spaces) == {
        "observation.state",
        "observation.images.wrist",
        "observation.images.front",
        "observation.mode",
        "observation.tools",
        "observation.flags",
    }

    state = observation_space["observation.state"]
    assert isinstance(state, Box)
    assert state.shape == (7,)
    assert state.dtype == np.float32

    wrist = observation_space["observation.images.wrist"]
    assert isinstance(wrist, Box)
    assert wrist.shape == (3, 128, 128)
    assert wrist.dtype == np.uint8

    assert isinstance(observation_space["observation.mode"], Discrete)
    assert observation_space["observation.mode"].n == 4
    assert isinstance(observation_space["observation.tools"], MultiDiscrete)
    np.testing.assert_array_equal(
        observation_space["observation.tools"].nvec, [2, 5, 3]
    )
    assert isinstance(observation_space["observation.flags"], MultiBinary)
    assert observation_space["observation.flags"].shape == (8,)

    assert list(action_space.spaces) == ["action"]
    action = action_space["action"]
    assert isinstance(action, Box)
    assert action.shape == (7,)
    assert action.dtype == np.float32

    observation_space.sample()
    action_space.sample()


def test_get_spaces_nested_action_uses_dotted_keys(make_spaces_yaml):
    path = make_spaces_yaml("""
        observation:
          state:
            type: box
            shape: [2]
            dtype: float32
        action:
          arm:
            type: box
            shape: [6]
            dtype: float32
          gripper:
            type: box
            shape: [1]
            dtype: float32
        """)
    observation_space, action_space = get_spaces(str(path))
    assert list(observation_space.spaces) == ["observation.state"]
    assert isinstance(action_space, Dict)
    assert set(action_space.spaces) == {"action.arm", "action.gripper"}


def test_get_spaces_rejects_unknown_space_type(make_spaces_yaml):
    path = make_spaces_yaml("""
        observation:
          state:
            type: tuple
            shape: [3]
        action:
          type: box
          shape: [1]
          dtype: float32
        """)
    with pytest.raises(ValueError, match="unsupported space type"):
        get_spaces(str(path))


def test_temporal_ensemble_step_matches_lerobot(make_ensemble_compare):
    coeff = 0.01
    chunk_size, action_dim, batch_size = 5, 3, 2
    model, ref, ensembled_actions, ensembled_counts = make_ensemble_compare(
        coeff=coeff,
        chunk_size=chunk_size,
        action_dim=action_dim,
        batch_size=batch_size,
    )

    for _ in range(8):
        chunk = th.randn(batch_size, chunk_size, action_dim)
        expected = ref.update(chunk.clone())
        action, ensembled_actions, ensembled_counts = model._temporal_ensemble_step(
            chunk, ensembled_actions, ensembled_counts
        )
        th.testing.assert_close(action, expected)
        th.testing.assert_close(ensembled_actions, ref.ensembled_actions)
        th.testing.assert_close(ensembled_counts[0], ref.ensembled_actions_count)
        th.testing.assert_close(ensembled_counts[0], ensembled_counts[1])


def test_act_module_is_stateless_without_ensemble(make_act_module):
    chunk = th.zeros(2, 4, 3)
    model, policy = make_act_module(state_dim=2, action_dim=3, chunk=chunk)
    assert not model.is_stateful
    (out,) = model.forward(th.zeros(2, 2))
    assert out.shape == (2, 3)
    th.testing.assert_close(out, chunk[:, 0])
    batch = policy.predict_action_chunk.call_args.args[0]
    assert list(batch) == ["observation.state"]


def test_act_module_ensemble_returns_action_and_state(make_act_module):
    chunk = th.arange(24, dtype=th.float32).reshape(2, 4, 3)
    model, _ = make_act_module(
        state_dim=2,
        action_dim=3,
        chunk_size=4,
        temporal_ensemble_coeff=0.01,
        chunk=chunk,
    )
    assert model.is_stateful
    assert set(model.input_state_keys) == {
        "state_in_ensembled_actions",
        "state_in_ensembled_actions_count",
    }
    assert set(model.output_state_keys) == {
        "state_out_ensembled_actions",
        "state_out_ensembled_actions_count",
    }

    obs = th.zeros(2, 2)
    state_actions = model.input_state_dict["state_in_ensembled_actions"].repeat(2, 1, 1)
    state_counts = model.input_state_dict["state_in_ensembled_actions_count"].repeat(
        2, 1, 1
    )
    action, new_actions, new_counts = model.forward(obs, state_actions, state_counts)
    assert action.shape == (2, 3)
    assert new_actions.shape == (2, 3, 3)
    assert new_counts.shape == (2, 3, 1)
    th.testing.assert_close(action, chunk[:, 0])
    th.testing.assert_close(new_actions, chunk[:, 1:])
