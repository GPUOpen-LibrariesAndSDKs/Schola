# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

import numpy as np
import pytest
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete

from schola.core.model_space_config import (
    load_model_spaces,
    load_policy_model_spaces,
)


def test_load_model_spaces_loads_nested_observation_leaves(make_spaces_yaml):
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
    observation_space, action_space = load_model_spaces(path)

    assert isinstance(observation_space, Dict)
    assert set(observation_space.spaces) == {
        "state",
        "images.wrist",
        "images.front",
        "mode",
        "tools",
        "flags",
    }

    state = observation_space["state"]
    assert isinstance(state, Box)
    assert state.shape == (7,)
    assert state.dtype == np.float32

    wrist = observation_space["images.wrist"]
    assert isinstance(wrist, Box)
    assert wrist.shape == (3, 128, 128)
    assert wrist.dtype == np.uint8

    assert isinstance(observation_space["mode"], Discrete)
    assert observation_space["mode"].n == 4
    assert isinstance(observation_space["tools"], MultiDiscrete)
    np.testing.assert_array_equal(observation_space["tools"].nvec, [2, 5, 3])
    assert isinstance(observation_space["flags"], MultiBinary)
    assert observation_space["flags"].shape == (8,)

    assert list(action_space.spaces) == ["action"]
    action = action_space["action"]
    assert isinstance(action, Box)
    assert action.shape == (7,)
    assert action.dtype == np.float32

    observation_space.sample()
    action_space.sample()


def test_load_model_spaces_nested_action_uses_dotted_keys(make_spaces_yaml):
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
    observation_space, action_space = load_model_spaces(path)
    assert list(observation_space.spaces) == ["state"]
    assert isinstance(action_space, Dict)
    assert set(action_space.spaces) == {"arm", "gripper"}


def test_load_model_spaces_flattens_deep_implicit_dicts(make_spaces_yaml):
    path = make_spaces_yaml("""
        observation:
          robot:
            joints:
              type: box
              shape: [6]
              dtype: float32
        action:
          type: box
          shape: [6]
          dtype: float32
        """)
    observation_space, _ = load_model_spaces(path)
    assert list(observation_space.spaces) == ["robot.joints"]
    assert observation_space["robot.joints"].shape == (6,)


def test_load_model_spaces_names_root_leaves_obs_and_action(make_spaces_yaml):
    path = make_spaces_yaml("""
        observation:
          type: box
          shape: [2]
          dtype: float32
        action:
          type: box
          shape: [1]
          dtype: float32
        """)

    observation_space, action_space = load_model_spaces(path)

    assert list(observation_space.spaces) == ["obs"]
    assert list(action_space.spaces) == ["action"]


def test_load_model_spaces_rejects_unknown_space_type(make_spaces_yaml):
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
        load_model_spaces(path)


def test_load_model_spaces_rejects_policy_keyed_config(make_spaces_yaml):
    path = make_spaces_yaml("""
        attacker:
          observation:
            type: box
            shape: [8]
            dtype: float32
          action:
            type: discrete
            shape: [4]
        """)
    with pytest.raises(ValueError, match="missing top-level"):
        load_model_spaces(path)


def test_load_policy_model_spaces_shares_unkeyed_spaces(make_spaces_yaml):
    path = make_spaces_yaml("""
        observation:
          type: box
          shape: [2]
          dtype: float32
        action:
          type: discrete
          shape: [3]
        """)

    observation_space, action_space = load_policy_model_spaces(path)

    assert observation_space == load_model_spaces(path)[0]
    assert action_space == load_model_spaces(path)[1]


def test_load_policy_model_spaces_keys_spaces_by_policy_id(make_spaces_yaml):
    path = make_spaces_yaml("""
        attacker:
          observation:
            type: box
            shape: [8]
            dtype: float32
          action:
            type: discrete
            shape: [4]
        defender:
          observation:
            type: box
            shape: [12]
            dtype: float32
          action:
            type: box
            shape: [2]
            dtype: float32
        """)

    observation_spaces, action_spaces = load_policy_model_spaces(path)

    assert observation_spaces["attacker"]["obs"].shape == (8,)
    assert observation_spaces["defender"]["obs"].shape == (12,)
    assert action_spaces["attacker"]["action"] == Discrete(4)
    assert action_spaces["defender"]["action"].shape == (2,)
