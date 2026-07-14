# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Unit tests for ``schola.scripts.env.utils``."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np

from schola.core.utils.id_manager import IdManager
from schola.scripts.env.utils import (
    _format_value_for_log,
    inspect_reset,
    log_environment_definition,
)


def test_format_value_for_log_truncates_large_arrays():
    value = np.zeros((32, 32), dtype=np.float32)
    text = _format_value_for_log(value)
    assert "shape=(32, 32)" in text
    assert "preview(first 8)=" in text


def test_format_value_for_log_truncates_large_dicts():
    value = {f"key_{i}": i for i in range(12)}
    text = _format_value_for_log(value)
    assert text.endswith(" ...}")
    assert "key_0=" in text
    assert "key_7=" in text
    assert "key_8=" not in text


def test_log_environment_definition(caplog):
    logger = logging.getLogger("test.env.utils.definition")
    env = MagicMock()
    env.id_manager = IdManager([["agent_0"]], [{"agent_0": "Pawn"}])
    env.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))
    env.action_space = gym.spaces.Discrete(2)
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space

    with caplog.at_level(logging.INFO, logger="test.env.utils.definition"):
        log_environment_definition(env, logger=logger)

    assert "Agent definitions:" in caplog.text
    assert "Total agent slots: 1" in caplog.text


def test_inspect_reset_gym_vector_env(caplog):
    logger = logging.getLogger("test.env.utils.reset")
    env = MagicMock()
    env.id_manager = IdManager([["agent_0"]])
    env.single_observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
    )
    env.observation_space = gym.vector.utils.batch_space(
        env.single_observation_space, n=1
    )
    env.single_action_space = gym.spaces.Discrete(2)
    env.reset.return_value = (
        np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
        {},
    )

    with caplog.at_level(logging.INFO, logger="test.env.utils.reset"):
        inspect_reset(env, logger=logger)

    env.reset.assert_called_once()
    assert "Observation Space:" in caplog.text
    assert "Action Space:" in caplog.text
    assert "Initial Obs:" in caplog.text
    assert "Initial Obs in Space: True" in caplog.text
