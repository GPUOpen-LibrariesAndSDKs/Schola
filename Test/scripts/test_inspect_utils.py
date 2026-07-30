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
    inspect_agents,
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


def test_inspect_agents_gym_vector_env(caplog):
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
        inspect_agents(env, logger=logger)

    env.reset.assert_called_once()
    assert "Observation Space:" in caplog.text
    assert "Action Space:" in caplog.text
    assert "Initial Obs:" in caplog.text
    assert "Initial Obs in Space: True" in caplog.text
