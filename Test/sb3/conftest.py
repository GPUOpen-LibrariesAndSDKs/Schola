# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Shared pytest fixtures for Stable-Baselines3 + Schola integration tests."""

from unittest.mock import MagicMock

import pytest

from schola.core.protocols.base_protocol import BaseRLProtocol
from schola.core.simulators.base_simulator import BaseSimulator


@pytest.fixture(scope="function")
def mock_protocol_and_simulator():
    """Build a mock protocol + simulator pair that pass the ``supported_protocols``
    isinstance check in ``VecEnv.__init__``.

    ``MagicMock(spec=BaseRLProtocol)`` makes ``isinstance(protocol, BaseRLProtocol)``
    return ``True``. ``simulator.supported_protocols`` is overridden to a real
    tuple so the isinstance check uses a concrete class object. All other
    lifecycle calls (``protocol.start()``, ``simulator.start(...)``,
    ``protocol.send_startup_msg(...)``) auto-resolve as MagicMock no-ops, which
    is what lets tests drive the real ``VecEnv.__init__`` end-to-end.
    """
    protocol = MagicMock(spec=BaseRLProtocol)
    simulator = MagicMock(spec=BaseSimulator)
    simulator.supported_protocols = (BaseRLProtocol,)
    return protocol, simulator
