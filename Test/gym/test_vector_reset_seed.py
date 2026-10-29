# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""GymVectorEnv.reset must forward non-negative seeds."""

import numpy as np
from gymnasium.spaces import Box

from schola.core.simulators.base_simulator import BaseSimulator
from schola.gym.env import GymVectorEnv


class _RecordingProtocol:
    """Stand-in protocol that records seeds passed to ``send_reset_msg``."""

    def __init__(self) -> None:
        self.reset_seeds: list[int] | None = None

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def send_startup_msg(self, *args: object, **kwargs: object) -> None:
        pass

    @property
    def properties(self) -> dict[str, object]:
        return {}

    def get_definition(self):
        space = Box(-1, 1, shape=(1,), dtype=np.float32)
        return (
            [["agent"]],
            [{}],
            {0: {"agent": space}},
            {0: {"agent": space}},
        )

    def send_reset_msg(self, seeds=None, options=None):
        self.reset_seeds = seeds
        obs = [{"agent": np.zeros(1, dtype=np.float32)}]
        info = [{"agent": {}}]
        return obs, info


class _FakeSimulator(BaseSimulator):
    """Simulator that accepts exactly one protocol class and never starts Unreal."""

    def __init__(self, protocol_type: type) -> None:
        self._protocol_type = protocol_type

    @property
    def supported_protocols(self):
        return (self._protocol_type,)

    def start(self, _protocol_properties: dict[str, object]) -> None:
        pass

    def stop(self) -> None:
        pass


def test_reset_int_seed_stays_non_negative():
    """``reset(seed=2)``generates a large number that can overflow np.uint16 and cause 
    a negative seed to be passed to a sub-environment.
    """
    protocol = _RecordingProtocol()
    simulator = _FakeSimulator(type(protocol))
    env = GymVectorEnv(simulator, protocol)
    try:
        env.reset(seed=2)
        seeds = protocol.reset_seeds
        assert seeds is not None
        assert all(seed >= 0 for seed in seeds), seeds
    finally:
        env.close()
