"""Schola environment plugin for LeRobot."""

from .config import ScholaEnvConfig, ScholaObservationConfig
from .observations import ObservationAdapter
from .vector_env import LeRobotScholaVectorEnv

__all__ = [
    "LeRobotScholaVectorEnv",
    "ObservationAdapter",
    "ScholaEnvConfig",
    "ScholaObservationConfig",
]
