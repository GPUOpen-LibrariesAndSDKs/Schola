"""Schola environment plugin for LeRobot."""

from .config import ScholaEnvConfig
from .vector_env import LeRobotScholaVectorEnv

__all__ = ["LeRobotScholaVectorEnv", "ScholaEnvConfig"]
