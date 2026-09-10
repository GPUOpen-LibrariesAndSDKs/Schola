# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""LeRobot environment configuration for Schola."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from draccus import decode
from gymnasium.spaces import Box
from gymnasium.spaces.utils import flatten_space
from lerobot.envs.configs import EnvConfig
from lerobot.utils.constants import OBS_IMAGES, OBS_PREFIX
from lerobot_env_schola.feature_mapping import as_source_tuple, infer_features
from schola.scripts.common.settings import (
    GrpcProtocolConfig,
    SingularExecutableSimulatorConfig,
    SingularExternalSimulatorConfig,
    SingularProjectSimulatorConfig,
)

if TYPE_CHECKING:
    import gymnasium as gym

logger = logging.getLogger(__name__)

ScholaObservationSource = str | list[str]


class ScholaObservationConfig(
    dict[str, ScholaObservationSource | dict[str, ScholaObservationSource]]
):
    """LeRobot-shaped observation fields mapped to Schola source paths.

    Top-level fields become ``observation.<field>`` policy features. The
    ``images`` field contains camera-name-to-source entries that become
    ``observation.images.<camera>``. Dots in Schola source paths traverse
    nested ``Dict`` spaces. Every source begins with ``observation``.
    """

    def to_policy_mapping(self) -> dict[str, tuple[str, ...]]:
        """Expand YAML observations into policy keys mapped to source tuples."""
        policy_sources: dict[str, tuple[str, ...]] = {}
        for field_name, configured_sources in self.items():
            if not field_name:
                raise ValueError("Observation field names cannot be empty")
            if field_name.startswith(OBS_PREFIX):
                raise ValueError(
                    f"Observation field {field_name!r} must omit the "
                    f"{OBS_PREFIX!r} prefix"
                )

            if field_name == "images":
                if not isinstance(configured_sources, Mapping):
                    raise TypeError(
                        "observations.images must map camera names to Schola sources"
                    )
                for camera_name, camera_sources in configured_sources.items():
                    if not camera_name:
                        raise ValueError("Observation camera names cannot be empty")
                    policy_key = f"{OBS_IMAGES}.{camera_name}"
                    policy_sources[policy_key] = as_source_tuple(
                        policy_key, camera_sources
                    )
            else:
                if isinstance(configured_sources, Mapping):
                    raise TypeError(
                        f"observations.{field_name} must be a Schola source string "
                        "or ordered list of source strings"
                    )
                policy_key = f"{OBS_PREFIX}{field_name}"
                policy_sources[policy_key] = as_source_tuple(
                    policy_key, configured_sources
                )

        return policy_sources


@decode.register(ScholaObservationConfig)
def _decode_observation_config(
    raw_value: Any, path: Sequence[str]
) -> ScholaObservationConfig:
    """Preserve the mirrored observation structure when Draccus decodes it."""
    del path
    if not isinstance(raw_value, Mapping):
        raise TypeError("observations must be a mapping")
    return ScholaObservationConfig(raw_value)


@dataclass(kw_only=True)
class BaseScholaEnvConfig(EnvConfig):
    """Configure a LeRobot evaluation environment backed by Schola.

    Supports one simulator process, which may expose multiple homogeneous
    environments through Schola's native ``GymVectorEnv``.
    """

    task: str | None = "schola"
    protocol: GrpcProtocolConfig = field(default_factory=GrpcProtocolConfig)
    verbosity: int = 0
    task_description: str | None = None
    episode_length: int = 300
    success_key: str | None = None
    """Optional Schola ``info`` key to expose as LeRobot ``is_success``."""
    observations: ScholaObservationConfig = field(
        default_factory=ScholaObservationConfig
    )
    """LeRobot-shaped observation fields mapped to Schola source paths."""
    render_camera: str | None = None
    render_fps: int = 30

    def __post_init__(self) -> None:
        if self.episode_length < 1:
            raise ValueError("episode_length must be at least 1")
        if self.render_fps < 1:
            raise ValueError("render_fps must be at least 1")

    @property
    def gym_kwargs(self) -> dict[str, Any]:
        """Return no ``gym.make`` arguments because Schola constructs the env."""
        return {}

    def _get_simulator_config(self) -> Any:
        """Return the concrete simulator configuration for this environment type."""
        raise NotImplementedError

    def create_envs(
        self, n_envs: int, use_async_envs: bool = False
    ) -> dict[str, dict[int, gym.vector.VectorEnv]]:
        """Create one Schola vector environment for LeRobot.

        Schola performs vectorization inside the connected simulator, so
        LeRobot must not add another ``AsyncVectorEnv`` layer around it.
        """
        from lerobot_env_schola.vector_env import (
            LeRobotScholaVectorEnv,
            _build_source_spaces,
        )

        if not self.observations:
            raise ValueError(
                "ScholaEnvConfig requires observations to declare how policy "
                "features map to Schola sources."
            )
        if self.features or self.features_map:
            raise ValueError(
                "Schola infers features and features_map from the connected "
                "environment; do not set them in YAML or on the config"
            )

        policy_sources = ScholaObservationConfig(self.observations).to_policy_mapping()
        env = None
        try:
            env = LeRobotScholaVectorEnv(config=self)
            if env.num_envs != n_envs:
                logger.warning(
                    "LeRobot requested %d environment(s), but Schola exposed %d "
                    "homogeneous environment(s); using Schola's native vector size.",
                    n_envs,
                    env.num_envs,
                )
            source_spaces = _build_source_spaces(env.env.single_observation_space)
            action_space = flatten_space(env.env.single_action_space)
            if not isinstance(action_space, Box):
                raise TypeError(
                    "Flattening Schola's action space did not produce a Box"
                )
            self.features, self.features_map = infer_features(
                policy_sources,
                source_spaces,
                action_space,
            )
        except Exception:
            if env is not None:
                env.close()
            raise

        return {self.type: {0: env}}


@EnvConfig.register_subclass("schola")
@dataclass(kw_only=True)
class ScholaEnvConfig(BaseScholaEnvConfig):
    """Connect to an externally managed simulator process."""

    simulator: SingularExternalSimulatorConfig = field(
        default_factory=SingularExternalSimulatorConfig
    )

    def _get_simulator_config(self) -> SingularExternalSimulatorConfig:
        return self.simulator


@EnvConfig.register_subclass("schola-external")
@dataclass(kw_only=True)
class ScholaExternalEnvConfig(ScholaEnvConfig):
    """Connect to an externally managed simulator process."""


@EnvConfig.register_subclass("schola-project")
@dataclass(kw_only=True)
class ScholaProjectEnvConfig(BaseScholaEnvConfig):
    """Build and launch an Unreal project for LeRobot evaluation."""

    simulator: SingularProjectSimulatorConfig = field()

    def _get_simulator_config(self) -> SingularProjectSimulatorConfig:
        return self.simulator


@EnvConfig.register_subclass("schola-executable")
@dataclass(kw_only=True)
class ScholaExecutableEnvConfig(BaseScholaEnvConfig):
    """Launch a packaged Unreal executable for LeRobot evaluation."""

    simulator: SingularExecutableSimulatorConfig = field()

    def _get_simulator_config(self) -> SingularExecutableSimulatorConfig:
        return self.simulator
