# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""LeRobot environment configuration for Schola."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from draccus import decode
from gymnasium.spaces import Box, Dict
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.utils.constants import (
    ACTION,
    OBS_ENV_STATE,
    OBS_IMAGE,
    OBS_IMAGES,
    OBS_PREFIX,
    OBS_STATE,
)
from schola.scripts.common.settings import (
    ExternalSimulatorConfig,
    GrpcProtocolConfig,
    UnrealExecutableSimulatorConfig,
    UnrealProjectSimulatorConfig,
)

if TYPE_CHECKING:
    import gymnasium as gym

logger = logging.getLogger(__name__)

SINGLE_IMAGE_NDIMS = 3
HWC_CHANNEL_DIM = -1
SUPPORTED_IMAGE_CHANNELS = (1, 3, 4)

_GYM_VALUE_POLICY_FEATURES = {
    "agent_pos": OBS_STATE,
    "environment_state": OBS_ENV_STATE,
}


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

    def to_policy_mapping(self) -> dict[str, ScholaObservationSource]:
        """Expand the mirrored configuration into canonical policy feature keys."""
        mapping: dict[str, ScholaObservationSource] = {}
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
                    mapping[f"{OBS_IMAGES}.{camera_name}"] = camera_sources
                continue

            if isinstance(configured_sources, Mapping):
                raise TypeError(
                    f"observations.{field_name} must be a Schola source string "
                    "or ordered list of source strings"
                )
            mapping[f"{OBS_PREFIX}{field_name}"] = configured_sources

        return mapping


@decode.register(ScholaObservationConfig)
def _decode_observation_config(
    raw_value: Any, path: Sequence[str]
) -> ScholaObservationConfig:
    """Preserve the mirrored observation structure when Draccus decodes it."""
    del path
    if not isinstance(raw_value, Mapping):
        raise TypeError("observations must be a mapping")
    return ScholaObservationConfig(raw_value)


def infer_features_from_spaces(
    observation_space: Dict,
    action_space: Box,
) -> tuple[dict[str, PolicyFeature], dict[str, str]]:
    """Infer LeRobot features from an environment's single-environment spaces."""
    features: dict[str, PolicyFeature] = {}
    features_map: dict[str, str] = {}

    for key, space in observation_space.spaces.items():
        if isinstance(space, Dict):
            for camera_name, camera_space in space.spaces.items():
                if (
                    not isinstance(camera_space, Box)
                    or camera_space.dtype.name != "uint8"
                    or len(camera_space.shape) != SINGLE_IMAGE_NDIMS
                    or camera_space.shape[HWC_CHANNEL_DIM]
                    not in SUPPORTED_IMAGE_CHANNELS
                ):
                    raise TypeError(
                        f"Nested observation {key!r}/{camera_name!r} must be a "
                        "channel-last, three-dimensional uint8 Box"
                    )
                feature_key = f"{key}/{camera_name}"
                features[feature_key] = PolicyFeature(
                    type=FeatureType.VISUAL, shape=camera_space.shape
                )
                features_map[feature_key] = f"{OBS_IMAGES}.{camera_name}"
            continue

        if not isinstance(space, Box):
            raise TypeError(
                f"Observation {key!r} uses unsupported space {type(space).__name__}; "
                "LeRobot feature inference supports Box observations only."
            )
        if not space.shape:
            raise ValueError(f"Observation {key!r} must have at least one dimension")
        if (
            space.dtype.name == "uint8"
            and len(space.shape) == SINGLE_IMAGE_NDIMS
            and space.shape[HWC_CHANNEL_DIM] in SUPPORTED_IMAGE_CHANNELS
        ):
            features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=space.shape)
            features_map[key] = OBS_IMAGE
            continue

        feature_type = (
            FeatureType.ENV if key == "environment_state" else FeatureType.STATE
        )
        features[key] = PolicyFeature(type=feature_type, shape=space.shape)
        features_map[key] = _GYM_VALUE_POLICY_FEATURES.get(key, f"{OBS_PREFIX}{key}")

    if not action_space.shape:
        raise ValueError(
            "The flattened Schola action space must have at least one dimension"
        )
    features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=action_space.shape)
    features_map[ACTION] = ACTION

    for feature_key, policy_key in features_map.items():
        feature = features[feature_key]
        logger.info(
            "Inferred LeRobot policy feature %s (type=%s, shape=%s)",
            policy_key,
            feature.type.value,
            feature.shape,
        )

    return features, features_map


@dataclass(kw_only=True)
class BaseScholaEnvConfig(EnvConfig):
    """Configure a LeRobot evaluation environment backed by Schola.

    Supports one simulator process, which may expose multiple homogeneous
    agent slots through Schola's native ``GymVectorEnv``.
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
        from lerobot_env_schola.vector_env import LeRobotScholaVectorEnv
        from schola.gym.env import GymVectorEnv

        if n_envs < 1:
            raise ValueError("n_envs must be at least 1")
        if use_async_envs:
            raise ValueError(
                "Schola manages vectorization through GymVectorEnv; "
                "LeRobot async environment wrapping is not supported."
            )
        if not self.observations:
            raise ValueError(
                "ScholaEnvConfig requires observations to declare how policy "
                "features map to Schola sources."
            )
        simulator_config = self._get_simulator_config()
        if simulator_config.num_simulators != 1:
            raise ValueError(
                "ScholaEnvConfig currently supports one simulator process; "
                f"got num_simulators={simulator_config.num_simulators}."
            )

        schola_env = GymVectorEnv(
            simulator=simulator_config.make(),
            protocol=self.protocol.make(),
            verbosity=self.verbosity,
        )

        if schola_env.num_envs != n_envs:
            logger.warning(
                "LeRobot requested %d environment(s), but Schola exposed %d "
                "homogeneous agent slot(s); using Schola's native vector size.",
                n_envs,
                schola_env.num_envs,
            )

        try:
            observation_mapping = ScholaObservationConfig(
                self.observations
            ).to_policy_mapping()
            env = LeRobotScholaVectorEnv(
                schola_env,
                task=self.task or "schola",
                task_description=self.task_description or self.task or "schola",
                max_episode_steps=self.episode_length,
                success_key=self.success_key,
                observation_config=ScholaObservationConfig(observation_mapping),
                render_camera=self.render_camera,
                render_fps=self.render_fps,
            )
        except Exception:
            schola_env.close()
            raise

        try:
            if bool(self.features) != bool(self.features_map):
                raise ValueError(
                    "features and features_map must either both be provided or both be empty"
                )
            if self.features:
                if self.features.keys() != self.features_map.keys():
                    raise ValueError(
                        "features and features_map must contain the same keys"
                    )
            else:
                self.features, self.features_map = infer_features_from_spaces(
                    env.single_observation_space,
                    env.single_action_space,
                )
        except Exception:
            env.close()
            raise

        return {self.type: {0: env}}


@EnvConfig.register_subclass("schola")
@dataclass(kw_only=True)
class ScholaEnvConfig(BaseScholaEnvConfig):
    """Connect to an externally managed simulator process."""

    simulator: ExternalSimulatorConfig = field(default_factory=ExternalSimulatorConfig)

    def _get_simulator_config(self) -> ExternalSimulatorConfig:
        return self.simulator


@EnvConfig.register_subclass("schola-external")
@dataclass(kw_only=True)
class ScholaExternalEnvConfig(ScholaEnvConfig):
    """Connect to an externally managed simulator process."""


@EnvConfig.register_subclass("schola-project")
@dataclass(kw_only=True)
class ScholaProjectEnvConfig(BaseScholaEnvConfig):
    """Build and launch an Unreal project for LeRobot evaluation."""

    simulator: UnrealProjectSimulatorConfig = field()

    def _get_simulator_config(self) -> UnrealProjectSimulatorConfig:
        return self.simulator


@EnvConfig.register_subclass("schola-executable")
@dataclass(kw_only=True)
class ScholaExecutableEnvConfig(BaseScholaEnvConfig):
    """Launch a packaged Unreal executable for LeRobot evaluation."""

    simulator: UnrealExecutableSimulatorConfig = field()

    def _get_simulator_config(self) -> UnrealExecutableSimulatorConfig:
        return self.simulator
