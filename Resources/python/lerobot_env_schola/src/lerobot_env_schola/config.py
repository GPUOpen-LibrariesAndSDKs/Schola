"""LeRobot environment configuration for Schola."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from gymnasium.spaces import Box, Dict
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE, OBS_STR
from schola.scripts.common.settings import ExternalSimulatorConfig, GrpcProtocolConfig

if TYPE_CHECKING:
    import gymnasium as gym

logger = logging.getLogger(__name__)


@dataclass
class ScholaObservationConfig:
    """Describe how Schola observations form LeRobot policy inputs.

    Mapping values are Schola observation keys. Vector source order is
    preserved when values are flattened and concatenated.
    """

    cameras: dict[str, str] = field(default_factory=dict)
    """LeRobot camera name to Schola image observation key."""

    vectors: dict[str, list[str]] = field(default_factory=dict)
    """LeRobot vector name to ordered Schola observation keys."""

    passthrough: dict[str, str] = field(default_factory=dict)
    """LeRobot output name to an unchanged Schola observation key."""

    ignore: list[str] = field(default_factory=list)
    """Schola observation keys intentionally omitted from LeRobot."""

    def is_empty(self) -> bool:
        """Return whether no observation behavior has been configured."""
        return not (self.cameras or self.vectors or self.passthrough or self.ignore)


def infer_features_from_spaces(
    observation_space: Dict,
    action_space: Box,
) -> tuple[dict[str, PolicyFeature], dict[str, str]]:
    """Infer LeRobot features from the wrapper's single-environment spaces."""
    features: dict[str, PolicyFeature] = {}
    features_map: dict[str, str] = {}

    for key, space in observation_space.spaces.items():
        if isinstance(space, Dict):
            for camera_name, camera_space in space.spaces.items():
                if (
                    not isinstance(camera_space, Box)
                    or camera_space.dtype.name != "uint8"
                    or len(camera_space.shape) != 3
                    or camera_space.shape[-1] not in (1, 3, 4)
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
            and len(space.shape) == 3
            and space.shape[-1] in (1, 3, 4)
        ):
            features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=space.shape)
            features_map[key] = OBS_IMAGE
            continue

        feature_type = FeatureType.ENV if key == "environment_state" else FeatureType.STATE
        features[key] = PolicyFeature(type=feature_type, shape=space.shape)
        if key == "agent_pos":
            features_map[key] = OBS_STATE
        elif key == "environment_state":
            features_map[key] = OBS_ENV_STATE
        else:
            features_map[key] = f"{OBS_STR}.{key}"

    if not action_space.shape:
        raise ValueError("The flattened Schola action space must have at least one dimension")
    features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=action_space.shape)
    features_map[ACTION] = ACTION

    for feature_key, policy_key in features_map.items():
        feature = features[feature_key]
        logger.info(
            "Inferred Schola feature mapping: %s -> %s (type=%s, shape=%s)",
            feature_key,
            policy_key,
            feature.type.value,
            feature.shape,
        )

    return features, features_map


@EnvConfig.register_subclass("schola")
@dataclass
class ScholaEnvConfig(EnvConfig):
    """Configure a LeRobot evaluation environment backed by Schola.

    The first integration slice supports a single externally managed Unreal
    process. That process may expose multiple homogeneous agent slots through
    Schola's native ``GymVectorEnv``.
    """

    task: str | None = "schola"
    simulator: ExternalSimulatorConfig = field(default_factory=ExternalSimulatorConfig)
    protocol: GrpcProtocolConfig = field(default_factory=GrpcProtocolConfig)
    verbosity: int = 0
    task_description: str | None = None
    episode_length: int = 300
    success_key: str | None = None
    """Optional Schola ``info`` key to expose as LeRobot ``is_success``."""
    observations: ScholaObservationConfig = field(
        default_factory=ScholaObservationConfig
    )
    """Target-oriented camera, vector, passthrough, and ignore configuration."""
    render_camera: str | None = None
    render_fps: int = 30

    @property
    def gym_kwargs(self) -> dict[str, Any]:
        """Return no ``gym.make`` arguments because Schola constructs the env."""
        return {}

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
        if self.observations.is_empty():
            raise ValueError(
                "ScholaEnvConfig requires observations to declare how Schola "
                "observations map to LeRobot."
            )
        if self.simulator.num_simulators != 1:
            raise ValueError(
                "ScholaEnvConfig currently supports one simulator process; "
                f"got num_simulators={self.simulator.num_simulators}."
            )

        schola_env = GymVectorEnv(
            simulator=self.simulator.make(),
            protocol=self.protocol.make(),
            verbosity=self.verbosity,
        )

        if schola_env.num_envs != n_envs:
            actual_num_envs = schola_env.num_envs
            schola_env.close()
            raise ValueError(
                "LeRobot requested "
                f"{n_envs} environment(s), but Schola exposed {actual_num_envs} "
                "homogeneous agent slot(s). Set --eval.batch_size to match the "
                "number exposed by Unreal."
            )

        try:
            env = LeRobotScholaVectorEnv(
                schola_env,
                task=self.task or "schola",
                task_description=self.task_description or self.task or "schola",
                max_episode_steps=self.episode_length,
                success_key=self.success_key,
                observation_config=self.observations,
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
                    raise ValueError("features and features_map must contain the same keys")
            else:
                self.features, self.features_map = infer_features_from_spaces(
                    env.single_observation_space,
                    env.single_action_space,
                )
        except Exception:
            env.close()
            raise

        return {self.type: {0: env}}
