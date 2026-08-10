"""Observation adaptation from Schola spaces to LeRobot conventions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict

from lerobot_env_schola.config import ScholaObservationConfig

SINGLE_IMAGE_NDIM = 3
BATCHED_IMAGE_NDIM = 4
CHW_CHANNEL_AXIS = -3
HWC_CHANNEL_AXIS = -1
SUPPORTED_IMAGE_CHANNELS = (1, 3, 4)
UINT8_MAX = np.iinfo(np.uint8).max
BATCH_AXIS = 0


def _convert_image_space(space: gym.Space, name: str) -> tuple[Box, str]:
    if not isinstance(space, Box) or len(space.shape) != SINGLE_IMAGE_NDIM:
        raise TypeError(f"Image observation {name!r} must be a three-dimensional Box")

    is_float = np.issubdtype(space.dtype, np.floating)
    is_uint8 = space.dtype == np.dtype(np.uint8)
    if not (is_float or is_uint8):
        raise TypeError(f"Image observation {name!r} must use float or uint8 values")
    if is_float and not (np.all(space.low >= 0) and np.all(space.high <= 1)):
        raise ValueError(
            f"Floating-point image observation {name!r} must be bounded "
            "within [0, 1]"
        )

    # Prefer channel-last when both edge dimensions look like channel counts.
    # This matches the normalized output layout and avoids treating a small
    # image height as channels (for example, an HWC shape of (4, 5, 3)).
    if space.shape[HWC_CHANNEL_AXIS] in SUPPORTED_IMAGE_CHANNELS:
        height, width, channels = space.shape
        layout = "hwc"
    elif space.shape[CHW_CHANNEL_AXIS] in SUPPORTED_IMAGE_CHANNELS:
        channels, height, width = space.shape
        layout = "chw"
    else:
        raise ValueError(
            f"Image observation {name!r} must have 1, 3, or 4 channels; "
            f"got shape {space.shape}"
        )

    mode = f"{layout}_{'float' if is_float else 'uint8'}"
    return (
        Box(0, UINT8_MAX, shape=(height, width, channels), dtype=np.uint8),
        mode,
    )


def _convert_image_value(value: Any, mode: str) -> np.ndarray:
    image = np.asarray(value)
    if mode.startswith("chw"):
        if image.ndim not in (SINGLE_IMAGE_NDIM, BATCHED_IMAGE_NDIM):
            raise ValueError(f"Expected a CHW image batch, got shape {image.shape}")
        image = np.moveaxis(image, CHW_CHANNEL_AXIS, HWC_CHANNEL_AXIS)
    if mode.endswith("float"):
        image = np.rint(np.clip(image, 0, 1) * UINT8_MAX).astype(np.uint8)
    return np.ascontiguousarray(image)


class ObservationAdapter:
    """Validate, declare, and convert Schola observations for LeRobot."""

    def __init__(
        self,
        source_space: gym.Space,
        config: ScholaObservationConfig,
        num_envs: int,
    ) -> None:
        self.config = config
        self.num_envs = num_envs
        self._image_modes: dict[str, str] = {}
        self._vector_dtypes: dict[str, np.dtype] = {}

        self._validate_config(source_space)
        self.single_observation_space = self._make_observation_space(source_space)

    def _validate_config(self, space: gym.Space) -> None:
        if not isinstance(space, Dict):
            raise TypeError(
                "Schola must expose a Dict observation space so observations "
                "can be grouped for LeRobot."
            )

        duplicate_outputs = set(self.config.vectors) & self.config.passthrough.keys()
        if duplicate_outputs:
            raise ValueError(
                "Vector and passthrough outputs overlap: "
                f"{sorted(duplicate_outputs)}"
            )

        reserved_outputs = {
            key
            for key in (*self.config.vectors, *self.config.passthrough)
            if key == "pixels" or key.startswith("pixels/")
        }
        if reserved_outputs:
            raise ValueError(
                "'pixels' outputs must be declared under cameras, not vectors "
                f"or passthrough: {sorted(reserved_outputs)}"
            )

        owners: dict[str, str] = {}

        def claim_source(source_key: str, owner: str) -> None:
            if source_key not in space.spaces:
                raise ValueError(
                    f"{owner} references unknown Schola observation " f"{source_key!r}"
                )
            if source_key in owners:
                raise ValueError(
                    f"Schola observation {source_key!r} is used by both "
                    f"{owners[source_key]} and {owner}"
                )
            owners[source_key] = owner

        for camera_name, source_key in self.config.cameras.items():
            if not camera_name:
                raise ValueError("Camera names cannot be empty")
            claim_source(source_key, f"camera {camera_name!r}")

        for target_key, source_keys in self.config.vectors.items():
            if not source_keys:
                raise ValueError(
                    f"Vector output {target_key!r} requires at least one source"
                )
            for source_key in source_keys:
                claim_source(source_key, f"vector {target_key!r}")
                if not isinstance(space.spaces[source_key], Box):
                    raise TypeError(
                        f"Vector source {source_key!r} must use a Box space"
                    )

        for target_key, source_key in self.config.passthrough.items():
            claim_source(source_key, f"passthrough {target_key!r}")
            if not isinstance(space.spaces[source_key], Box):
                raise TypeError(
                    f"Passthrough source {source_key!r} must use a Box space"
                )

        for source_key in self.config.ignore:
            claim_source(source_key, "ignore")

        missing_keys = space.spaces.keys() - owners.keys()
        if missing_keys:
            raise ValueError(
                "Observation configuration does not account for Schola keys: "
                f"{sorted(missing_keys)}"
            )

    def _make_observation_space(self, space: gym.Space) -> Dict:
        if not isinstance(space, Dict):
            raise TypeError(
                "Expected the validated Schola observation space to be a Dict"
            )

        output_spaces: dict[str, gym.Space] = {}
        if self.config.cameras:
            camera_spaces: dict[str, gym.Space] = {}
            for camera_name, source_key in self.config.cameras.items():
                camera_spaces[camera_name], self._image_modes[source_key] = (
                    _convert_image_space(space.spaces[source_key], source_key)
                )
            output_spaces["pixels"] = Dict(camera_spaces)

        for target_key, source_keys in self.config.vectors.items():
            source_spaces = [space.spaces[source_key] for source_key in source_keys]
            dtype = np.result_type(
                np.float32,
                *(source_space.dtype for source_space in source_spaces),
            )
            low = np.concatenate(
                [
                    np.asarray(source_space.low).reshape(-1)
                    for source_space in source_spaces
                ]
            ).astype(dtype)
            high = np.concatenate(
                [
                    np.asarray(source_space.high).reshape(-1)
                    for source_space in source_spaces
                ]
            ).astype(dtype)
            output_spaces[target_key] = Box(
                low=low,
                high=high,
                dtype=dtype,
            )
            self._vector_dtypes[target_key] = np.dtype(dtype)

        for target_key, source_key in self.config.passthrough.items():
            output_spaces[target_key] = space.spaces[source_key]

        return Dict(output_spaces)

    def convert(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Convert one batched Schola observation."""
        converted: dict[str, Any] = {}

        if self.config.cameras:
            converted["pixels"] = {
                camera_name: _convert_image_value(
                    observation[source_key],
                    self._image_modes[source_key],
                )
                for camera_name, source_key in self.config.cameras.items()
            }

        for target_key, source_keys in self.config.vectors.items():
            values = [
                np.asarray(observation[source_key]).reshape(self.num_envs, -1)
                for source_key in source_keys
            ]
            converted[target_key] = np.concatenate(values, axis=-1).astype(
                self._vector_dtypes[target_key],
                copy=False,
            )

        for target_key, source_key in self.config.passthrough.items():
            converted[target_key] = observation[source_key]

        return converted
