# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Gymnasium vector adapter between Schola and LeRobot."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Tuple
from gymnasium.spaces.utils import flatten_space, unflatten
from gymnasium.vector.utils import batch_space, concatenate, create_empty_array
from lerobot.envs.utils import NEW_ROLLOUT_OPTION
from lerobot_env_schola.config import (
    HWC_CHANNEL_DIM,
    SINGLE_IMAGE_NDIMS,
    SUPPORTED_IMAGE_CHANNELS,
    ScholaObservationConfig,
)

BATCHED_IMAGE_NDIMS = 4
CHW_CHANNEL_DIM = -3
UINT8_MAX = np.iinfo(np.uint8).max
BATCH_DIM = 0


def _contains_only_boxes(space: gym.Space) -> bool:
    if isinstance(space, Box):
        return True
    if isinstance(space, Dict):
        return all(_contains_only_boxes(child) for child in space.spaces.values())
    return False


def _coerce_success(value: Any) -> Any:
    """Parse Schola ``info`` success flags for LeRobot eval (UE convention: ``true`` / ``false``)."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()
        values = {"true": True, "false": False}
        if normalized not in values:
            raise ValueError(
                "Success info values must be 'true' or 'false' "
                f"(Schola string info); got {value!r}"
            )
        return values[normalized]

    if isinstance(value, np.ndarray):
        values = np.asarray(value, dtype=object)
        parsed = np.fromiter(
            (_coerce_success(item) for item in values.flat),
            dtype=np.bool_,
            count=values.size,
        )
        return parsed.reshape(values.shape)

    raise TypeError(
        "Success info must be a bool or 'true'/'false' string; "
        f"got {type(value).__name__}"
    )


def _convert_image_space(space: gym.Space, name: str) -> tuple[Box, str]:
    if not isinstance(space, Box) or len(space.shape) != SINGLE_IMAGE_NDIMS:
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
    if space.shape[HWC_CHANNEL_DIM] in SUPPORTED_IMAGE_CHANNELS:
        height, width, channels = space.shape
        layout = "hwc"
    elif space.shape[CHW_CHANNEL_DIM] in SUPPORTED_IMAGE_CHANNELS:
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
        if image.ndim not in (SINGLE_IMAGE_NDIMS, BATCHED_IMAGE_NDIMS):
            raise ValueError(f"Expected a CHW image batch, got shape {image.shape}")
        image = np.moveaxis(image, CHW_CHANNEL_DIM, HWC_CHANNEL_DIM)
    if mode.endswith("float"):
        image = np.rint(np.clip(image, 0, 1) * UINT8_MAX).astype(np.uint8)
    return np.ascontiguousarray(image)


class LeRobotScholaVectorEnv(gym.vector.VectorEnv):
    """Adapt Schola's vector environment to LeRobot's ``gym.vector.VectorEnv`` interface."""

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        *,
        task: str,
        task_description: str,
        max_episode_steps: int,
        observation_config: ScholaObservationConfig,
        success_key: str | None = None,
        render_camera: str | None = None,
        render_fps: int = 30,
    ) -> None:
        super().__init__()
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be at least 1")
        if render_fps < 1:
            raise ValueError("render_fps must be at least 1")
        if not _contains_only_boxes(env.single_action_space):
            raise TypeError(
                "LeRobot requires continuous actions; Schola's action space must be "
                "a Box or a nested Dict containing only Box spaces."
            )

        self.env = env
        self.num_envs = env.num_envs
        self.task = task
        self.task_description = task_description
        self._max_episode_steps = max_episode_steps
        self.success_key = success_key
        self.render_camera = render_camera
        self._latest_observation: dict[str, Any] | None = None
        self.metadata = dict(getattr(env.unwrapped, "metadata", {}))
        self.metadata["render_fps"] = render_fps
        env.unwrapped.metadata = self.metadata

        flat_action_space = flatten_space(env.single_action_space)
        if not isinstance(flat_action_space, Box):
            raise TypeError("Flattening Schola's action space did not produce a Box")
        self.single_action_space = flat_action_space
        self.action_space = batch_space(flat_action_space, n=env.num_envs)

        self._observation_config = observation_config
        self._image_modes: dict[str, str] = {}
        self._vector_dtypes: dict[str, np.dtype] = {}
        self._validate_observation_config(env.single_observation_space)
        self.single_observation_space = self._build_observation_space(
            env.single_observation_space
        )
        self.observation_space = batch_space(
            self.single_observation_space, n=env.num_envs
        )
        self._validate_render_camera()

    @property
    def unwrapped(self) -> gym.vector.VectorEnv:
        return self.env.unwrapped

    def _validate_render_camera(self) -> None:
        pixels_space = self.single_observation_space.spaces.get("pixels")
        if isinstance(pixels_space, Dict):
            camera_names = list(pixels_space.spaces)
            if not camera_names:
                raise ValueError("The mapped pixels observation contains no cameras")
            if self.render_camera is None:
                self.render_camera = camera_names[0]
            elif self.render_camera not in pixels_space.spaces:
                raise ValueError(
                    f"render_camera {self.render_camera!r} is not available; "
                    f"choose one of {camera_names}"
                )
        elif isinstance(pixels_space, Box) and self.render_camera is not None:
            raise ValueError(
                "render_camera can only be set when observations are mapped as pixels/<camera>"
            )
        elif pixels_space is None and self.render_camera is not None:
            raise ValueError(
                "render_camera was set, but no observations are mapped under pixels"
            )

    def _validate_observation_config(self, space: gym.Space) -> None:
        """Check the config against Schola's actual observation space.

        ``ScholaObservationConfig`` validates its own internal consistency
        (overlaps, reserved names, duplicate sources) when constructed; this
        only checks what can't be known until the real space is available.
        """
        if not isinstance(space, Dict):
            raise TypeError(
                "Schola must expose a Dict observation space so observations "
                "can be grouped for LeRobot."
            )

        config = self._observation_config
        claimed_keys: set[str] = set()

        def claim_source(source_key: str, owner: str) -> None:
            if source_key not in space.spaces:
                raise ValueError(
                    f"{owner} references unknown Schola observation " f"{source_key!r}"
                )
            claimed_keys.add(source_key)

        for camera_name, source_key in config.cameras.items():
            claim_source(source_key, f"camera {camera_name!r}")

        for target_key, source_keys in config.vectors.items():
            for source_key in source_keys:
                claim_source(source_key, f"vector {target_key!r}")
                if not isinstance(space.spaces[source_key], Box):
                    raise TypeError(
                        f"Vector source {source_key!r} must use a Box space"
                    )

        for target_key, source_key in config.passthrough.items():
            claim_source(source_key, f"passthrough {target_key!r}")
            if not isinstance(space.spaces[source_key], Box):
                raise TypeError(
                    f"Passthrough source {source_key!r} must use a Box space"
                )

        missing_keys = space.spaces.keys() - claimed_keys
        if missing_keys:
            raise ValueError(
                "Observation configuration does not account for Schola keys: "
                f"{sorted(missing_keys)}"
            )

    def _build_observation_space(self, space: gym.Space) -> Dict:
        if not isinstance(space, Dict):
            raise TypeError(
                "Expected the validated Schola observation space to be a Dict"
            )

        config = self._observation_config
        output_spaces: dict[str, gym.Space] = {}
        if config.cameras:
            camera_spaces: dict[str, gym.Space] = {}
            for camera_name, source_key in config.cameras.items():
                camera_spaces[camera_name], self._image_modes[source_key] = (
                    _convert_image_space(space.spaces[source_key], source_key)
                )
            output_spaces["pixels"] = Dict(camera_spaces)

        for target_key, source_keys in config.vectors.items():
            source_spaces = [space.spaces[source_key] for source_key in source_keys]
            flattened_space = flatten_space(Tuple(tuple(source_spaces)))
            if not isinstance(flattened_space, Box):
                raise TypeError(
                    f"Vector output {target_key!r} did not flatten to a Box"
                )
            dtype = np.result_type(np.float32, flattened_space.dtype)
            output_spaces[target_key] = Box(
                low=flattened_space.low.astype(dtype, copy=False),
                high=flattened_space.high.astype(dtype, copy=False),
                dtype=dtype,
            )
            self._vector_dtypes[target_key] = np.dtype(dtype)

        for target_key, source_key in config.passthrough.items():
            output_spaces[target_key] = space.spaces[source_key]

        return Dict(output_spaces)

    def _convert_observation(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Convert one batched Schola observation to LeRobot's layout."""
        config = self._observation_config
        converted: dict[str, Any] = {}

        if config.cameras:
            converted["pixels"] = {
                camera_name: _convert_image_value(
                    observation[source_key],
                    self._image_modes[source_key],
                )
                for camera_name, source_key in config.cameras.items()
            }

        for target_key, source_keys in config.vectors.items():
            values = [
                np.asarray(observation[source_key]).reshape(self.num_envs, -1)
                for source_key in source_keys
            ]
            converted[target_key] = np.concatenate(values, axis=-1).astype(
                self._vector_dtypes[target_key],
                copy=False,
            )

        for target_key, source_key in config.passthrough.items():
            converted[target_key] = observation[source_key]

        return converted

    def _convert_action(self, action: np.ndarray) -> Any:
        action = np.asarray(action)
        expected_shape = self.action_space.shape
        if action.shape != expected_shape:
            raise ValueError(
                f"Expected LeRobot action shape {expected_shape}, got {action.shape}"
            )

        unflattened = [
            unflatten(self.env.single_action_space, value) for value in action
        ]
        batched_action = create_empty_array(
            self.env.single_action_space, n=self.num_envs
        )
        return concatenate(self.env.single_action_space, unflattened, batched_action)

    def _normalize_info(self, info: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(info)
        if self.success_key is not None and self.success_key in normalized:
            source_mask = normalized.get(f"_{self.success_key}")
            if source_mask is None:
                normalized["is_success"] = _coerce_success(normalized[self.success_key])
            else:
                source_values = np.asarray(normalized[self.success_key], dtype=object)
                source_mask = np.asarray(source_mask, dtype=np.bool_)
                success_values = np.zeros(source_mask.shape, dtype=np.bool_)
                success_values[source_mask] = _coerce_success(
                    source_values[source_mask]
                )
                normalized["is_success"] = success_values
                normalized["_is_success"] = source_mask

        final_info = normalized.get("final_info")
        if isinstance(final_info, dict):
            normalized["final_info"] = self._normalize_info(final_info)
        elif isinstance(final_info, np.ndarray):
            normalized["final_info"] = np.asarray(
                [
                    self._normalize_info(item) if isinstance(item, dict) else item
                    for item in final_info
                ],
                dtype=object,
            )
        return normalized

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        schola_options = dict(options or {})
        schola_options.pop(NEW_ROLLOUT_OPTION, None)
        observation, info = self.env.reset(
            seed=seed,
            options=schola_options or None,
        )
        converted_observation = self._convert_observation(observation)
        self._latest_observation = converted_observation
        return converted_observation, self._normalize_info(info)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(
            self._convert_action(action)
        )
        converted_observation = self._convert_observation(observation)
        self._latest_observation = converted_observation
        return (
            converted_observation,
            reward,
            terminated,
            truncated,
            self._normalize_info(info),
        )

    def render(self) -> tuple[np.ndarray, ...]:
        if self._latest_observation is None:
            raise RuntimeError("reset() must be called before render()")

        pixels = self._latest_observation.get("pixels")
        if pixels is None:
            raise NotImplementedError(
                "Schola rendering requires at least one observation mapped to pixels"
            )
        if isinstance(pixels, dict):
            if self.render_camera is None:
                raise RuntimeError("No render camera was selected")
            pixels = pixels[self.render_camera]

        frames = np.asarray(pixels)
        if (
            frames.ndim != BATCHED_IMAGE_NDIMS
            or frames.shape[BATCH_DIM] != self.num_envs
        ):
            raise ValueError(
                "Render observations must have shape "
                f"(num_envs, height, width, channels); got {frames.shape}"
            )
        return tuple(frames[index] for index in range(self.num_envs))

    def get_attr(self, name: str) -> tuple[Any, ...]:
        if name in {"task", "task_description", "_max_episode_steps"}:
            return (getattr(self, name),) * self.num_envs
        return tuple(self.env.get_attr(name))

    def call(self, name: str, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        if name == "render":
            if args or kwargs:
                raise TypeError("render() does not accept arguments")
            return self.render()
        if name in {"task", "task_description", "_max_episode_steps"}:
            value = getattr(self, name)
            return tuple(
                value(*args, **kwargs) if callable(value) else value
                for _ in range(self.num_envs)
            )

        call = getattr(self.env, "call", None)
        if call is None:
            raise AttributeError(
                f"{type(self.env).__name__} does not support call({name!r})"
            )
        return tuple(call(name, *args, **kwargs))

    def close_extras(self, **kwargs: Any) -> None:
        # `VectorEnv.__del__` may call this on an instance whose `__init__` raised
        # before `self.env` was assigned; tolerate that instead of erroring in `__del__`.
        env = getattr(self, "env", None)
        if env is not None:
            env.close(**kwargs)
