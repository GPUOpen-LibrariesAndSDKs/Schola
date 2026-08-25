# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Gymnasium vector adapter between Schola and LeRobot."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Tuple
from gymnasium.spaces.utils import flatten, flatten_space, unflatten
from gymnasium.vector.utils import (
    batch_space,
    concatenate,
    create_empty_array,
    iterate,
)
from lerobot.envs.utils import NEW_ROLLOUT_OPTION
from lerobot.utils.constants import (
    OBS_ENV_STATE,
    OBS_IMAGE,
    OBS_IMAGES,
    OBS_PREFIX,
    OBS_STATE,
)
from lerobot_env_schola.config import (
    HWC_CHANNEL_DIM,
    SINGLE_IMAGE_NDIMS,
    SUPPORTED_IMAGE_CHANNELS,
    ScholaObservationConfig,
)

logger = logging.getLogger(__name__)

BATCHED_IMAGE_NDIMS = 4
CHW_CHANNEL_DIM = -3
UINT8_MAX = np.iinfo(np.uint8).max
BATCH_DIM = 0
SCHOLA_OBSERVATION_ROOT = "observation"

_EXACT_POLICY_OUTPUTS = {
    OBS_IMAGE: ("single_image", "pixels"),
    OBS_STATE: ("value", "agent_pos"),
    OBS_ENV_STATE: ("value", "environment_state"),
}


def _contains_only_boxes(space: gym.Space) -> bool:
    if isinstance(space, Box):
        return True
    if isinstance(space, Dict):
        return all(_contains_only_boxes(child) for child in space.spaces.values())
    return False


def _flatten_observation_spaces(
    space: gym.Space, prefix: tuple[str, ...] = (SCHOLA_OBSERVATION_ROOT,)
) -> dict[str, gym.Space]:
    """Flatten Schola leaves below a virtual ``observation`` root."""
    if not isinstance(space, Dict):
        return {".".join(prefix): space}

    flattened: dict[str, gym.Space] = {}
    for key, child in space.spaces.items():
        if not key:
            raise ValueError("Schola observation keys cannot be empty")
        if "." in key:
            raise ValueError(
                f"Schola observation key {key!r} contains '.', which is reserved "
                "for nested source paths"
            )
        flattened.update(_flatten_observation_spaces(child, (*prefix, key)))
    return flattened


def _get_observation_value(observation: Any, source: str) -> Any:
    """Resolve a dot-separated source path in one batched Schola observation."""
    segments = source.split(".")
    if not segments or segments[0] != SCHOLA_OBSERVATION_ROOT:
        raise ValueError(
            f"Schola source {source!r} must start with " f"{SCHOLA_OBSERVATION_ROOT!r}"
        )
    if len(segments) == 1:
        return observation

    value = observation
    for segment in segments[1:]:
        if not isinstance(value, Mapping):
            raise TypeError(
                f"Cannot traverse Schola source {source!r}: {segment!r} is below "
                f"a non-mapping {type(value).__name__}"
            )
        if segment not in value:
            raise KeyError(f"Schola observation is missing source {source!r}")
        value = value[segment]
    return value


def _flatten_batched_observation(
    space: gym.Space, value: Any, num_envs: int
) -> np.ndarray:
    """Apply Gymnasium's flattening convention to a batched space value."""
    return np.stack(
        [
            flatten(space, item)
            for item in iterate(batch_space(space, n=num_envs), value)
        ],
        axis=0,
    ).reshape(num_envs, -1)


def _policy_output(policy_key: str) -> tuple[str, str]:
    """Return adapter behavior and Gym key for a canonical policy feature."""
    exact_output = _EXACT_POLICY_OUTPUTS.get(policy_key)
    if exact_output is not None:
        return exact_output

    image_prefix = f"{OBS_IMAGES}."
    if policy_key.startswith(image_prefix):
        camera_name = policy_key.removeprefix(image_prefix)
        if not camera_name:
            raise ValueError(f"Policy image feature {policy_key!r} has no camera name")
        return "camera", camera_name

    if policy_key.startswith(OBS_PREFIX):
        gym_key = policy_key.removeprefix(OBS_PREFIX)
        if gym_key:
            return "value", gym_key

    raise ValueError(
        f"Observation mapping key {policy_key!r} is not a canonical LeRobot "
        "observation feature"
    )


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
        self._source_spaces: dict[str, gym.Space] = {}
        self._source_box_spaces: dict[str, Box] = {}
        self._camera_sources: dict[str, str] = {}
        self._single_image_source: str | None = None
        self._value_sources: dict[str, tuple[str, ...]] = {}
        self._concatenated_outputs: set[str] = set()
        self._image_modes: dict[str, str] = {}
        self._vector_dtypes: dict[str, np.dtype] = {}
        self._validate_observation_config(env.single_observation_space)
        self.single_observation_space = self._build_observation_space()
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
        elif isinstance(pixels_space, Box):
            if self.render_camera not in (None, "image"):
                raise ValueError(
                    "A singular observation.image can only use render_camera 'image'"
                )
            self.render_camera = "image"
        elif pixels_space is None and self.render_camera is not None:
            raise ValueError(
                "render_camera was set, but no observations are mapped under pixels"
            )

    def _validate_observation_config(self, space: gym.Space) -> None:
        """Check the config against Schola's actual observation space.

        Policy feature names determine output behavior. Source strings address
        flattened Schola leaves; lists flatten and concatenate in order.
        """
        config = self._observation_config
        self._source_spaces = _flatten_observation_spaces(space)
        claimed_sources: dict[str, list[str]] = {}
        claimed_gym_keys: dict[str, str] = {}

        def claim_source(source: str, owner: str) -> None:
            if not source:
                raise ValueError(f"{owner} contains an empty Schola source path")
            if source not in self._source_spaces:
                raise ValueError(
                    f"{owner} references unknown Schola observation {source!r}; "
                    f"available sources are {sorted(self._source_spaces)}"
                )
            previous_owners = claimed_sources.setdefault(source, [])
            if previous_owners:
                logger.warning(
                    "Schola observation %r is reused by %s; previous use(s): %s. "
                    "This duplicates policy input data, may increase preprocessing "
                    "and device-memory costs, and can be semantically incorrect.",
                    source,
                    owner,
                    ", ".join(previous_owners),
                )
            source_space = self._source_spaces[source]
            box_space = (
                source_space
                if isinstance(source_space, Box)
                else flatten_space(source_space)
            )
            if not isinstance(box_space, Box):
                raise TypeError(
                    f"{owner} source {source!r} uses "
                    f"{type(source_space).__name__}, which Gymnasium cannot "
                    "flatten to a fixed-shape Box"
                )
            self._source_box_spaces[source] = box_space
            previous_owners.append(owner)

        for policy_key, configured_sources in config.items():
            behavior, gym_key = _policy_output(policy_key)
            owner = f"policy feature {policy_key!r}"
            if gym_key in claimed_gym_keys:
                raise ValueError(
                    f"Policy features {claimed_gym_keys[gym_key]} and {policy_key!r} "
                    f"both produce adapter output {gym_key!r}"
                )
            claimed_gym_keys[gym_key] = policy_key

            if isinstance(configured_sources, str):
                sources = (configured_sources,)
            elif isinstance(configured_sources, list):
                if not configured_sources:
                    raise ValueError(f"{owner} requires at least one source")
                if not all(isinstance(source, str) for source in configured_sources):
                    raise TypeError(f"{owner} sources must all be strings")
                sources = tuple(configured_sources)
            else:
                raise TypeError(
                    f"{owner} must map to a source string or list of strings"
                )

            if behavior in {"camera", "single_image"} and len(sources) != 1:
                raise ValueError(f"{owner} must map to exactly one image source")
            for source in sources:
                claim_source(source, owner)

            if behavior == "camera":
                self._camera_sources[gym_key] = sources[0]
            elif behavior == "single_image":
                if self._camera_sources:
                    raise ValueError(
                        "observation.image cannot be combined with observation.images.*"
                    )
                self._single_image_source = sources[0]
            else:
                self._value_sources[gym_key] = sources
                if isinstance(configured_sources, list):
                    self._concatenated_outputs.add(gym_key)

        if self._single_image_source is not None and self._camera_sources:
            raise ValueError(
                "observation.image cannot be combined with observation.images.*"
            )

        missing_sources = self._source_spaces.keys() - claimed_sources.keys()
        if missing_sources:
            logger.warning(
                "Schola observations %s are not mapped to policy inputs and will "
                "be ignored by the LeRobot adapter.",
                sorted(missing_sources),
            )

    def _build_observation_space(self) -> Dict:
        output_spaces: dict[str, gym.Space] = {}
        if self._camera_sources:
            camera_spaces: dict[str, gym.Space] = {}
            for camera_name, source in self._camera_sources.items():
                camera_spaces[camera_name], self._image_modes[camera_name] = (
                    _convert_image_space(self._source_spaces[source], source)
                )
            output_spaces["pixels"] = Dict(camera_spaces)
        elif self._single_image_source is not None:
            output_spaces["pixels"], self._image_modes["pixels"] = _convert_image_space(
                self._source_spaces[self._single_image_source],
                self._single_image_source,
            )

        for gym_key, sources in self._value_sources.items():
            if gym_key not in self._concatenated_outputs:
                output_spaces[gym_key] = self._source_box_spaces[sources[0]]
                continue

            source_spaces = [self._source_box_spaces[source] for source in sources]
            flattened_space = flatten_space(Tuple(tuple(source_spaces)))
            if not isinstance(flattened_space, Box):
                raise TypeError(f"Adapter output {gym_key!r} did not flatten to a Box")
            dtype = np.result_type(np.float32, flattened_space.dtype)
            output_spaces[gym_key] = Box(
                low=flattened_space.low.astype(dtype, copy=False),
                high=flattened_space.high.astype(dtype, copy=False),
                dtype=dtype,
            )
            self._vector_dtypes[gym_key] = np.dtype(dtype)

        return Dict(output_spaces)

    def _convert_observation(self, observation: Any) -> dict[str, Any]:
        """Convert one batched Schola observation to LeRobot's layout."""
        converted: dict[str, Any] = {}

        if self._camera_sources:
            converted["pixels"] = {
                camera_name: _convert_image_value(
                    _get_observation_value(observation, source),
                    self._image_modes[camera_name],
                )
                for camera_name, source in self._camera_sources.items()
            }
        elif self._single_image_source is not None:
            converted["pixels"] = _convert_image_value(
                _get_observation_value(observation, self._single_image_source),
                self._image_modes["pixels"],
            )

        for gym_key, sources in self._value_sources.items():
            if gym_key not in self._concatenated_outputs:
                source = sources[0]
                value = _get_observation_value(observation, source)
                source_space = self._source_spaces[source]
                converted[gym_key] = (
                    value
                    if isinstance(source_space, Box)
                    else _flatten_batched_observation(
                        source_space, value, self.num_envs
                    )
                )
                continue
            values = [
                (
                    np.asarray(_get_observation_value(observation, source)).reshape(
                        self.num_envs, -1
                    )
                    if isinstance(self._source_spaces[source], Box)
                    else _flatten_batched_observation(
                        self._source_spaces[source],
                        _get_observation_value(observation, source),
                        self.num_envs,
                    )
                )
                for source in sources
            ]
            converted[gym_key] = np.concatenate(values, axis=-1).astype(
                self._vector_dtypes[gym_key],
                copy=False,
            )

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
