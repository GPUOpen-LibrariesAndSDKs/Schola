"""Gymnasium vector adapter between Schola and LeRobot."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict
from gymnasium.spaces.utils import flatten_space, unflatten
from gymnasium.vector.utils import batch_space, concatenate, create_empty_array

LEROBOT_NEW_ROLLOUT_OPTION = "lerobot_new_rollout"


def _contains_only_boxes(space: gym.Space) -> bool:
    if isinstance(space, Box):
        return True
    if isinstance(space, Dict):
        return all(_contains_only_boxes(child) for child in space.spaces.values())
    return False


def _coerce_success(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _coerce_success(value.item())
        return np.asarray(
            [_coerce_success(item) for item in value], dtype=np.bool_
        ).reshape(value.shape)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return type(value)(_coerce_success(item) for item in value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no", ""}:
            return False
        raise ValueError(f"Cannot convert success value {value!r} to bool")
    return bool(value)


def _convert_image_space(space: gym.Space, name: str) -> tuple[Box, str]:
    if not isinstance(space, Box) or len(space.shape) != 3:
        raise TypeError(f"Image observation {name!r} must be a three-dimensional Box")

    is_float = np.issubdtype(space.dtype, np.floating)
    is_uint8 = space.dtype == np.dtype(np.uint8)
    if not (is_float or is_uint8):
        raise TypeError(f"Image observation {name!r} must use float or uint8 values")
    if is_float and not (np.all(space.low >= 0) and np.all(space.high <= 1)):
        raise ValueError(f"Floating-point image observation {name!r} must be bounded within [0, 1]")

    if is_uint8 and space.shape[-1] in (1, 3, 4):
        height, width, channels = space.shape
        layout = "hwc"
    elif space.shape[0] in (1, 3, 4):
        channels, height, width = space.shape
        layout = "chw"
    elif space.shape[-1] in (1, 3, 4):
        height, width, channels = space.shape
        layout = "hwc"
    else:
        raise ValueError(
            f"Image observation {name!r} must have 1, 3, or 4 channels; got shape {space.shape}"
        )

    mode = f"{layout}_{'float' if is_float else 'uint8'}"
    return Box(0, 255, shape=(height, width, channels), dtype=np.uint8), mode


def _convert_image_value(value: Any, mode: str | dict[str, Any]) -> Any:
    if isinstance(mode, dict):
        return {
            key: _convert_image_value(value[key], child_mode)
            for key, child_mode in mode.items()
        }

    image = np.asarray(value)
    if mode.startswith("chw"):
        if image.ndim == 4:
            image = np.transpose(image, (0, 2, 3, 1))
        elif image.ndim == 3:
            image = np.transpose(image, (1, 2, 0))
        else:
            raise ValueError(f"Expected a CHW image batch, got shape {image.shape}")
    if mode.endswith("float"):
        image = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    return np.ascontiguousarray(image)


class LeRobotScholaVectorEnv(gym.vector.VectorWrapper):
    """Adapt Schola's native vector environment to LeRobot's rollout contract."""

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        *,
        task: str,
        task_description: str,
        max_episode_steps: int,
        success_key: str = "is_success",
        observation_map: Mapping[str, str],
        render_camera: str | None = None,
        render_fps: int = 30,
    ) -> None:
        super().__init__(env)
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be at least 1")
        if render_fps < 1:
            raise ValueError("render_fps must be at least 1")
        if not _contains_only_boxes(env.single_action_space):
            raise TypeError(
                "LeRobot requires continuous actions; Schola's action space must be "
                "a Box or a nested Dict containing only Box spaces."
            )

        self.task = task
        self.task_description = task_description
        self._max_episode_steps = max_episode_steps
        self.success_key = success_key
        self.render_camera = render_camera
        self._latest_observation: dict[str, Any] | None = None
        self.metadata = dict(getattr(env.unwrapped, "metadata", {}))
        self.metadata["render_fps"] = render_fps
        env.unwrapped.metadata = self.metadata
        self.observation_map = self._resolve_observation_map(
            env.single_observation_space, observation_map
        )

        flat_action_space = flatten_space(env.single_action_space)
        if not isinstance(flat_action_space, Box):
            raise TypeError("Flattening Schola's action space did not produce a Box")
        self.single_action_space = flat_action_space
        self.action_space = batch_space(flat_action_space, n=env.num_envs)

        self._image_modes: dict[str, str | dict[str, Any]] = {}
        self.single_observation_space = self._make_observation_space(
            env.single_observation_space
        )
        self.observation_space = batch_space(self.single_observation_space, n=env.num_envs)
        self._validate_render_camera()

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
            raise ValueError("render_camera was set, but no observations are mapped under pixels")

    def _resolve_observation_map(
        self,
        space: gym.Space,
        observation_map: Mapping[str, str],
    ) -> dict[str, str]:
        if not isinstance(space, Dict):
            raise TypeError(
                "Schola must expose a Dict observation space so every observation "
                "can be declared in observation_map."
            )

        requested = dict(observation_map)
        if not requested:
            raise ValueError("observation_map is required and cannot be empty")

        unknown_keys = requested.keys() - space.spaces.keys()
        if unknown_keys:
            raise ValueError(f"observation_map contains unknown Schola keys: {sorted(unknown_keys)}")
        missing_keys = space.spaces.keys() - requested.keys()
        if missing_keys:
            raise ValueError(
                f"observation_map is missing Schola keys: {sorted(missing_keys)}"
            )
        return requested

    def _make_observation_space(self, space: gym.Space) -> Dict:
        if not isinstance(space, Dict):
            raise TypeError("Expected the validated Schola observation space to be a Dict")

        renamed_spaces: dict[str, gym.Space] = {}
        pixel_spaces: dict[str, gym.Space] = {}
        for source_key, child_space in space.spaces.items():
            target_key = self.observation_map.get(source_key, source_key)
            if target_key.startswith("pixels/"):
                camera_name = target_key.removeprefix("pixels/")
                if not camera_name:
                    raise ValueError("A pixels/<camera> mapping requires a camera name")
                if camera_name in pixel_spaces:
                    raise ValueError(f"Multiple Schola observations map to camera {camera_name!r}")
                pixel_spaces[camera_name], self._image_modes[source_key] = (
                    _convert_image_space(child_space, source_key)
                )
                continue
            if target_key in renamed_spaces:
                raise ValueError(f"Multiple Schola observations map to {target_key!r}")
            if target_key == "pixels":
                if isinstance(child_space, Dict):
                    converted_cameras: dict[str, gym.Space] = {}
                    camera_modes: dict[str, Any] = {}
                    for camera_name, camera_space in child_space.spaces.items():
                        converted_cameras[camera_name], camera_modes[camera_name] = (
                            _convert_image_space(camera_space, f"{source_key}/{camera_name}")
                        )
                    renamed_spaces[target_key] = Dict(converted_cameras)
                    self._image_modes[source_key] = camera_modes
                else:
                    renamed_spaces[target_key], self._image_modes[source_key] = (
                        _convert_image_space(child_space, source_key)
                    )
            else:
                renamed_spaces[target_key] = child_space
        if pixel_spaces:
            if "pixels" in renamed_spaces:
                raise ValueError("Cannot combine a 'pixels' observation with pixels/<camera> mappings")
            renamed_spaces["pixels"] = Dict(pixel_spaces)
        return Dict(renamed_spaces)

    def _convert_observation(self, observation: Any) -> dict[str, Any]:
        converted: dict[str, Any] = {}
        for source_key, value in observation.items():
            target_key = self.observation_map[source_key]
            if source_key in self._image_modes:
                value = _convert_image_value(value, self._image_modes[source_key])
            if target_key.startswith("pixels/"):
                camera_name = target_key.removeprefix("pixels/")
                converted.setdefault("pixels", {})[camera_name] = value
            else:
                converted[target_key] = value
        return converted

    def _convert_action(self, action: np.ndarray) -> Any:
        action = np.asarray(action)
        expected_shape = self.action_space.shape
        if action.shape != expected_shape:
            raise ValueError(f"Expected LeRobot action shape {expected_shape}, got {action.shape}")

        unflattened = [unflatten(self.env.single_action_space, value) for value in action]
        batched_action = create_empty_array(self.env.single_action_space, n=self.num_envs)
        return concatenate(self.env.single_action_space, unflattened, batched_action)

    def _normalize_info(self, info: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(info)
        source_key = self.success_key if self.success_key in normalized else "is_success"
        if source_key in normalized:
            normalized["is_success"] = _coerce_success(normalized[source_key])
            source_mask = normalized.get(f"_{source_key}")
            if source_mask is not None:
                normalized["_is_success"] = source_mask
        else:
            normalized["is_success"] = np.zeros(self.num_envs, dtype=np.bool_)
            normalized["_is_success"] = np.ones(self.num_envs, dtype=np.bool_)

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
        schola_options.pop(LEROBOT_NEW_ROLLOUT_OPTION, None)
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
        if frames.ndim != 4 or frames.shape[0] != self.num_envs:
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
            raise AttributeError(f"{type(self.env).__name__} does not support call({name!r})")
        return tuple(call(name, *args, **kwargs))
