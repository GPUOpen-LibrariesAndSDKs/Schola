# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Gymnasium vector adapter between Schola and LeRobot."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict
from gymnasium.spaces.utils import flatten_space, unflatten
from gymnasium.vector.utils import batch_space, concatenate, create_empty_array
from lerobot.envs.utils import NEW_ROLLOUT_OPTION
from lerobot_env_schola.config import ScholaObservationConfig
from lerobot_env_schola.observations import (
    BATCH_DIM,
    BATCHED_IMAGE_NDIMS,
    ObservationAdapter,
)


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


class LeRobotScholaVectorEnv(gym.vector.VectorWrapper):
    """Adapt a Gymnasium vector environment carrying Schola data to LeRobot."""

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

        flat_action_space = flatten_space(env.single_action_space)
        if not isinstance(flat_action_space, Box):
            raise TypeError("Flattening Schola's action space did not produce a Box")
        self.single_action_space = flat_action_space
        self.action_space = batch_space(flat_action_space, n=env.num_envs)

        self.observation_adapter = ObservationAdapter(
            source_space=env.single_observation_space,
            config=observation_config,
            num_envs=env.num_envs,
        )
        self.single_observation_space = (
            self.observation_adapter.single_observation_space
        )
        self.observation_space = batch_space(
            self.single_observation_space, n=env.num_envs
        )
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
            raise ValueError(
                "render_camera was set, but no observations are mapped under pixels"
            )

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
        converted_observation = self.observation_adapter.convert(observation)
        self._latest_observation = converted_observation
        return converted_observation, self._normalize_info(info)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(
            self._convert_action(action)
        )
        converted_observation = self.observation_adapter.convert(observation)
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
