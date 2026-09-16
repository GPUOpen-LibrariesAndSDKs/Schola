# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Gymnasium vector adapter between Schola and LeRobot."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict
from gymnasium.spaces.utils import flatten, flatten_space, unflatten
from gymnasium.vector import AutoresetMode
from gymnasium.vector.utils import (
    batch_space,
    concatenate,
    create_empty_array,
    iterate,
)
from numpy.typing import NDArray

from lerobot.envs.utils import NEW_ROLLOUT_OPTION
from lerobot.utils.constants import OBS_IMAGE, OBS_IMAGES, OBS_PREFIX
from lerobot_env_schola.config import BaseScholaEnvConfig, ScholaObservationConfig
from lerobot_env_schola.feature_mapping import is_image_policy_key
from schola.gym.env import GymVectorEnv

BATCHED_IMAGE_NDIMS = 4
BATCH_DIM = 0
SCHOLA_OBSERVATION_ROOT = "observation"

# Batched Schola Gym observations: nested dicts whose leaves are arrays.
type ScholaObservationTree = NDArray[np.generic] | Mapping[str, ScholaObservationTree]


def _contains_only_boxes(space: gym.Space) -> bool:
    if isinstance(space, Box):
        return True
    if isinstance(space, Dict):
        return all(_contains_only_boxes(child) for child in space.spaces.values())
    return False


def _iter_schola_observation_leaves(
    space: gym.Space,
    observation: ScholaObservationTree | None = None,
    *,
    prefix: tuple[str, ...] = (SCHOLA_OBSERVATION_ROOT,),
) -> Iterator[tuple[str, gym.Space, ScholaObservationTree | None]]:
    """Yield dotted Schola sources and their leaf spaces, with values when given."""
    source = ".".join(prefix)
    if not isinstance(space, Dict):
        yield source, space, observation
        return

    if observation is not None and not isinstance(observation, Mapping):
        raise TypeError(
            f"Cannot traverse Schola source {source!r}: expected a mapping, "
            f"got {type(observation).__name__}"
        )

    for key, child in space.spaces.items():
        if not key:
            raise ValueError("Schola observation keys cannot be empty")
        if "." in key:
            raise ValueError(
                f"Schola observation key {key!r} contains '.', which is reserved "
                "for nested source paths"
            )
        child_observation: ScholaObservationTree | None = None
        if observation is not None:
            if key not in observation:
                raise KeyError(
                    f"Schola observation is missing source {'.'.join((*prefix, key))!r}"
                )
            child_observation = observation[key]
        yield from _iter_schola_observation_leaves(
            child, child_observation, prefix=(*prefix, key)
        )


def _flatten_batched_observation(
    space: gym.Space, value: ScholaObservationTree, num_envs: int
) -> np.ndarray:
    """Apply Gymnasium's flattening convention to a batched space value."""
    return np.stack(
        [
            flatten(space, item)
            for item in iterate(batch_space(space, n=num_envs), value)
        ],
        axis=0,
    ).reshape(num_envs, -1)


def _build_source_spaces(space: gym.Space) -> dict[str, gym.Space]:
    """Index original Schola leaves by their canonical YAML source paths."""
    return {
        source: leaf_gym_space
        for source, leaf_gym_space, _ in _iter_schola_observation_leaves(space)
    }


def _camera_name(policy_key: str) -> str:
    """Return the camera name a user configures for an image policy feature."""
    if policy_key == OBS_IMAGE:
        return "image"
    return policy_key.removeprefix(f"{OBS_IMAGES}.")


def _policy_observation_key(policy_key: str) -> str:
    """Remove the prefix LeRobot adds in ``preprocess_observation``."""
    if not policy_key.startswith(OBS_PREFIX):
        raise ValueError(
            f"Schola policy observation key {policy_key!r} must start with {OBS_PREFIX!r}"
        )
    return policy_key.removeprefix(OBS_PREFIX)


def _build_policy_observation_space(
    policy_sources: Mapping[str, tuple[str, ...]],
    source_spaces: Mapping[str, gym.Space],
) -> Dict:
    """Build the policy-shaped Gym space returned by this vector adapter."""
    gym_spaces: dict[str, gym.Space] = {}
    for policy_key, sources in policy_sources.items():
        if is_image_policy_key(policy_key):
            output_space = source_spaces[sources[0]]
            if not isinstance(output_space, Box):
                raise TypeError(f"Image observation {policy_key!r} must be a Box")
            if output_space.dtype == np.dtype(np.uint8):
                output_space = Box(0.0, 1.0, shape=output_space.shape, dtype=np.float32)
        else:
            boxes: list[Box] = []
            for source in sources:
                flattened = flatten_space(source_spaces[source])
                if not isinstance(flattened, Box):
                    raise TypeError(
                        f"Policy observation {policy_key!r} contains a source that "
                        "cannot be flattened to a Box"
                    )
                boxes.append(flattened)
            output_space = Box(
                low=np.concatenate([box.low for box in boxes]),
                high=np.concatenate([box.high for box in boxes]),
                dtype=np.result_type(*(box.dtype for box in boxes)).type,
            )
        gym_spaces[_policy_observation_key(policy_key)] = output_space
    return Dict(gym_spaces)


def _parse_success_string(value: str) -> bool:
    """Parse one Schola success string."""
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(
            "Success info values must be 'true' or 'false' "
            f"(Schola string info); got {value!r}"
        )
    return normalized == "true"


def _parse_success_array(values: NDArray[Any]) -> NDArray[np.bool_]:
    """Parse a NumPy array of Schola success strings."""
    parsed: NDArray[np.bool_] = np.empty(values.shape, dtype=np.bool_)
    for index, value in enumerate(values):
        parsed.flat[index] = _parse_success_string(value)
    return parsed


def _normalize_info(info: dict[str, Any], success_key: str | None) -> dict[str, Any]:
    """Map a Schola success key to LeRobot ``is_success`` on top-level info.

    NextStep vector info is Gymnasium-batched: ``success_key`` is an array and
    ``_{success_key}`` marks which slots actually sent that key. There is no
    ``final_info`` walk.
    """
    normalized = dict(info)
    if success_key is None or success_key not in normalized:
        return normalized

    source_mask = normalized.get(f"_{success_key}")
    if source_mask is None:
        raise TypeError(
            "Schola vector info must include Gymnasium mask "
            f"'_{success_key}' next to {success_key!r}"
        )

    source_values = np.asarray(normalized[success_key], dtype=object)
    source_mask = np.asarray(source_mask, dtype=np.bool_)
    success_values = np.zeros(source_mask.shape, dtype=np.bool_)
    success_values[source_mask] = _parse_success_array(source_values[source_mask])
    normalized["is_success"] = success_values
    normalized["_is_success"] = source_mask
    return normalized


class LeRobotScholaVectorEnv(gym.vector.VectorEnv):
    """Adapt Schola's vector environment to LeRobot's ``gym.vector.VectorEnv`` interface.

    Constructs Schola's ``GymVectorEnv`` in ``AutoresetMode.NEXT_STEP``, so
    LeRobot reads ``is_success`` from top-level ``info`` on the done step.
    """

    def __init__(self, *, config: BaseScholaEnvConfig) -> None:
        super().__init__()
        policy_sources = ScholaObservationConfig(
            config.observations
        ).to_policy_mapping()
        task = config.task or "schola"
        env = GymVectorEnv(
            config._get_simulator_config().make(),
            config.protocol.make(),
            verbosity=config.verbosity,
            autoreset_mode=AutoresetMode.NEXT_STEP,
        )
        try:
            if not _contains_only_boxes(env.single_action_space):
                raise TypeError(
                    "LeRobot requires continuous actions; Schola's action space must be "
                    "a Box or a nested Dict containing only Box spaces."
                )

            self.env = env
            self.num_envs = env.num_envs
            self.task = task
            self.task_description = config.task_description or task
            self._max_episode_steps = config.episode_length
            self.success_key = config.success_key
            self._render_key: str | None = None
            self._latest_observation: dict[str, NDArray[np.generic]] | None = None
            self.autoreset_mode = AutoresetMode.NEXT_STEP
            self.metadata = dict(getattr(env.unwrapped, "metadata", {}))
            self.metadata["autoreset_mode"] = self.autoreset_mode
            self.metadata["render_fps"] = config.render_fps
            env.unwrapped.metadata = self.metadata

            flat_action_space = flatten_space(env.single_action_space)
            if not isinstance(flat_action_space, Box):
                raise TypeError(
                    "Flattening Schola's action space did not produce a Box"
                )
            self.single_action_space = flat_action_space
            self.action_space = batch_space(flat_action_space, n=env.num_envs)

            source_spaces = _build_source_spaces(env.single_observation_space)
            self._policy_sources = policy_sources
            self.single_observation_space = _build_policy_observation_space(
                self._policy_sources,
                source_spaces,
            )
            self.observation_space = batch_space(
                self.single_observation_space, n=env.num_envs
            )
            self.set_render_camera(config.render_camera)
        except Exception:
            env.close()
            raise

    @property
    def unwrapped(self) -> gym.vector.VectorEnv:
        return self.env.unwrapped

    def set_render_camera(self, camera_name: str | None) -> None:
        """Select which configured camera ``render()`` returns frames from."""
        available = {
            _camera_name(policy_key): policy_key
            for policy_key in self._policy_sources
            if is_image_policy_key(policy_key)
        }
        if not available:
            if camera_name is not None:
                raise ValueError(
                    "render_camera was set, but no image observation is configured"
                )
            self._render_key = None
            return

        selected = camera_name or next(iter(available))
        if selected not in available:
            raise ValueError(
                f"render_camera {selected!r} is not available; "
                f"choose one of {list(available)}"
            )
        self._render_key = _policy_observation_key(available[selected])

    def _convert_action(self, action: np.ndarray) -> ScholaObservationTree:
        unflattened = [
            unflatten(self.env.single_action_space, value)
            for value in np.asarray(action)
        ]
        batched_action = create_empty_array(
            self.env.single_action_space, n=self.num_envs
        )
        return concatenate(self.env.single_action_space, unflattened, batched_action)

    def _convert_schola_observation(
        self, observation: ScholaObservationTree
    ) -> dict[str, NDArray[np.generic]]:
        """Convert a nested Schola value directly into policy-shaped Gym arrays."""
        source_values: dict[str, NDArray[np.generic]] = {}
        for source, leaf_space, value in _iter_schola_observation_leaves(
            self.env.single_observation_space, observation
        ):
            if value is None:
                raise ValueError(f"Schola observation source {source!r} cannot be None")
            source_values[source] = (
                np.asarray(value)
                if isinstance(leaf_space, Box)
                else _flatten_batched_observation(leaf_space, value, self.num_envs)
            )

        converted: dict[str, NDArray[np.generic]] = {}
        for policy_key, sources in self._policy_sources.items():
            values = [source_values[source] for source in sources]
            if is_image_policy_key(policy_key):
                image = values[0]
                if image.dtype == np.dtype(np.uint8):
                    image = image.astype(np.float32) / np.float32(255)
                converted[_policy_observation_key(policy_key)] = image
            else:
                converted[_policy_observation_key(policy_key)] = np.concatenate(
                    [value.reshape(self.num_envs, -1) for value in values],
                    axis=-1,
                )
        return converted

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
        self._latest_observation = self._convert_schola_observation(observation)
        return self._latest_observation, _normalize_info(info, self.success_key)

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(
            self._convert_action(actions)
        )
        self._latest_observation = self._convert_schola_observation(observation)
        return (
            self._latest_observation,
            reward,
            terminated,
            truncated,
            _normalize_info(info, self.success_key),
        )

    def render(self) -> tuple[np.ndarray, ...]:
        if self._latest_observation is None:
            raise RuntimeError("reset() must be called before render()")
        if self._render_key is None:
            raise ValueError("Schola rendering requires a configured image observation")
        frames = np.asarray(self._latest_observation[self._render_key])
        if (
            frames.ndim != BATCHED_IMAGE_NDIMS
            or frames.shape[BATCH_DIM] != self.num_envs
        ):
            raise ValueError(
                "Render observations must have shape (num_envs, channels, "
                f"height, width); got {frames.shape}"
            )
        frames = np.moveaxis(frames, 1, -1)
        if np.issubdtype(frames.dtype, np.floating):
            frames = np.rint(np.clip(frames, 0, 1) * 255).astype(np.uint8)
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
