# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Utility functions for working with stable baselines 3
"""

from collections.abc import Mapping
from typing import Any, cast
from functools import singledispatch
from numpy.typing import NDArray
import torch as th
import os
import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvObs,
    VecEnv,
    VecEnvWrapper,
)
from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary


@singledispatch
def merge_spaces(space: gym.Space, *other_spaces: gym.Space) -> gym.Space:
    """
    Merge multiple gymnasium spaces of the same type into a single space.

    Parameters
    ----------
    space : gym.Space
        The first space to merge.
    *other_spaces : gym.Space
        Additional spaces to merge with the first space.

    Returns
    -------
    gym.Space
        The merged space.

    Raises
    ------
    NotImplementedError
        If the space type doesn't have a merge implementation.
    """
    raise NotImplementedError(f"Merge not implemented for space type: {type(space)}")


@merge_spaces.register(Box)
def _merge_box_spaces(space: Box, *other_spaces: Box) -> Box:
    """
    Merge multiple Box spaces by concatenating their dimensions along the last axis.
    All spaces must have the same dtype and number of dimensions.

    Parameters
    ----------
    space : Box
        The first Box space to merge.
    *other_spaces : Box
        Additional Box spaces to merge with the first space.

    Returns
    -------
    Box
        The merged Box space.
    """
    all_spaces = [space] + list(other_spaces)

    # Verify all spaces are Box spaces
    for s in all_spaces:
        if not isinstance(s, Box):
            raise TypeError(f"Cannot merge Box space with {type(s)}")

    # Verify all spaces have the same dtype
    dtype = space.dtype
    for s in all_spaces:
        if s.dtype != dtype:
            raise ValueError(
                f"All Box spaces must have the same dtype. Got {dtype} and {s.dtype}"
            )

    # Verify all spaces have the same number of dimensions
    ndim = len(space.shape)
    for s in all_spaces:
        if len(s.shape) != ndim:
            raise ValueError(
                f"All Box spaces must have the same number of dimensions. Got {ndim} and {len(s.shape)}"
            )

    # Merge by concatenating along the last axis
    if ndim == 1:
        # For 1D spaces, simply concatenate
        low = np.concatenate([s.low for s in all_spaces])
        high = np.concatenate([s.high for s in all_spaces])
    else:
        # For multi-dimensional spaces, concatenate along the last axis
        low = np.concatenate([s.low for s in all_spaces], axis=-1)
        high = np.concatenate([s.high for s in all_spaces], axis=-1)

    return Box(low=low, high=high, dtype=dtype.type)


@merge_spaces.register(MultiBinary)
def _merge_multibinary_spaces(
    space: MultiBinary, *other_spaces: MultiBinary
) -> MultiBinary:
    """
    Merge multiple MultiBinary spaces by summing their dimensions.

    Parameters
    ----------
    space : MultiBinary
        The first MultiBinary space to merge.
    *other_spaces : MultiBinary
        Additional MultiBinary spaces to merge with the first space.

    Returns
    -------
    MultiBinary
        The merged MultiBinary space.
    """
    all_spaces = [space] + list(other_spaces)

    # Verify all spaces are MultiBinary spaces
    for s in all_spaces:
        if not isinstance(s, MultiBinary):
            raise TypeError(f"Cannot merge MultiBinary space with {type(s)}")

    # Sum the dimensions
    total_n = sum(int(np.prod(s.n)) if isinstance(s.n,tuple) else s.n for s in all_spaces)
    return MultiBinary(total_n)


@merge_spaces.register(Discrete)
@merge_spaces.register(MultiDiscrete)
def _merge_discrete_spaces(
    space: Discrete | MultiDiscrete, *other_spaces: Discrete | MultiDiscrete
) -> MultiDiscrete:
    """
    Merge multiple Discrete and/or MultiDiscrete spaces into a single MultiDiscrete space.

    Parameters
    ----------
    space : Discrete | MultiDiscrete
        The first space to merge.
    *other_spaces : Discrete | MultiDiscrete
        Additional spaces to merge with the first space.

    Returns
    -------
    MultiDiscrete
        The merged MultiDiscrete space.
    """
    all_spaces = [space] + list(other_spaces)
    dims = []
    # Collect dimensions from all spaces
    for s in all_spaces:
        if isinstance(s, MultiDiscrete):
            dims += list(s.nvec)
        elif isinstance(s, Discrete):
            dims.append(s.n)
        else:
            raise TypeError(
                f"Cannot merge Discrete or MultiDiscrete space with {type(s)}"
            )
    return MultiDiscrete(dims)


def split_box_value(
    value: np.ndarray, original_spaces: dict[str, gym.Space[Any]] | gym.spaces.Dict
) -> dict[str, np.ndarray]:
    """
    Split a Box space value back into original Box spaces.
    Values are split along the last axis, which is where they were concatenated during merge.

    Parameters
    ----------
    value : np.ndarray
        The value to split.
    original_spaces : dict[str, Box]
        Dictionary mapping names to the original Box spaces.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping names to split values.
    """
    result = {}
    start_idx = 0

    for name, space in original_spaces.items():
        if not isinstance(space, Box):
            raise TypeError(f"Expected Box space for {name}, got {type(space)}")

        # Get the size along the concatenation axis (last axis), 1 if no shape information provided
        size = space.shape[-1] if len(space.shape) > 0 else 1

        # Split along the last axis
        if len(space.shape) == 0:
            # Scalar space
            result[name] = value[..., start_idx]
        else:
            result[name] = value[..., start_idx : start_idx + size]

        start_idx += size

    return result


def split_multibinary_value(
    value: np.ndarray, original_spaces: dict[str, gym.Space[Any]] | gym.spaces.Dict
) -> dict[str, np.ndarray]:
    """
    Split a MultiBinary space value back into original MultiBinary spaces.

    Parameters
    ----------
    value : np.ndarray
        The value to split.
    original_spaces : dict[str, MultiBinary]
        Dictionary mapping names to the original MultiBinary spaces.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping names to split values.
    """
    result = {}
    start_idx = 0

    for name, space in original_spaces.items():
        if not isinstance(space, MultiBinary):
            raise TypeError(f"Expected MultiBinary space for {name}, got {type(space)}")

        # Get the size for this space
        size = int(np.prod(space.n)) if isinstance(space.n,tuple) else space.n

        # Split along the last axis
        result[name] = value[..., start_idx : start_idx + size]

        start_idx += size

    return result


def split_multidiscrete_value(
    value: np.ndarray,
    original_spaces: dict[str, gym.Space[Any]] | gym.spaces.Dict,
) -> dict[str, np.ndarray]:
    """
    Split a MultiDiscrete space value back into original Discrete/MultiDiscrete spaces.

    Parameters
    ----------
    value : np.ndarray
        The value to split.
    original_spaces : dict[str, Discrete | MultiDiscrete]
        Dictionary mapping names to the original Discrete or MultiDiscrete spaces.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping names to split values.
    """
    result = {}
    start_idx = 0

    for name, space in original_spaces.items():
        if isinstance(space, Discrete):
            # Extract a single value
            result[name] = value[..., start_idx]
            start_idx += 1
        elif isinstance(space, MultiDiscrete):
            # Extract multiple values
            size = len(space.nvec)
            result[name] = value[..., start_idx : start_idx + size]
            start_idx += size
        else:
            raise TypeError(
                f"Expected Discrete or MultiDiscrete space for {name}, got {type(space)}"
            )

    return result


def split_value(
    value: NDArray[Any], original_spaces: dict[str, gym.Space[Any]] | gym.spaces.Dict
) -> dict[str, NDArray[Any]]:
    """
    Split a value from a merged space back into a dictionary of values for the original spaces.

    Parameters
    ----------
    value : np.ndarray
        The value to split.
    original_spaces : dict[str, gym.Space]
        Dictionary mapping names to the original spaces that were merged.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping names to split values.
    """
    first_space = next(iter(original_spaces.values()))
    if isinstance(first_space, Box):
        return split_box_value(value, original_spaces)
    elif isinstance(first_space, MultiBinary):
        return split_multibinary_value(value, original_spaces)
    elif isinstance(first_space, Discrete | MultiDiscrete):
        return split_multidiscrete_value(value, original_spaces)
    else:
        raise TypeError(
            f"Expected Box, MultiBinary, or Discrete | MultiDiscrete space, got {type(first_space)}"
        )


class VecMergeDictActionWrapper(VecEnvWrapper):
    """
    A vectorized wrapper for merging a dictionary of actions into a single flat action.
    All actions in the dictionary must be of compatible types.

    This wrapper presents a merged (flat) action space to the user, and automatically
    splits flat actions back into the dictionary format expected by the underlying environment.

    This wrapper uses singledispatch-based merge methods to handle different space types:
    - Box spaces: concatenated along the last axis
    - Discrete spaces: concatenated to form a MultiDiscrete space
    - MultiBinary spaces: summed dimensions

    Parameters
    ----------
    venv : VecEnv
        The vectorized environment being wrapped (must have Dict action space).
    """

    def __init__(self, venv: VecEnv):
        assert isinstance(
            venv.action_space, gym.spaces.Dict
        ), "VecMergeDictActionWrapper only supports VecEnvs with Dict action spaces."
        all_action_spaces = list(venv.action_space.spaces.values())
        assert len(all_action_spaces) > 0, "No Action Spaces to merge."

        self.unmerged_action_space = venv.action_space
        merged_action_space = merge_spaces(all_action_spaces[0], *all_action_spaces[1:])
        super().__init__(venv=venv, action_space=merged_action_space)

    def reset(self) -> VecEnvObs:
        return self.venv.reset()

    def step(
        self, actions: NDArray[Any]
    ) -> tuple[VecEnvObs, np.ndarray, np.ndarray, list[dict]]:
        return self.venv.step(actions)

    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    def step_wait(self) -> tuple[VecEnvObs, np.ndarray, np.ndarray, list[dict]]:
        return self.venv.step_wait()


try:
    from matplotlib import pyplot as plt
except ImportError:
    # matplot lib is not installed, raise a lazy error if someone tries to use the RenderImagesWrapper
    plt = None # type: ignore


class RenderImagesWrapper(VecEnvWrapper):
    """
    Renders image observations to an interactive matplotlib window. It assumes that the observations are square RGB images, and attempts to reshape any box observation to 3xLxL.

    Parameters
    ----------
    venv : VecEnv
        The vectorized environment being wrapped.
    """

    def __init__(self, venv: VecEnv):
        if plt is None:
            raise ImportError(
                "You must install matplotlib in order to use the RenderImagesWrapper."
            )
        self.image_obs : list[tuple[str, gym.spaces.Box]] = []
        self._num_envs = venv.num_envs
        assert isinstance(venv.observation_space, gym.spaces.Dict), "RenderImagesWrapper only supports VecEnvs with Dict observation spaces."
        for obs_space_name, obs_space in venv.observation_space.spaces.items():
            if isinstance(obs_space, gym.spaces.Box):
                self.image_obs.append((obs_space_name, obs_space))
        plt.ion()
        self.axs = []
        self.ims = [[] for _ in range(self._num_envs)]
        for row in range(self._num_envs):
            for col in range(len(self.image_obs)):
                # Note index starts at 1
                index = row * len(self.image_obs) + col + 1
                self.axs.append(plt.subplot(self._num_envs, len(self.image_obs), index))
                # TODO dynamically toggle greyscale
                if self.image_obs[col][1].shape[0] == 1:
                    cmap = "grey"
                else:
                    cmap = None
                default_ndarray = self.convert_to_plt_format(np.zeros(self.image_obs[col][1].shape))
                self.ims[row].append(
                    self.axs[index - 1].imshow(
                        default_ndarray, cmap=cmap, vmin=0.0, vmax=1.0
                    )
                )

        super().__init__(venv=venv)

    def convert_to_plt_format(self, obs: np.ndarray) -> np.ndarray:
        """
        Convert to a format supported by matplotlib. (e.g. (W,H), (W,H,3), and (W,H,4)). No Chanels or Chanels last, from Chanels first.

        Parameters
        ----------
        obs : np.ndarray
            The observation to convert.

        Returns
        -------
        np.ndarray
            The converted observation.
        """
        if obs.shape[0] == 1:
            return obs.squeeze()
        else:
            return obs.transpose(1, 2, 0)

    def close(self):
        super().close()
        plt.ioff()
        plt.show()

    def update_images(self, obs: VecEnvObs) -> VecEnvObs:
        """
        Updates the images in the plt window with the given observations.

        Parameters
        ----------
        obs: dict[str, np.ndarray]
            Maps the names of the observations to the observation data.

        Returns
        -------
        dict[str, np.ndarray]
            The original observation.
        """
        assert isinstance(obs, dict), f"Expected Observations to be a dict but got {type(obs)}"
        for col, (image_obs_name, obs_space) in enumerate(self.image_obs):
            temp_obs = np.clip(obs[image_obs_name], 0.0, 1.0)
            # yoink out the batch dim at the front of the buffer
            for row in range(temp_obs.shape[0]):
                new_data = self.convert_to_plt_format(temp_obs[row])
                self.ims[row][col].set_data(new_data)
        plt.pause(0.001)

        return obs

    def reset(self) -> VecEnvObs:
        return self.update_images(self.venv.reset())

    def step(
        self, actions: np.ndarray
    ) -> tuple[VecEnvObs, np.ndarray, np.ndarray, list[dict]]:

        obs, rewards, dones, infos = self.venv.step(actions)
        return self.update_images(obs), rewards, dones, infos

    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    def step_wait(self) -> tuple[VecEnvObs, np.ndarray, np.ndarray, list[dict]]:
        obs, rewards, dones, infos = self.venv.step_wait()
        return self.update_images(obs), rewards, dones, infos
