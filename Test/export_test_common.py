# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Spaces and helpers shared by the SB3 and RLlib ONNX/IREE export tests."""

from typing import Any

import gymnasium as gym
import numpy as np

box_obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
discrete_obs_space = gym.spaces.Discrete(2)
binary_obs_space = gym.spaces.MultiBinary(2)
multi_discrete_obs_space = gym.spaces.MultiDiscrete([2, 3])
dict_obs_space = gym.spaces.Dict(
    {
        "box": box_obs_space,
        "discrete": discrete_obs_space,
        "binary": binary_obs_space,
        "multi_discrete": multi_discrete_obs_space,
    }
)
# we use slightly altered action spaces, so that no-ops can't work by accident.
box_action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
# Bounds away from zero so an unsquashed, unscaled actor mean fails contains().
offset_box_action_space = gym.spaces.Box(low=12, high=14, shape=(3,), dtype=np.float32)
discrete_action_space = gym.spaces.Discrete(3)
binary_action_space = gym.spaces.MultiBinary(3)
multi_discrete_action_space = gym.spaces.MultiDiscrete([3, 4])


def batch_observation(observation: Any, observation_space: gym.Space) -> dict[str, np.ndarray]:
    """Batch one environment observation the way the exported model expects it."""
    if isinstance(observation_space, gym.spaces.Dict):
        return {
            key: np.expand_dims(np.asarray(observation[key]), axis=0)
            for key in observation_space.spaces
        }
    return {"obs": np.expand_dims(np.asarray(observation), axis=0)}


def unbatch_output(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.shape[:1] == (1,):
        return array[0]
    return array


def action_from_outputs(outputs: dict[str, Any], action_space: gym.Space) -> Any:
    """Build an environment action from model outputs keyed by output name."""
    output_map = {
        name: value
        for name, value in outputs.items()
        if not name.startswith("state_out_")
    }
    if isinstance(action_space, gym.spaces.Dict):
        return {
            key: unbatch_output(output_map[key]) for key in action_space.spaces
        }
    return unbatch_output(output_map["action"])


def assert_recorded_actions_match(
    reference_actions: list[Any],
    exported_actions: list[Any],
    action_space: gym.Space,
) -> None:
    """Assert the training framework's actions match the exported model's, step by step."""
    assert len(reference_actions) == len(exported_actions), (
        "Reference and exported-model evaluations recorded different numbers of actions: "
        f"{len(reference_actions)} != {len(exported_actions)}"
    )
    for step, (reference_action, exported_action) in enumerate(
        zip(reference_actions, exported_actions)
    ):
        np.testing.assert_allclose(
            np.asarray(gym.spaces.flatten(action_space, reference_action), dtype=np.float64),
            np.asarray(gym.spaces.flatten(action_space, exported_action), dtype=np.float64),
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"Action mismatch at step {step}",
        )
