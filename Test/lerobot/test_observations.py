from __future__ import annotations

import numpy as np
import pytest
from gymnasium.spaces import Box, Dict

from lerobot_env_schola.config import ScholaObservationConfig
from lerobot_env_schola.observations import ObservationAdapter


@pytest.fixture
def make_adapter():
    def _make(
        observation_space: Dict,
        observation_config: ScholaObservationConfig,
        *,
        num_envs: int = 2,
    ) -> ObservationAdapter:
        return ObservationAdapter(
            source_space=observation_space,
            config=observation_config,
            num_envs=num_envs,
        )

    return _make


def test_adapter_builds_target_oriented_observations(make_adapter):
    observation_space = Dict(
        {
            "debug": Box(-1, 1, shape=(1,), dtype=np.float32),
            "front_camera": Box(0, 255, shape=(8, 8, 3), dtype=np.uint8),
            "gripper": Box(-1, 1, shape=(1,), dtype=np.float32),
            "joint_positions": Box(-1, 1, shape=(2,), dtype=np.float32),
            "joint_velocities": Box(-1, 1, shape=(2,), dtype=np.float32),
            "target": Box(-1, 1, shape=(3,), dtype=np.float32),
        }
    )
    adapter = make_adapter(
        observation_space,
        ScholaObservationConfig(
            cameras={"front": "front_camera"},
            vectors={
                "agent_pos": [
                    "joint_positions",
                    "joint_velocities",
                    "gripper",
                ]
            },
            passthrough={"environment_state": "target"},
            ignore=["debug"],
        ),
    )

    observation = {
        "debug": np.zeros((2, 1), dtype=np.float32),
        "front_camera": np.zeros((2, 8, 8, 3), dtype=np.uint8),
        "gripper": np.array([[5.0], [6.0]], dtype=np.float32),
        "joint_positions": np.array(
            [[1.0, 2.0], [3.0, 4.0]], dtype=np.float32
        ),
        "joint_velocities": np.array(
            [[0.1, 0.2], [0.3, 0.4]], dtype=np.float32
        ),
        "target": np.ones((2, 3), dtype=np.float32),
    }
    converted = adapter.convert(observation)

    assert set(converted) == {"pixels", "agent_pos", "environment_state"}
    assert set(converted["pixels"]) == {"front"}
    np.testing.assert_array_equal(
        converted["agent_pos"],
        np.array(
            [
                [1.0, 2.0, 0.1, 0.2, 5.0],
                [3.0, 4.0, 0.3, 0.4, 6.0],
            ],
            dtype=np.float32,
        ),
    )
    assert adapter.single_observation_space["agent_pos"].shape == (5,)


def test_adapter_converts_native_schola_camera_to_lerobot_format(make_adapter):
    camera_values = np.empty((3, 4, 5), dtype=np.float32)
    camera_values[0] = 0.0
    camera_values[1] = 0.5
    camera_values[2] = 1.0
    adapter = make_adapter(
        Dict(
            {
                "camera": Box(camera_values, camera_values, dtype=np.float32),
                "joints": Box(-1, 1, shape=(2,), dtype=np.float32),
            }
        ),
        ScholaObservationConfig(
            cameras={"front": "camera"},
            vectors={"agent_pos": ["joints"]},
        ),
        num_envs=1,
    )

    converted = adapter.convert(
        {
            "camera": camera_values[np.newaxis, ...],
            "joints": np.zeros((1, 2), dtype=np.float32),
        }
    )
    pixels = converted["pixels"]["front"]

    assert adapter.single_observation_space["pixels"]["front"] == Box(
        0, 255, shape=(4, 5, 3), dtype=np.uint8
    )
    assert pixels.shape == (1, 4, 5, 3)
    assert pixels.dtype == np.uint8
    np.testing.assert_array_equal(pixels[0, 0, 0], [0, 128, 255])


def test_adapter_prefers_hwc_for_ambiguous_float_image_shape(make_adapter):
    camera_values = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
    camera_values /= camera_values.max()
    adapter = make_adapter(
        Dict({"camera": Box(0.0, 1.0, shape=(4, 5, 3), dtype=np.float32)}),
        ScholaObservationConfig(cameras={"front": "camera"}),
        num_envs=1,
    )

    pixels = adapter.convert(
        {"camera": camera_values[np.newaxis, ...]}
    )["pixels"]["front"]

    assert adapter.single_observation_space["pixels"]["front"].shape == (4, 5, 3)
    assert pixels.shape == (1, 4, 5, 3)
    np.testing.assert_array_equal(
        pixels[0],
        np.rint(camera_values * 255).astype(np.uint8),
    )


def test_adapter_rejects_unaccounted_observations(make_adapter):
    observation_space = Dict(
        {
            "joints": Box(-1, 1, shape=(2,), dtype=np.float32),
            "unused": Box(-1, 1, shape=(1,), dtype=np.float32),
        }
    )

    with pytest.raises(ValueError, match="does not account for.*unused"):
        make_adapter(
            observation_space,
            ScholaObservationConfig(vectors={"agent_pos": ["joints"]}),
        )


def test_adapter_rejects_duplicate_source_ownership(make_adapter):
    observation_space = Dict(
        {"joints": Box(-1, 1, shape=(2,), dtype=np.float32)}
    )

    with pytest.raises(ValueError, match="used by both"):
        make_adapter(
            observation_space,
            ScholaObservationConfig(
                vectors={"agent_pos": ["joints"]},
                passthrough={"raw_joints": "joints"},
            ),
        )


@pytest.mark.parametrize("target_key", ["pixels", "pixels/front"])
def test_adapter_reserves_pixel_outputs_for_cameras(make_adapter, target_key):
    observation_space = Dict(
        {"joints": Box(-1, 1, shape=(2,), dtype=np.float32)}
    )

    with pytest.raises(ValueError, match="declared under cameras"):
        make_adapter(
            observation_space,
            ScholaObservationConfig(vectors={target_key: ["joints"]}),
        )


def test_adapter_rejects_unsupported_camera_dtype(make_adapter):
    observation_space = Dict(
        {"camera": Box(0, 255, shape=(8, 8, 3), dtype=np.int32)}
    )

    with pytest.raises(TypeError, match="float or uint8"):
        make_adapter(
            observation_space,
            ScholaObservationConfig(cameras={"front": "camera"}),
        )
