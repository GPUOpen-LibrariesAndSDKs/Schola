# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for RLlib offline Parquet datasets and space sidecars."""

import json

import numpy as np
import pytest
from gymnasium import spaces

from schola.rllib.checkpoint import load_rl_module_from_algorithm_checkpoint
from schola.rllib.offline import (
    MANIFEST_FILE_NAME,
    build_rllib_episode,
    get_training_observation_space,
    load_offline_dataset,
    space_from_json,
    space_to_json,
    write_episodes_as_parquet,
    write_offline_dataset,
)

DICT_OBSERVATION_SPACE = spaces.Dict(
    {
        "RelativeDirections": spaces.Box(-1.0, 1.0, (6,), np.float32),
        "KeyCaptured": spaces.MultiBinary(1),
    }
)
MULTI_DISCRETE_ACTION_SPACE = spaces.MultiDiscrete([2, 3])


def make_episode(num_steps: int = 4):
    observations = [
        {
            "RelativeDirections": np.arange(i * 6, (i + 1) * 6, dtype=np.float32),
            "KeyCaptured": np.zeros((1,), dtype=np.int8),
        }
        for i in range(num_steps + 1)
    ]
    actions = [np.array([1, 2], dtype=np.int64) for _ in range(num_steps)]
    rewards = [1.0] * num_steps
    return build_rllib_episode(
        observations,
        actions,
        rewards,
        DICT_OBSERVATION_SPACE,
        MULTI_DISCRETE_ACTION_SPACE,
        terminated=True,
        truncated=False,
    )


def test_get_training_observation_space_leaves_box_untouched():
    """A Box observation is already flat, so it must pass through unchanged."""
    box = spaces.Box(-1.0, 1.0, (7,), np.float32)
    assert get_training_observation_space(box) is box


def test_get_training_observation_space_flattens_dict():
    flat = get_training_observation_space(DICT_OBSERVATION_SPACE)
    assert isinstance(flat, spaces.Box)
    assert flat.shape == (7,)


def test_flattening_matches_rllib_inference_flattening():
    """Guards the invariant that makes collection-time flattening safe.

    Observations are flattened when the dataset is written, but the exported
    ONNX model flattens per-sensor inputs at inference with RLlib's
    ``flatten_inputs_to_1d_tensor``. If those two ever disagree the policy would
    train on a different feature ordering than it sees in Unreal, and nothing
    else in the pipeline would notice.
    """
    import torch
    from gymnasium.spaces.utils import flatten
    from ray.rllib.utils.torch_utils import flatten_inputs_to_1d_tensor

    observation = {
        "RelativeDirections": np.arange(6, dtype=np.float32),
        "KeyCaptured": np.array([1], dtype=np.int8),
    }

    training_time = flatten(DICT_OBSERVATION_SPACE, observation)
    inference_time = flatten_inputs_to_1d_tensor(
        {
            key: torch.as_tensor(np.asarray([value]))
            for key, value in observation.items()
        },
        DICT_OBSERVATION_SPACE.spaces,
    )

    inference_np = np.asarray(inference_time)
    if inference_np.ndim > 1:
        inference_np = inference_np[0]
    np.testing.assert_allclose(np.asarray(training_time), inference_np)


def test_build_rllib_episode_flattens_dict_observations():
    episode = make_episode()

    assert len(episode) == 4
    assert episode.is_terminated
    assert not episode.is_truncated
    observation_space = episode.observation_space
    assert observation_space is not None
    assert observation_space.shape == (7,)
    assert np.asarray(episode.get_observations(0)).shape == (7,)


def test_build_rllib_episode_keeps_action_space():
    episode = make_episode()
    assert episode.action_space == MULTI_DISCRETE_ACTION_SPACE
    np.testing.assert_array_equal(np.asarray(episode.get_actions(0)), [1, 2])


def test_build_rllib_episode_records_one_more_observation_than_actions():
    """Off-by-one here silently misaligns every observation/action pair."""
    episode = make_episode(num_steps=6)
    assert len(episode.get_actions()) == 6
    assert len(episode.get_observations()) == 7


def test_build_rllib_episode_box_observations_are_not_flattened():
    box_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
    episode = build_rllib_episode(
        [np.zeros(3, dtype=np.float32) for _ in range(3)],
        [np.zeros(2, dtype=np.int64) for _ in range(2)],
        [0.0, 0.0],
        box_space,
        MULTI_DISCRETE_ACTION_SPACE,
        terminated=True,
        truncated=False,
    )
    assert episode.observation_space is box_space


def test_space_json_round_trips_dict_and_multidiscrete():
    encoded_obs = space_to_json(DICT_OBSERVATION_SPACE)
    encoded_act = space_to_json(MULTI_DISCRETE_ACTION_SPACE)
    restored_obs = space_from_json(encoded_obs)
    restored_act = space_from_json(encoded_act)

    assert restored_obs == DICT_OBSERVATION_SPACE
    assert restored_act == MULTI_DISCRETE_ACTION_SPACE


@pytest.mark.parametrize(
    "space",
    [
        spaces.Box(-1.0, 1.0, (6,), np.float32),
        spaces.Box(-1.0, 1.0, (2, 3), np.float64),
        spaces.Discrete(5),
        spaces.MultiBinary(4),
        spaces.MultiDiscrete([2, 3]),
        spaces.Text(max_length=10),
        get_training_observation_space(DICT_OBSERVATION_SPACE),
    ],
)
def test_space_json_round_trips_every_space_unreal_can_send(space):
    """The manifest must express whatever Schola's protocol can deliver.

    ``Text`` is the interesting case: it is reachable through Schola's protobuf
    layer, so a manifest encoder that cannot represent it would fail only once
    somebody recorded a text observation.
    """
    encoded = space_to_json(space)
    assert json.loads(json.dumps(encoded)) == encoded
    assert space_from_json(encoded) == space


def test_space_json_reads_legacy_manifest_payload():
    """Demonstrations recorded before the protobuf manifest must still load."""
    legacy = {
        "format": "schola",
        "space": {
            "type": "Box",
            "low": [-1.0, -1.0],
            "high": [1.0, 1.0],
            "shape": [2],
            "dtype": "<f4",
        },
    }

    assert space_from_json(legacy) == spaces.Box(-1.0, 1.0, (2,), np.float32)


def test_write_episodes_as_parquet_rejects_empty_dataset(tmp_path):
    """An empty dataset must fail loudly rather than train on nothing."""
    with pytest.raises(ValueError, match="No episodes to write"):
        write_episodes_as_parquet([], tmp_path / "out")


def test_load_module_from_checkpoint_reports_missing_module(tmp_path):
    checkpoint = tmp_path / "checkpoint_000000"
    checkpoint.mkdir()
    with pytest.raises(
        FileNotFoundError, match="No RLModule checkpoint directory found"
    ):
        load_rl_module_from_algorithm_checkpoint(checkpoint)


def test_writing_restores_ray_data_log_level(tmp_path):
    """Quieting Ray Data during a write must not leak into the rest of the run."""
    import logging

    pytest.importorskip("ray")
    ray_data_logger = logging.getLogger("ray.data")
    before = ray_data_logger.level

    write_episodes_as_parquet([make_episode()], tmp_path / "episodes")

    assert ray_data_logger.level == before


def test_write_episodes_as_parquet_streams_multiple_shards(tmp_path):
    pytest.importorskip("ray")
    output = write_episodes_as_parquet(
        [make_episode() for _ in range(5)],
        tmp_path / "episodes",
        episodes_per_shard=2,
    )

    assert len(list(output.glob("part-*"))) == 3
    assert len(list(output.rglob("*.parquet"))) >= 3


def test_parquet_shard_round_trips_msgpack_episode_state(tmp_path):
    import msgpack
    import msgpack_numpy
    import ray
    from ray.rllib.env.single_agent_episode import SingleAgentEpisode

    pytest.importorskip("ray")
    episode = make_episode()

    output = write_episodes_as_parquet([episode], tmp_path / "episodes")
    if not ray.is_initialized():
        ray.init(num_cpus=1, include_dashboard=False, ignore_reinit_error=True)
    row = ray.data.read_parquet(str(output)).take(1)[0]
    restored = SingleAgentEpisode.from_state(
        msgpack.unpackb(row["item"], object_hook=msgpack_numpy.decode)
    )

    assert restored.get_state().keys() == episode.get_state().keys()


def test_write_offline_dataset_sidecar_reloads_spaces(tmp_path):
    pytest.importorskip("ray")
    output = write_offline_dataset(
        [make_episode()],
        tmp_path / "demos",
        DICT_OBSERVATION_SPACE,
        MULTI_DISCRETE_ACTION_SPACE,
    )

    data_path, training_space, observation_space, action_space = load_offline_dataset(
        output
    )

    assert data_path == output.resolve()
    assert (output / MANIFEST_FILE_NAME).is_file()
    assert observation_space == DICT_OBSERVATION_SPACE
    assert action_space == MULTI_DISCRETE_ACTION_SPACE
    assert training_space.shape == (7,)


def test_load_offline_dataset_resolves_relative_path(tmp_path, monkeypatch):
    pytest.importorskip("ray")
    write_offline_dataset(
        [make_episode()],
        tmp_path / "demos",
        DICT_OBSERVATION_SPACE,
        MULTI_DISCRETE_ACTION_SPACE,
    )
    monkeypatch.chdir(tmp_path)

    data_path, _, _, _ = load_offline_dataset("demos")

    assert data_path.is_absolute()
    assert data_path == (tmp_path / "demos").resolve()


def test_load_offline_dataset_requires_manifest(tmp_path):
    dataset_dir = tmp_path / "demos"
    dataset_dir.mkdir()
    (dataset_dir / "part-00000.parquet").write_bytes(b"not-a-real-parquet")
    with pytest.raises(FileNotFoundError, match="schola-offline-manifest"):
        load_offline_dataset(dataset_dir)
