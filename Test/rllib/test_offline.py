# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for converting Minari demonstrations into RLlib offline data."""

import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces
from types import SimpleNamespace
import sys

from schola.rllib.checkpoint import load_rl_module_from_algorithm_checkpoint
from schola.rllib.offline import (
    MANIFEST_FILE_NAME,
    convert_minari_dataset,
    get_training_observation_space,
    minari_episode_to_rllib,
    structure_of_arrays_to_list,
    write_episodes_as_parquet,
)


DICT_OBSERVATION_SPACE = spaces.Dict(
    {
        "RelativeDirections": spaces.Box(-1.0, 1.0, (6,), np.float32),
        "KeyCaptured": spaces.MultiBinary(1),
    }
)
MULTI_DISCRETE_ACTION_SPACE = spaces.MultiDiscrete([2, 3])


class _FakeMinariEpisode:
    """Stands in for ``minari.EpisodeData``, which stores data column-wise."""

    def __init__(self, observations, actions, rewards, terminations, truncations):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.terminations = terminations
        self.truncations = truncations


def make_episode(num_steps=4):
    return _FakeMinariEpisode(
        observations={
            # One more observation than actions: the reset obs plus one per step.
            "RelativeDirections": np.arange(
                (num_steps + 1) * 6, dtype=np.float32
            ).reshape(num_steps + 1, 6),
            "KeyCaptured": np.zeros((num_steps + 1, 1), dtype=np.int8),
        },
        actions=np.tile(np.array([1, 2], dtype=np.int64), (num_steps, 1)),
        rewards=np.ones(num_steps, dtype=np.float32),
        terminations=np.array([False] * (num_steps - 1) + [True]),
        truncations=np.zeros(num_steps, dtype=bool),
    )


def test_structure_of_arrays_to_list_splits_dict_columns():
    """Minari stores Dict spaces column-wise; RLlib needs one sample per step."""
    data = {
        "RelativeDirections": np.arange(12, dtype=np.float32).reshape(2, 6),
        "KeyCaptured": np.array([[0], [1]], dtype=np.int8),
    }
    result = structure_of_arrays_to_list(DICT_OBSERVATION_SPACE, data, 2)

    assert len(result) == 2
    assert set(result[0]) == {"RelativeDirections", "KeyCaptured"}
    np.testing.assert_array_equal(result[1]["KeyCaptured"], np.array([1]))
    np.testing.assert_array_equal(
        result[1]["RelativeDirections"], np.arange(6, 12, dtype=np.float32)
    )


def test_structure_of_arrays_to_list_handles_tuple_spaces():
    space = spaces.Tuple(
        (spaces.Box(-1.0, 1.0, (2,), np.float32), spaces.Discrete(3))
    )
    data = [np.zeros((2, 2), dtype=np.float32), np.array([1, 2])]
    result = structure_of_arrays_to_list(space, data, 2)

    assert len(result) == 2
    assert isinstance(result[0], tuple)
    assert result[1][1] == 2


def test_get_training_observation_space_leaves_box_untouched():
    """A Box observation is already flat, so it must pass through unchanged."""
    box = spaces.Box(-1.0, 1.0, (7,), np.float32)
    assert get_training_observation_space(box) is box


def test_get_training_observation_space_flattens_dict():
    flat = get_training_observation_space(DICT_OBSERVATION_SPACE)
    assert isinstance(flat, spaces.Box)
    assert flat.shape == (7,)


def test_flattening_matches_rllib_inference_flattening():
    """Guards the invariant that makes conversion-time flattening safe.

    Observations are flattened when the dataset is converted, but the exported
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
        {key: torch.as_tensor(np.asarray([value])) for key, value in observation.items()},
        DICT_OBSERVATION_SPACE.spaces,
    )

    inference_np = np.asarray(inference_time)
    if inference_np.ndim > 1:
        inference_np = inference_np[0]
    np.testing.assert_allclose(np.asarray(training_time), inference_np)


def test_minari_episode_to_rllib_flattens_dict_observations():
    episode = minari_episode_to_rllib(
        make_episode(), DICT_OBSERVATION_SPACE, MULTI_DISCRETE_ACTION_SPACE
    )

    assert len(episode) == 4
    assert episode.is_terminated
    assert not episode.is_truncated
    observation_space = episode.observation_space
    assert observation_space is not None
    assert observation_space.shape == (7,)
    assert np.asarray(episode.get_observations(0)).shape == (7,)


def test_minari_episode_to_rllib_keeps_action_space():
    episode = minari_episode_to_rllib(
        make_episode(), DICT_OBSERVATION_SPACE, MULTI_DISCRETE_ACTION_SPACE
    )
    assert episode.action_space == MULTI_DISCRETE_ACTION_SPACE
    np.testing.assert_array_equal(np.asarray(episode.get_actions(0)), [1, 2])


def test_minari_episode_to_rllib_records_one_more_observation_than_actions():
    """Off-by-one here silently misaligns every observation/action pair."""
    episode = minari_episode_to_rllib(
        make_episode(num_steps=6), DICT_OBSERVATION_SPACE, MULTI_DISCRETE_ACTION_SPACE
    )
    assert len(episode.get_actions()) == 6
    assert len(episode.get_observations()) == 7


def test_minari_episode_to_rllib_box_observations_are_not_flattened():
    box_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
    raw = _FakeMinariEpisode(
        observations=np.zeros((3, 3), dtype=np.float32),
        actions=np.zeros((2, 2), dtype=np.int64),
        rewards=np.zeros(2, dtype=np.float32),
        terminations=np.array([False, True]),
        truncations=np.array([False, False]),
    )
    episode = minari_episode_to_rllib(raw, box_space, MULTI_DISCRETE_ACTION_SPACE)
    assert episode.observation_space is box_space


def test_write_episodes_as_parquet_rejects_empty_dataset(tmp_path):
    """An empty dataset must fail loudly rather than train on nothing."""
    with pytest.raises(ValueError, match="No episodes to convert"):
        write_episodes_as_parquet([], tmp_path / "out")


def test_load_module_from_checkpoint_reports_missing_module(tmp_path):
    checkpoint = tmp_path / "checkpoint_000000"
    checkpoint.mkdir()
    with pytest.raises(
        FileNotFoundError, match="No RLModule checkpoint directory found"
    ):
        load_rl_module_from_algorithm_checkpoint(checkpoint)


class _FakeMinariDataset:
    """Small local-dataset stand-in used to test conversion cache behavior."""

    observation_space = DICT_OBSERVATION_SPACE
    action_space = MULTI_DISCRETE_ACTION_SPACE

    def __init__(self, episodes):
        self._episodes = episodes
        self.total_episodes = len(episodes)
        self.total_steps = sum(
            len(np.asarray(episode.rewards))
            for episode in episodes
            if hasattr(episode, "rewards")
        )
        self.iteration_count = 0

    def iterate_episodes(self):
        self.iteration_count += 1
        return iter(self._episodes)


def _install_fake_minari(monkeypatch, dataset):
    monkeypatch.setitem(
        sys.modules, "minari", SimpleNamespace(load_dataset=lambda _dataset_id: dataset)
    )


def test_write_episodes_as_parquet_streams_multiple_shards(tmp_path):
    pytest.importorskip("pyarrow")
    output = write_episodes_as_parquet(
        [
            minari_episode_to_rllib(
                make_episode(), DICT_OBSERVATION_SPACE, MULTI_DISCRETE_ACTION_SPACE
            )
            for _ in range(5)
        ],
        tmp_path / "episodes",
        episodes_per_shard=2,
    )

    assert len(list(output.glob("*.parquet"))) == 3


def test_parquet_shard_round_trips_msgpack_episode_state(tmp_path):
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")
    msgpack = pytest.importorskip("msgpack")
    msgpack_numpy = pytest.importorskip("msgpack_numpy")
    episode = minari_episode_to_rllib(
        make_episode(), DICT_OBSERVATION_SPACE, MULTI_DISCRETE_ACTION_SPACE
    )

    output = write_episodes_as_parquet([episode], tmp_path / "episodes")
    encoded = pyarrow_parquet.read_table(next(output.glob("*.parquet")))[
        "item"
    ][0].as_py()
    state = msgpack.unpackb(encoded, object_hook=msgpack_numpy.decode)

    assert state.keys() == episode.get_state().keys()


def test_conversion_reuses_matching_owned_cache(monkeypatch, tmp_path):
    pytest.importorskip("pyarrow")
    dataset = _FakeMinariDataset([make_episode()])
    _install_fake_minari(monkeypatch, dataset)

    first = convert_minari_dataset("demo-v0", tmp_path / "cache")
    second = convert_minari_dataset("demo-v0", tmp_path / "cache")

    assert first[0] == second[0]
    assert dataset.iteration_count == 1
    assert (first[0] / MANIFEST_FILE_NAME).is_file()


def test_conversion_preserves_unowned_cache_root_contents(monkeypatch, tmp_path):
    pytest.importorskip("pyarrow")
    dataset = _FakeMinariDataset([make_episode()])
    _install_fake_minari(monkeypatch, dataset)
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    keep_file = cache_root / "keep.txt"
    keep_file.write_text("do not delete", encoding="utf-8")

    output, *_ = convert_minari_dataset("demo-v0", cache_root)

    assert keep_file.read_text(encoding="utf-8") == "do not delete"
    assert output.parent == cache_root


def test_failed_conversion_leaves_no_partial_cache(monkeypatch, tmp_path):
    pytest.importorskip("pyarrow")
    dataset = _FakeMinariDataset([make_episode(), object()])
    _install_fake_minari(monkeypatch, dataset)
    cache_root = tmp_path / "cache"

    with pytest.raises(AttributeError):
        convert_minari_dataset("broken-v0", cache_root, episodes_per_shard=1)

    assert not list(cache_root.glob("*.staging"))
    assert not list(path for path in cache_root.iterdir() if path.is_dir())
