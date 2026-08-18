# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
from __future__ import annotations

"""End-to-end checks that a demonstration recorded through Schola's imitation API
carries everything the offline RLlib algorithms need.

The unit tests in ``test_offline.py`` feed the converter hand-built stand-ins for
``minari.EpisodeData``, so they only prove the converter matches our reading of
the Minari layout. These tests record a dataset the way a user does -- over the
imitation gRPC connector, through ``ScholaDataCollector`` -- and then convert it,
so the spaces, the observation/action alignment and the column layout are the
real ones.
"""

import os
import subprocess
import sys
from typing import Any, cast
import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

from schola.core.protocols.protobuf.offline_grpc_protocol import GrpcImitationProtocol
from schola.core.simulators.unreal.editor_simulator import UnrealEditor
from schola.minari.datacollector import ScholaDataCollector
from schola.rllib.offline import (
    get_training_observation_space,
    minari_episode_to_rllib,
)


_DATASET_KWARGS = {
    "algorithm_name": "schola_test_expert",
    "author": "Schola CI",
    "author_email": "schola-ci@example.com",
    "code_permalink": "https://github.com/GPUOpen-LibrariesAndSDKs/Schola",
    "description": "Imitation dataset for offline training tests",
}

EPISODE_LENGTH = 6
KEY_PICKUP_STEP = 3


class _KeyAndDoorEnv(gym.Env[Any, Any]):
    """Shaped like a Schola environment: one observation per sensor, branched actions.

    Deterministic, so the recorded actions can be checked against the states they
    were taken in.
    """

    def __init__(self):
        self.observation_space = spaces.Dict(
            {
                "RelativeDirections": spaces.Box(-1.0, 1.0, (6,), np.float32),
                "KeyCaptured": spaces.MultiBinary(1),
            }
        )
        self.action_space = spaces.MultiDiscrete([2, 3])
        self._step = 0

    def _observe(self):
        return {
            "RelativeDirections": np.full(
                (6,), self._step / EPISODE_LENGTH, dtype=np.float32
            ),
            "KeyCaptured": np.array([self._step >= KEY_PICKUP_STEP], dtype=np.int8),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return self._observe(), {}

    def step(self, action):
        self._step += 1
        return self._observe(), 1.0, self._step >= EPISODE_LENGTH, False, {}


class _KeyAndDoorExpert:
    """Turns once the key is held, so actions depend on the observation."""

    def __init__(self, env: gym.Env[Any, Any]):
        self.action_space = env.action_space

    def __call__(self, observation):
        has_key = bool(observation["KeyCaptured"][0])
        return np.array([1, 2 if has_key else 1], dtype=np.int64)


@pytest.fixture
def minari_dataset_dir(tmp_path, monkeypatch):
    dataset_path = tmp_path / "minari_datasets"
    dataset_path.mkdir()
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(dataset_path))
    return dataset_path


@pytest.fixture
def recorded_dataset(make_imitation_server, minari_dataset_dir):
    """Record demonstrations over the imitation connector, as a user would."""
    port = make_imitation_server(_KeyAndDoorEnv, _KeyAndDoorExpert)
    collector = ScholaDataCollector(
        GrpcImitationProtocol(url="localhost", port=port),
        UnrealEditor(),
        seed=123,
    )
    # Two full episodes; the environment resets itself on termination.
    for _ in range(2 * EPISODE_LENGTH):
        collector.step()

    dataset = collector.create_dataset(
        "schola-imitation-v0", **cast(Any, _DATASET_KWARGS)
    )
    yield dataset
    collector.close()


def test_recorded_spaces_survive_the_round_trip(recorded_dataset):
    """The dataset must describe the spaces well enough to configure training.

    Nothing else supplies them offline: there is no environment to ask, and the
    ONNX export needs the per-sensor names to build one model input per sensor.
    """
    observation_space = recorded_dataset.observation_space
    action_space = recorded_dataset.action_space

    assert isinstance(observation_space, spaces.Dict)
    assert set(observation_space.spaces) == {"RelativeDirections", "KeyCaptured"}
    assert observation_space["RelativeDirections"].shape == (6,)
    assert observation_space["KeyCaptured"].shape == (1,)
    assert isinstance(action_space, spaces.MultiDiscrete)
    np.testing.assert_array_equal(action_space.nvec, [2, 3])

    assert get_training_observation_space(observation_space).shape == (7,)


def test_recorded_episodes_convert_to_rllib_episodes(recorded_dataset):
    """Conversion must handle the column layout Minari actually writes."""
    episodes = [
        minari_episode_to_rllib(
            episode,
            recorded_dataset.observation_space,
            recorded_dataset.action_space,
        )
        for episode in recorded_dataset.iterate_episodes()
    ]

    assert episodes, "The imitation session recorded no complete episodes"
    for episode in episodes:
        assert len(episode.get_observations()) == len(episode.get_actions()) + 1
        assert np.asarray(episode.get_observations(0)).shape == (7,)


def test_recorded_actions_stay_aligned_with_their_observations(recorded_dataset):
    """Each action must still sit against the state it was taken in.

    An off-by-one anywhere between the connector, Minari's collector and the
    converter would train the policy on the next state's label, and it would
    still look like a healthy training run.
    """
    episodes = [
        minari_episode_to_rllib(
            episode,
            recorded_dataset.observation_space,
            recorded_dataset.action_space,
        )
        for episode in recorded_dataset.iterate_episodes()
    ]

    for episode in episodes:
        observations = episode.get_observations()
        actions = episode.get_actions()
        for step, action in enumerate(actions):
            # Dict spaces flatten in sorted key order, so KeyCaptured leads.
            has_key = bool(np.asarray(observations[step])[0])
            expected_turn = 2 if has_key else 1
            assert np.asarray(action)[1] == expected_turn, (
                f"Action at step {step} does not match the observation it was "
                "recorded against"
            )


def test_expert_behaviour_is_visible_in_the_recording(recorded_dataset):
    """The recording must contain both phases, or alignment proves nothing."""
    turns = set()
    for episode in recorded_dataset.iterate_episodes():
        turns.update(int(action[1]) for action in np.asarray(episode.actions))

    assert turns == {1, 2}


def test_recorded_dataset_trains_and_exports_onnx(recorded_dataset, tmp_path):
    """Train straight from the recording, with nothing else supplied.

    Runs in a subprocess for two reasons: it needs its own Ray instance, and
    other tests in this suite hold a session-wide one; and offline training used
    to hang rather than fail when Ray was short of CPUs, so a regression there
    must time out instead of stalling the run.
    """
    checkpoint_dir = tmp_path / "run"
    checkpoint_dir.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "schola.scripts.launch",
            "rllib",
            "train",
            "bc",
            "--dataset-id",
            "schola-imitation-v0",
            "--timesteps",
            "512",
            "--export-onnx",
            "--checkpoint-dir",
            str(checkpoint_dir),
        ],
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        capture_output=True,
        # Ray logs non-ASCII characters that the console encoding cannot decode.
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )

    assert result.returncode == 0, result.stderr

    exported = list(checkpoint_dir.rglob("*.onnx"))
    assert exported, "Training finished without exporting a model"

    onnx = pytest.importorskip("onnx")
    model = onnx.load(str(exported[0]))
    # Unreal feeds inference one input per sensor, so the recorded sensor names
    # have to reach the exported model.
    assert {graph_input.name for graph_input in model.graph.input} == {
        "RelativeDirections",
        "KeyCaptured",
    }
