# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
from __future__ import annotations

"""End-to-end checks that a demonstration recorded through Schola's imitation API
carries everything the offline RLlib algorithms need.

These tests record over the imitation gRPC connector through
``RllibImitationCollector`` and write RLlib Parquet, so the spaces and
observation/action alignment are the real ones.
"""

import os
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

from schola.core.error_manager import UnrealCrashedError
from schola.core.protocols.protobuf.offline_grpc_protocol import GrpcImitationProtocol
from schola.core.simulators.unreal.editor_simulator import UnrealEditor
from schola.rllib.collector import RllibImitationCollector
from schola.rllib.offline import (
    get_training_observation_space,
    load_offline_dataset,
    write_offline_dataset,
)

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
def recorded_dataset(make_imitation_server, tmp_path):
    """Record demonstrations over the imitation connector, as a user would."""
    port = make_imitation_server(_KeyAndDoorEnv, _KeyAndDoorExpert)
    collector = RllibImitationCollector(
        GrpcImitationProtocol(url="localhost", port=port),
        UnrealEditor(),
        seed=123,
    )
    try:
        episodes = collector.collect_until_closed(max_steps=2 * EPISODE_LENGTH)
        output = write_offline_dataset(
            episodes,
            tmp_path / "demos",
            collector.observation_space,
            collector.action_space,
        )
    finally:
        collector.close()
    yield load_offline_dataset(output), episodes


def test_recorded_spaces_survive_the_round_trip(recorded_dataset):
    """The dataset must describe the spaces well enough to configure training.

    Nothing else supplies them offline: there is no environment to ask, and the
    ONNX export needs the per-sensor names to build one model input per sensor.
    """
    (_path, training_space, observation_space, action_space), _episodes = (
        recorded_dataset
    )

    assert isinstance(observation_space, spaces.Dict)
    assert set(observation_space.spaces) == {"RelativeDirections", "KeyCaptured"}
    assert observation_space["RelativeDirections"].shape == (6,)
    assert observation_space["KeyCaptured"].shape == (1,)
    assert isinstance(action_space, spaces.MultiDiscrete)
    np.testing.assert_array_equal(action_space.nvec, [2, 3])
    assert training_space.shape == (7,)
    assert get_training_observation_space(observation_space).shape == (7,)


def test_recorded_episodes_are_rllib_episodes(recorded_dataset):
    _loaded, episodes = recorded_dataset

    assert episodes, "The imitation session recorded no complete episodes"
    for episode in episodes:
        assert len(episode.get_observations()) == len(episode.get_actions()) + 1
        assert np.asarray(episode.get_observations(0)).shape == (7,)


def test_recorded_actions_stay_aligned_with_their_observations(recorded_dataset):
    """Each action must still sit against the state it was taken in.

    An off-by-one anywhere between the connector and the collector would train
    the policy on the next state's label, and it would still look like a
    healthy training run.
    """
    _loaded, episodes = recorded_dataset

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
    _loaded, episodes = recorded_dataset
    turns = set()
    for episode in episodes:
        turns.update(int(np.asarray(action)[1]) for action in episode.get_actions())

    assert turns == {1, 2}


def test_collection_stops_when_the_session_ends(make_imitation_server):
    """A dropped imitation stream is a clean end, not a failed collection."""
    port = make_imitation_server(_KeyAndDoorEnv, _KeyAndDoorExpert)
    protocol = GrpcImitationProtocol(url="localhost", port=port)
    collector = RllibImitationCollector(protocol, UnrealEditor(), seed=1)
    original_get_data = protocol.get_data
    calls = {"count": 0}

    def get_data_then_drop():
        calls["count"] += 1
        if calls["count"] > 3:
            raise UnrealCrashedError(Exception("session ended"))
        return original_get_data()

    protocol.get_data = get_data_then_drop  # type: ignore[method-assign]
    try:
        episodes = collector.collect_until_closed()
    finally:
        collector.close()

    assert episodes
    assert all(len(episode) >= 1 for episode in episodes)


def test_recorded_dataset_trains_and_exports_onnx(recorded_dataset, tmp_path):
    """Train straight from the recording, with nothing else supplied.

    Runs in a subprocess for two reasons: it needs its own Ray instance, and
    other tests in this suite hold a session-wide one; and offline training used
    to hang rather than fail when Ray was short of CPUs, so a regression there
    must time out instead of stalling the run.
    """
    (data_path, _training, _obs, _act), _episodes = recorded_dataset
    checkpoint_dir = tmp_path / "run"
    checkpoint_dir.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "schola.scripts.launch",
            "rllib",
            "bc",
            "--input",
            str(data_path),
            "--timesteps",
            "512",
            "--export-onnx",
            "--checkpoint-dir",
            str(checkpoint_dir),
        ],
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )

    assert result.returncode == 0, result.stderr

    exported = list(checkpoint_dir.rglob("*.onnx"))
    assert exported, "Training finished without exporting a model"

    onnx = pytest.importorskip("onnx")
    model = onnx.load(str(exported[0]))
    assert {graph_input.name for graph_input in model.graph.input} == {
        "RelativeDirections",
        "KeyCaptured",
    }
