# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""
RLlib new-API-stack offline data: Parquet episodes plus a space sidecar.

RLlib's offline algorithms (BC, MARWIL) read Ray Data sources containing
msgpack-serialized :class:`~ray.rllib.env.single_agent_episode.SingleAgentEpisode`
states. This module writes that layout and stores the original and training
observation spaces so training does not need a live environment.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, SupportsFloat

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces.utils import flatten, flatten_space

if TYPE_CHECKING:
    from ray.rllib.env.single_agent_episode import SingleAgentEpisode


logger = logging.getLogger(__name__)

MANIFEST_FORMAT_VERSION = 1
MANIFEST_FILE_NAME = "schola-offline-manifest.json"
DEFAULT_EPISODES_PER_SHARD = 64


def get_training_observation_space(
    observation_space: gym.Space[Any],
) -> gym.Space[Any]:
    """
    Return the observation space an offline algorithm must be configured with.

    Composite spaces are flattened, because :func:`build_rllib_episode`
    flattens the observations themselves. Keep the original space for ONNX export
    so Unreal still sees one input per sensor.
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space
    return flatten_space(observation_space)


def _maybe_flatten_observation(
    observation_space: gym.Space[Any], observation: Any
) -> Any:
    """Flatten a composite observation; leave a Box sample unchanged."""
    training_space = get_training_observation_space(observation_space)
    if training_space is observation_space:
        return observation
    return flatten(observation_space, observation)


def build_rllib_episode(
    observations: Sequence[Any],
    actions: Sequence[Any],
    rewards: Sequence[SupportsFloat],
    observation_space: gym.Space[Any],
    action_space: gym.Space[Any],
    *,
    terminated: bool,
    truncated: bool,
) -> "SingleAgentEpisode":
    """
    Build an RLlib ``SingleAgentEpisode`` from per-timestep samples.

    Composite observations are flattened here rather than by a ConnectorV2 piece.
    RLlib prepends custom learner connectors ahead of
    ``AddObservationsFromEpisodesToBatch``, so a connector would run before the
    observations were pulled out of the episodes and would never see them.
    Flattening up front stays consistent with the online path because gymnasium's
    ``flatten`` and RLlib's ``flatten_inputs_to_1d_tensor`` produce identical
    output, and the latter is what :mod:`schola.rllib.export` uses.

    Parameters
    ----------
    observations : sequence
        Reset observation plus one observation per action.
    actions : sequence
        One action per environment step.
    rewards : sequence
        One reward per environment step.
    observation_space : gymnasium.Space
        Per-agent observation space as recorded (possibly composite).
    action_space : gymnasium.Space
        Per-agent action space as recorded.
    terminated : bool
        Whether the episode ended on a terminal state.
    truncated : bool
        Whether the episode was cut short (time limit or session end).

    Returns
    -------
    ray.rllib.env.single_agent_episode.SingleAgentEpisode
        Episode ready to be serialized for RLlib's offline data pipeline.
    """
    from ray.rllib.env.single_agent_episode import SingleAgentEpisode

    if len(observations) != len(actions) + 1:
        raise ValueError(
            "Episodes store the reset observation plus one observation per action; "
            f"got {len(observations)} observations and {len(actions)} actions."
        )
    if len(actions) != len(rewards):
        raise ValueError(
            "Each action must have a reward; "
            f"got {len(actions)} actions and {len(rewards)} rewards."
        )

    episode_observation_space = get_training_observation_space(observation_space)
    flat_observations = [
        _maybe_flatten_observation(observation_space, observation)
        for observation in observations
    ]
    reward_list: list[SupportsFloat] = [float(reward) for reward in rewards]

    return SingleAgentEpisode(
        observations=flat_observations,
        actions=list(actions),
        rewards=reward_list,
        observation_space=episode_observation_space,
        action_space=action_space,
        terminated=terminated,
        truncated=truncated,
        len_lookback_buffer=0,
    )


def _encode_space(space: gym.Space[Any]) -> dict[str, Any]:
    """Encode a Gymnasium space as JSON-friendly nested dicts."""
    if isinstance(space, spaces.Box):
        return {
            "type": "Box",
            "low": np.asarray(space.low).tolist(),
            "high": np.asarray(space.high).tolist(),
            "shape": list(space.shape),
            "dtype": np.dtype(space.dtype).str,
        }
    if isinstance(space, spaces.Discrete):
        return {
            "type": "Discrete",
            "n": int(space.n),
            "start": int(space.start),
        }
    if isinstance(space, spaces.MultiDiscrete):
        return {
            "type": "MultiDiscrete",
            "nvec": np.asarray(space.nvec).tolist(),
            "dtype": np.dtype(space.dtype).str,
        }
    if isinstance(space, spaces.MultiBinary):
        n_value: Any = space.n
        if isinstance(n_value, (int, np.integer)):
            encoded_n: int | list[int] = int(n_value)
        else:
            encoded_n = np.asarray(n_value).tolist()
        return {"type": "MultiBinary", "n": encoded_n}
    if isinstance(space, spaces.Dict):
        return {
            "type": "Dict",
            "spaces": {
                key: _encode_space(subspace) for key, subspace in space.spaces.items()
            },
        }
    if isinstance(space, spaces.Tuple):
        return {
            "type": "Tuple",
            "spaces": [_encode_space(subspace) for subspace in space.spaces],
        }
    raise TypeError(f"Cannot serialize Gymnasium space of type {type(space)!r}.")


def _decode_space(payload: Mapping[str, Any]) -> gym.Space[Any]:
    """Decode a space produced by :func:`_encode_space`."""
    space_type = payload["type"]
    if space_type == "Box":
        return spaces.Box(
            low=np.asarray(payload["low"], dtype=np.dtype(payload["dtype"])),
            high=np.asarray(payload["high"], dtype=np.dtype(payload["dtype"])),
            shape=tuple(payload["shape"]),
            dtype=np.dtype(payload["dtype"]),
        )
    if space_type == "Discrete":
        return spaces.Discrete(n=int(payload["n"]), start=int(payload["start"]))
    if space_type == "MultiDiscrete":
        return spaces.MultiDiscrete(
            nvec=np.asarray(payload["nvec"]),
            dtype=np.dtype(payload["dtype"]),
        )
    if space_type == "MultiBinary":
        return spaces.MultiBinary(n=payload["n"])
    if space_type == "Dict":
        nested = payload["spaces"]
        if not isinstance(nested, Mapping):
            raise TypeError("Dict space payload must map names to subspaces.")
        return spaces.Dict(
            {key: _decode_space(subspace) for key, subspace in nested.items()}
        )
    if space_type == "Tuple":
        nested_list = payload["spaces"]
        if not isinstance(nested_list, Sequence) or isinstance(
            nested_list, (str, bytes)
        ):
            raise TypeError("Tuple space payload must be a sequence of subspaces.")
        return spaces.Tuple(tuple(_decode_space(subspace) for subspace in nested_list))
    raise TypeError(f"Unknown serialized Gymnasium space type {space_type!r}.")


def space_to_json(space: gym.Space[Any]) -> dict[str, Any]:
    """Serialize a Gymnasium space for the offline manifest sidecar."""
    to_json = getattr(space, "to_json", None)
    if callable(to_json):
        return {"format": "gymnasium", "space": to_json()}
    return {"format": "schola", "space": _encode_space(space)}


def space_from_json(payload: Mapping[str, Any]) -> gym.Space[Any]:
    """Deserialize a space written by :func:`space_to_json`."""
    fmt = payload.get("format", "schola")
    space_payload = payload.get("space", payload)
    if fmt == "gymnasium":
        from_json = getattr(spaces, "from_json", None)
        if from_json is None:
            from gymnasium.spaces.utils import from_json as from_json
        return from_json(space_payload)
    if not isinstance(space_payload, Mapping):
        raise TypeError("Schola space payload must be a mapping.")
    return _decode_space(space_payload)


def _serialize_episodes(
    episodes: Iterable["SingleAgentEpisode"],
) -> Iterator[Mapping[str, bytes]]:
    """Yield RLlib's Parquet rows without retaining the complete dataset."""
    import msgpack
    # Optional extra (schola[offline]); the package ships no type stubs.
    import msgpack_numpy  # pyright: ignore[reportMissingImports]

    for episode in episodes:
        packed = msgpack.packb(
            episode.get_state(), default=msgpack_numpy.encode
        )
        if packed is None:
            raise RuntimeError("Failed to serialize RLlib episode state.")
        yield {"item": packed}


def _batched(
    items: Iterable[Mapping[str, bytes]], size: int
) -> Iterator[list[Mapping[str, bytes]]]:
    """Yield bounded batches from *items*."""
    batch: list[Mapping[str, bytes]] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _write_episode_shards(
    episodes: Iterable["SingleAgentEpisode"],
    output_dir: Path,
    *,
    episodes_per_shard: int,
) -> int:
    """Write episode states to independently readable Parquet shard files."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if episodes_per_shard < 1:
        raise ValueError("episodes_per_shard must be at least one.")

    output_dir.mkdir(parents=True, exist_ok=False)
    shard_count = 0
    episode_count = 0
    for shard_count, batch in enumerate(
        _batched(_serialize_episodes(episodes), episodes_per_shard)
    ):
        episode_count += len(batch)
        pq.write_table(
            pa.Table.from_pylist(batch),
            output_dir / f"part-{shard_count:05d}.parquet",
        )
    if episode_count == 0:
        raise ValueError(
            "No episodes to write. The collection session recorded nothing "
            + "to learn from."
        )
    return episode_count


def _discard_staging_directory(staging_dir: Path) -> None:
    """Remove a directory created exclusively for an unfinished write."""
    shutil.rmtree(staging_dir, ignore_errors=True)


def write_episodes_as_parquet(
    episodes: Iterable["SingleAgentEpisode"],
    output_dir: Path,
    *,
    episodes_per_shard: int = DEFAULT_EPISODES_PER_SHARD,
) -> Path:
    """
    Serialize RLlib episodes to the Parquet layout expected by ``input_read_episodes``.

    Parameters
    ----------
    episodes : iterable of SingleAgentEpisode
        Episodes to write.
    output_dir : pathlib.Path
        Destination directory. It must not already exist.
    episodes_per_shard : int, optional
        Maximum number of episodes in each Parquet file.

    Returns
    -------
    pathlib.Path
        *output_dir*, for convenience.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to replace existing dataset directory {output_dir}."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.parent / f".{output_dir.name}.{uuid.uuid4().hex}.staging"
    try:
        _write_episode_shards(
            episodes, staging_dir, episodes_per_shard=episodes_per_shard
        )
        os.replace(staging_dir, output_dir)
    except Exception:
        _discard_staging_directory(staging_dir)
        raise
    return output_dir


def write_offline_dataset(
    episodes: Sequence["SingleAgentEpisode"],
    output_dir: Path,
    observation_space: gym.Space[Any],
    action_space: gym.Space[Any],
    *,
    episodes_per_shard: int = DEFAULT_EPISODES_PER_SHARD,
) -> Path:
    """
    Write RLlib Parquet shards and a space sidecar for later offline training.

    Parameters
    ----------
    episodes : sequence of SingleAgentEpisode
        Completed episodes to persist.
    output_dir : pathlib.Path
        Destination directory. It must not already exist.
    observation_space : gymnasium.Space
        Original (possibly composite) observation space, kept for ONNX export.
    action_space : gymnasium.Space
        Action space recorded with the episodes.
    episodes_per_shard : int, optional
        Maximum number of episodes in each Parquet file.

    Returns
    -------
    pathlib.Path
        *output_dir*, for convenience.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to replace existing dataset directory {output_dir}."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.parent / f".{output_dir.name}.{uuid.uuid4().hex}.staging"
    training_observation_space = get_training_observation_space(observation_space)
    try:
        episode_count = _write_episode_shards(
            episodes, staging_dir, episodes_per_shard=episodes_per_shard
        )
        manifest = {
            "format_version": MANIFEST_FORMAT_VERSION,
            "observation_space": space_to_json(observation_space),
            "training_observation_space": space_to_json(training_observation_space),
            "action_space": space_to_json(action_space),
            "total_episodes": episode_count,
            "total_steps": sum(len(episode) for episode in episodes),
        }
        with (staging_dir / MANIFEST_FILE_NAME).open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(staging_dir, output_dir)
    except Exception:
        _discard_staging_directory(staging_dir)
        raise
    logger.info(
        "Wrote %s RLlib episodes (%s steps) to %s",
        episode_count,
        manifest["total_steps"],
        output_dir,
    )
    return output_dir


def load_offline_dataset(
    input_dir: Path,
) -> tuple[Path, gym.Space[Any], gym.Space[Any], gym.Space[Any]]:
    """
    Load spaces from an offline dataset written by :func:`write_offline_dataset`.

    Parameters
    ----------
    input_dir : pathlib.Path
        Directory containing Parquet shards and ``schola-offline-manifest.json``.

    Returns
    -------
    tuple
        ``(parquet_dir, training_observation_space, observation_space, action_space)``.

    Raises
    ------
    FileNotFoundError
        If the directory, manifest, or Parquet shards are missing.
    ValueError
        If the manifest is not a valid Schola offline dataset.
    """
    input_dir = Path(input_dir)
    manifest_path = input_dir / MANIFEST_FILE_NAME
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Offline dataset directory does not exist: {input_dir}")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Offline dataset is missing {MANIFEST_FILE_NAME}: {input_dir}"
        )
    if not any(input_dir.glob("*.parquet")):
        raise FileNotFoundError(
            f"Offline dataset contains no Parquet shards: {input_dir}"
        )
    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"Offline dataset manifest is not a mapping: {manifest_path}")
    if manifest.get("format_version") != MANIFEST_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported offline dataset format in {manifest_path}: "
            f"{manifest.get('format_version')!r}"
        )
    observation_space = space_from_json(manifest["observation_space"])
    action_space = space_from_json(manifest["action_space"])
    if "training_observation_space" in manifest:
        training_observation_space = space_from_json(
            manifest["training_observation_space"]
        )
    else:
        training_observation_space = get_training_observation_space(observation_space)
    return (
        input_dir,
        training_observation_space,
        observation_space,
        action_space,
    )
