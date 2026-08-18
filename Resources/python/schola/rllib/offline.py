# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Bridge Minari demonstration datasets to RLlib's new-API-stack offline RL.

RLlib's offline algorithms (BC, MARWIL) read Ray Data sources containing
msgpack-serialized :class:`~ray.rllib.env.single_agent_episode.SingleAgentEpisode`
states. Minari stores demonstrations column-wise in HDF5. This module converts
between the two so demonstrations recorded with
:class:`~schola.minari.datacollector.ScholaDataCollector` can train an RLlib
policy.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, SupportsFloat, cast

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces.utils import flatten, flatten_space

if TYPE_CHECKING:
    from ray.rllib.env.single_agent_episode import SingleAgentEpisode

logger = logging.getLogger(__name__)

CONVERSION_FORMAT_VERSION = 1
MANIFEST_FILE_NAME = "schola-offline-manifest.json"
DEFAULT_EPISODES_PER_SHARD = 64


def structure_of_arrays_to_list(
    space: gym.Space[Any], data: Any, length: int
) -> list[Any]:
    """
    Convert Minari's column-wise storage into one sample per timestep.

    Minari stores composite spaces as a mapping of arrays; RLlib episodes want a
    list of individual samples.

    Parameters
    ----------
    space : gymnasium.Space
        Space describing *data*.
    data : Any
        Column-wise data as loaded from Minari.
    length : int
        Number of timesteps to extract.

    Returns
    -------
    list
        ``length`` samples, each matching *space*.
    """
    if isinstance(space, spaces.Dict):
        columns = {
            key: structure_of_arrays_to_list(subspace, data[key], length)
            for key, subspace in space.spaces.items()
        }
        return [{key: columns[key][i] for key in columns} for i in range(length)]
    if isinstance(space, spaces.Tuple):
        columns = [
            structure_of_arrays_to_list(subspace, data[i], length)
            for i, subspace in enumerate(space.spaces)
        ]
        return [tuple(column[i] for column in columns) for i in range(length)]
    array = np.asarray(data)
    return [array[i] for i in range(length)]


def get_training_observation_space(
    observation_space: gym.Space[Any],
) -> gym.Space[Any]:
    """
    Return the observation space an offline algorithm must be configured with.

    Composite spaces are flattened, because :func:`minari_episode_to_rllib`
    flattens the observations themselves. Keep the original space for ONNX export
    so Unreal still sees one input per sensor.
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space
    return flatten_space(observation_space)


def minari_episode_to_rllib(
    episode: Any,
    observation_space: gym.Space[Any],
    action_space: gym.Space[Any],
) -> "SingleAgentEpisode":
    """
    Convert a single Minari episode into an RLlib ``SingleAgentEpisode``.

    Composite observations are flattened here rather than by a ConnectorV2 piece.
    RLlib prepends custom learner connectors ahead of
    ``AddObservationsFromEpisodesToBatch``, so a connector would run before the
    observations were pulled out of the episodes and would never see them.
    Flattening up front stays consistent with the online path because gymnasium's
    ``flatten`` and RLlib's ``flatten_inputs_to_1d_tensor`` produce identical
    output, and the latter is what :mod:`schola.rllib.export` uses.

    Parameters
    ----------
    episode : minari.EpisodeData
        Episode as yielded by ``MinariDataset.iterate_episodes()``.
    observation_space : gymnasium.Space
        Per-agent observation space recorded in the dataset.
    action_space : gymnasium.Space
        Per-agent action space recorded in the dataset.

    Returns
    -------
    ray.rllib.env.single_agent_episode.SingleAgentEpisode
        Episode ready to be serialized for RLlib's offline data pipeline.
    """
    from ray.rllib.env.single_agent_episode import SingleAgentEpisode

    num_actions = len(np.asarray(episode.rewards))
    # Episodes store the reset observation plus one observation per action.
    num_observations = num_actions + 1

    observations = structure_of_arrays_to_list(
        observation_space, episode.observations, num_observations
    )
    actions = structure_of_arrays_to_list(action_space, episode.actions, num_actions)
    # ``SingleAgentEpisode`` wants ``List[SupportsFloat]``; ``List`` is invariant,
    # so a ``list[float]`` needs this wider element-type annotation to satisfy it.
    rewards: list[SupportsFloat] = [
        float(reward) for reward in np.asarray(episode.rewards)
    ]

    episode_observation_space = get_training_observation_space(observation_space)
    if episode_observation_space is not observation_space:
        observations = [
            flatten(observation_space, observation) for observation in observations
        ]

    return SingleAgentEpisode(
        observations=observations,
        actions=actions,
        rewards=rewards,
        observation_space=episode_observation_space,
        action_space=action_space,
        terminated=bool(np.asarray(episode.terminations)[-1]),
        truncated=bool(np.asarray(episode.truncations)[-1]),
        len_lookback_buffer=0,
    )


def _serialize_episodes(
    episodes: Iterable["SingleAgentEpisode"],
) -> Iterator[Mapping[str, bytes]]:
    """Yield RLlib's Parquet rows without retaining the complete dataset."""
    import msgpack
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
    for shard_count, batch in enumerate(
        _batched(_serialize_episodes(episodes), episodes_per_shard)
    ):
        pq.write_table(
            pa.Table.from_pylist(batch),
            output_dir / f"part-{shard_count:05d}.parquet",
        )
    if shard_count == 0 and not any(output_dir.iterdir()):
        raise ValueError(
            "No episodes to convert. The Minari dataset is empty, so there is "
            + "nothing to learn from."
        )
    return shard_count + 1


def _discard_staging_directory(staging_dir: Path) -> None:
    """Remove a directory created exclusively for an unfinished conversion."""
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
            f"Refusing to replace existing conversion directory {output_dir}."
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


def _space_fingerprint(space: gym.Space[Any]) -> str:
    """Return a stable identifier for a Gymnasium space's public representation."""
    return hashlib.sha256(repr(space).encode("utf-8")).hexdigest()


def _conversion_manifest(
    dataset_id: str,
    dataset: Any,
    observation_space: gym.Space[Any],
    action_space: gym.Space[Any],
    episodes_per_shard: int,
) -> dict[str, Any]:
    """Describe the source data and conversion settings used by a cache entry."""
    return {
        "format_version": CONVERSION_FORMAT_VERSION,
        "dataset_id": dataset_id,
        "total_episodes": dataset.total_episodes,
        "total_steps": dataset.total_steps,
        "observation_space_fingerprint": _space_fingerprint(observation_space),
        "action_space_fingerprint": _space_fingerprint(action_space),
        "episodes_per_shard": episodes_per_shard,
    }


def _cache_key(manifest: Mapping[str, Any]) -> str:
    """Return a deterministic, filesystem-safe conversion directory name."""
    source_name = re.sub(
        r"[^A-Za-z0-9_.-]+", "-", str(manifest["dataset_id"])
    ).strip("-")
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return f"{source_name}-{digest}"


def _read_matching_manifest(
    output_dir: Path, expected: Mapping[str, Any]
) -> bool:
    """Return whether *output_dir* is a complete conversion for *expected*."""
    manifest_path = output_dir / MANIFEST_FILE_NAME
    if not manifest_path.is_file() or not any(output_dir.glob("*.parquet")):
        return False
    try:
        with manifest_path.open(encoding="utf-8") as manifest_file:
            return json.load(manifest_file) == expected
    except (OSError, json.JSONDecodeError):
        return False


def _publish_conversion(
    episode_factory: Callable[[], Iterable["SingleAgentEpisode"]],
    cache_root: Path,
    cache_key: str,
    manifest: Mapping[str, Any],
    *,
    episodes_per_shard: int,
) -> Path:
    """Write a conversion privately, then publish it atomically."""
    cache_root.mkdir(parents=True, exist_ok=True)
    output_dir = cache_root / cache_key
    if _read_matching_manifest(output_dir, manifest):
        logger.info("Reusing converted offline dataset at %s", output_dir)
        return output_dir
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to replace existing conversion directory {output_dir}."
        )

    staging_dir = cache_root / f".{cache_key}.{uuid.uuid4().hex}.staging"
    try:
        _write_episode_shards(
            episode_factory(), staging_dir, episodes_per_shard=episodes_per_shard
        )
        with (staging_dir / MANIFEST_FILE_NAME).open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, sort_keys=True)
            file.write("\n")
        try:
            os.replace(staging_dir, output_dir)
        except FileExistsError:
            # Another process won the race. Only reuse a conversion that proves it
            # was created from this exact immutable Minari dataset.
            if _read_matching_manifest(output_dir, manifest):
                return output_dir
            raise FileExistsError(
                f"Refusing to replace existing conversion directory {output_dir}."
            )
    except Exception:
        _discard_staging_directory(staging_dir)
        raise
    return output_dir


def convert_minari_dataset(
    dataset_id: str,
    cache_root: Path,
    *,
    episodes_per_shard: int = DEFAULT_EPISODES_PER_SHARD,
) -> tuple[Path, gym.Space[Any], gym.Space[Any], gym.Space[Any]]:
    """
    Convert a local Minari dataset into RLlib offline Parquet data.

    Parameters
    ----------
    dataset_id : str
        Identifier of a locally available Minari dataset, e.g. ``my-demo-v0``.
    cache_root : pathlib.Path
        Parent directory for owned, fingerprinted conversion-cache entries. Existing
        contents are never removed or replaced.
    episodes_per_shard : int, optional
        Maximum number of episode states in each Parquet file.

    Returns
    -------
    tuple
        ``(parquet_dir, training_observation_space, observation_space, action_space)``.
        ``training_observation_space`` configures the algorithm; ``observation_space``
        is the original (possibly composite) space, kept for ONNX export.

    Raises
    ------
    ValueError
        If the dataset contains no episodes.
    """
    import minari

    dataset = minari.load_dataset(dataset_id)
    observation_space = cast(gym.Space[Any], dataset.observation_space)
    action_space = cast(gym.Space[Any], dataset.action_space)
    manifest = _conversion_manifest(
        dataset_id,
        dataset,
        observation_space,
        action_space,
        episodes_per_shard,
    )
    parquet_dir = _publish_conversion(
        lambda: (
            minari_episode_to_rllib(episode, observation_space, action_space)
            for episode in dataset.iterate_episodes()
        ),
        Path(cache_root),
        _cache_key(manifest),
        manifest,
        episodes_per_shard=episodes_per_shard,
    )

    logger.info(
        "Converted Minari dataset '%s' (%s episodes, %s steps) to %s",
        dataset_id,
        dataset.total_episodes,
        dataset.total_steps,
        parquet_dir,
    )
    return (
        parquet_dir,
        get_training_observation_space(observation_space),
        observation_space,
        action_space,
    )
