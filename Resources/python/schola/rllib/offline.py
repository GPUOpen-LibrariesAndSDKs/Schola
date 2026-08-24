# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""
RLlib new-API-stack offline data: Parquet episodes plus a space sidecar.

RLlib's offline algorithms (BC, MARWIL) read Ray Data sources containing
msgpack-serialized :class:`~ray.rllib.env.single_agent_episode.SingleAgentEpisode`
states. This module writes that layout with the same msgpack packing and
``ray.data.Dataset.write_parquet`` calls that RLlib's
:class:`~ray.rllib.offline.offline_env_runner.OfflineSingleAgentEnvRunner`
uses, and stores the original and training observation spaces so training does
not need a live environment.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, SupportsFloat, TypeVar

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
T = TypeVar("T")


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


def _decode_legacy_schola_space(payload: Mapping[str, Any]) -> gym.Space[Any]:
    """
    Decode a space from the hand-rolled manifest format.

    Read-only compatibility for demonstrations recorded before the manifest
    moved to Schola's protobuf space encoding. Hand-recorded demonstrations are
    expensive to replace, so old datasets keep loading.
    """
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
            {
                key: _decode_legacy_schola_space(subspace)
                for key, subspace in nested.items()
            }
        )
    if space_type == "Tuple":
        nested_list = payload["spaces"]
        if not isinstance(nested_list, Sequence) or isinstance(
            nested_list, (str, bytes)
        ):
            raise TypeError("Tuple space payload must be a sequence of subspaces.")
        return spaces.Tuple(
            tuple(_decode_legacy_schola_space(subspace) for subspace in nested_list)
        )
    raise TypeError(f"Unknown serialized Gymnasium space type {space_type!r}.")


def space_to_json(space: gym.Space[Any]) -> dict[str, Any]:
    """
    Serialize a Gymnasium space for the offline manifest sidecar.

    Reuses Schola's protobuf space encoding, so the manifest can express
    exactly the spaces Unreal can send and nothing else. Anything outside that
    set raises from :func:`~schola.core.protocols.protobuf.serialize.space_to_proto`
    rather than being silently approximated here.
    """
    from google.protobuf import json_format

    from schola.core.protocols.protobuf.serialize import make_generic, space_to_proto

    return {
        "format": "schola-proto",
        "space": json_format.MessageToDict(make_generic(space_to_proto(space))),
    }


def space_from_json(payload: Mapping[str, Any]) -> gym.Space[Any]:
    """Deserialize a space written by :func:`space_to_json`."""
    from google.protobuf import json_format

    import schola.generated.Spaces_pb2 as proto_spaces
    from schola.core.protocols.protobuf.deserialize import from_proto

    fmt = payload.get("format", "schola")
    space_payload = payload.get("space", payload)
    if not isinstance(space_payload, Mapping):
        raise TypeError("Serialized space payload must be a mapping.")
    if fmt == "schola-proto":
        return from_proto(json_format.ParseDict(space_payload, proto_spaces.Space()))
    return _decode_legacy_schola_space(space_payload)


def _ensure_ray_for_data_io() -> None:
    """Start a local Ray runtime when offline Parquet I/O runs outside Tune."""
    import ray

    if ray.is_initialized():
        return
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    ray.init(num_cpus=1, include_dashboard=False, ignore_reinit_error=True)


def _pack_episode_states(episodes: Sequence["SingleAgentEpisode"]) -> list[bytes]:
    """Pack episodes the same way RLlib's ``OfflineSingleAgentEnvRunner`` does."""
    import msgpack

    # Optional extra (schola[rllib-offline]); the package ships no type stubs.
    import msgpack_numpy  # pyright: ignore[reportMissingImports]

    packed: list[bytes] = []
    for episode in episodes:
        if episode.is_numpy:
            raise TypeError(
                "RLlib offline episode recording requires list-based episodes; "
                f"got numpy-backed episode {episode!r}."
            )
        blob = msgpack.packb(episode.get_state(), default=msgpack_numpy.encode)
        if blob is None:
            raise RuntimeError("Failed to serialize RLlib episode state.")
        packed.append(blob)
    return packed


def _batched_episodes(
    episodes: Iterable["SingleAgentEpisode"], size: int
) -> Iterator[list["SingleAgentEpisode"]]:
    """Yield bounded episode batches without materializing the full dataset."""
    batch: list["SingleAgentEpisode"] = []
    for episode in episodes:
        batch.append(episode)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


@contextlib.contextmanager
def _quiet_ray_data_logging() -> Iterator[None]:
    """
    Silence Ray Data's per-write pipeline commentary.

    Ray Data decorates its progress lines with emoji, and its log handlers open
    the session log file using the locale code page. On any non-UTF-8 locale
    (cp1252 is the Windows default) emitting those records raises
    ``UnicodeEncodeError``, so every status line prints a traceback and buries
    the real output. Raising the level on ``ray.data`` drops the records before
    a handler sees them; child loggers inherit it, and errors still surface.
    """
    ray_data_logger = logging.getLogger("ray.data")
    previous_level = ray_data_logger.level
    ray_data_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        ray_data_logger.setLevel(previous_level)


def _write_parquet_shard(packed_episodes: Sequence[bytes], shard_dir: Path) -> None:
    """Write one msgpack episode shard with Ray Data, matching RLlib's env runner."""
    import ray.data

    _ensure_ray_for_data_io()
    with _quiet_ray_data_logging():
        ray.data.from_items(list(packed_episodes)).write_parquet(
            str(shard_dir),
            try_create_dir=True,
        )


def _write_episode_shards(
    episodes: Iterable["SingleAgentEpisode"],
    output_dir: Path,
    *,
    episodes_per_shard: int,
) -> int:
    """Write episode states to independently readable Parquet shard directories."""
    if episodes_per_shard < 1:
        raise ValueError("episodes_per_shard must be at least one.")

    output_dir.mkdir(parents=True, exist_ok=False)
    episode_count = 0
    for shard_count, episode_batch in enumerate(
        _batched_episodes(episodes, episodes_per_shard)
    ):
        episode_count += len(episode_batch)
        _write_parquet_shard(
            _pack_episode_states(episode_batch),
            output_dir / f"part-{shard_count:05d}",
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


def _commit_new_directory(output_dir: Path, populate: Callable[[Path], T]) -> T:
    """Populate a staging directory, then rename it to *output_dir*."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to replace existing dataset directory {output_dir}."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.parent / f".{output_dir.name}.{uuid.uuid4().hex}.staging"
    try:
        result = populate(staging_dir)
        os.replace(staging_dir, output_dir)
        return result
    except Exception:
        _discard_staging_directory(staging_dir)
        raise


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
        Maximum number of episodes in each Parquet shard directory.

    Returns
    -------
    pathlib.Path
        *output_dir*, for convenience.
    """
    output_dir = Path(output_dir)

    def populate(staging_dir: Path) -> None:
        _write_episode_shards(
            episodes, staging_dir, episodes_per_shard=episodes_per_shard
        )

    _commit_new_directory(output_dir, populate)
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
        Maximum number of episodes in each Parquet shard directory.

    Returns
    -------
    pathlib.Path
        *output_dir*, for convenience.
    """
    output_dir = Path(output_dir)
    training_observation_space = get_training_observation_space(observation_space)

    def populate(staging_dir: Path) -> tuple[int, int]:
        episode_count = _write_episode_shards(
            episodes, staging_dir, episodes_per_shard=episodes_per_shard
        )
        total_steps = sum(len(episode) for episode in episodes)
        manifest = {
            "format_version": MANIFEST_FORMAT_VERSION,
            "observation_space": space_to_json(observation_space),
            "training_observation_space": space_to_json(training_observation_space),
            "action_space": space_to_json(action_space),
            "total_episodes": episode_count,
            "total_steps": total_steps,
        }
        with (staging_dir / MANIFEST_FILE_NAME).open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, sort_keys=True)
            file.write("\n")
        return episode_count, total_steps

    episode_count, total_steps = _commit_new_directory(output_dir, populate)
    logger.info(
        "Wrote %s RLlib episodes (%s steps) to %s",
        episode_count,
        total_steps,
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
        ``parquet_dir`` is absolute so Ray workers can read it from any working directory.

    Raises
    ------
    FileNotFoundError
        If the directory, manifest, or Parquet shards are missing.
    ValueError
        If the manifest is not a valid Schola offline dataset.
    """
    input_dir = Path(input_dir).resolve()
    manifest_path = input_dir / MANIFEST_FILE_NAME
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Offline dataset directory does not exist: {input_dir}"
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Offline dataset is missing {MANIFEST_FILE_NAME}: {input_dir}"
        )
    if not any(input_dir.rglob("*.parquet")):
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
