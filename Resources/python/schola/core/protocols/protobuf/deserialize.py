# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Map Schola protobuf messages to Gymnasium spaces, points, and numpy buffers.
"""

from functools import singledispatch
from typing import Any, cast

import gymnasium.spaces as spaces
import schola.generated.Spaces_pb2 as proto_spaces
import schola.generated.Points_pb2 as proto_points
import schola.generated.State_pb2 as state
import schola.generated.Definitions_pb2 as definitions
import schola.generated.ImitationState_pb2 as imitation_state_messages
import numpy as np
from numpy.typing import NDArray
import gymnasium as gym
import schola.generated.DType_pb2 as proto_dtype

PROTO_DTYPE_TO_NUMPY_DTYPE_MAPPING = {
    proto_dtype.DType.FLOAT16: np.float16,
    proto_dtype.DType.FLOAT32: np.float32,
    proto_dtype.DType.FLOAT64: np.float64,
    proto_dtype.DType.UINT8: np.uint8,
    proto_dtype.DType.UINT16: np.uint16,
    proto_dtype.DType.UINT32: np.uint32,
    proto_dtype.DType.UINT64: np.uint64,
    proto_dtype.DType.INT8: np.int8,
    proto_dtype.DType.INT16: np.int16,
    proto_dtype.DType.INT32: np.int32,
    proto_dtype.DType.INT64: np.int64,
    proto_dtype.DType.BOOL: np.bool_,
}


def dtype_from_proto(msg: proto_dtype.DType) -> np.dtype[Any]:
    """
    Convert a protobuf DType message to a NumPy dtype.

    Parameters
    ----------
    msg : proto_dtype.DType
        The protobuf DType message to convert.

    Returns
    -------
    np.dtype
        The corresponding NumPy data type.

    Raises
    ------
    KeyError
        If the protobuf DType is not recognized.
    """
    if not msg in PROTO_DTYPE_TO_NUMPY_DTYPE_MAPPING:
        raise KeyError(
            f"DType {msg} not recognized. Valid DTypes are {list(PROTO_DTYPE_TO_NUMPY_DTYPE_MAPPING.keys())}"
        )
    return np.dtype(PROTO_DTYPE_TO_NUMPY_DTYPE_MAPPING[msg])


@singledispatch
def from_proto(_msg: object) -> Any:
    """
    Deserialize a protobuf message to Python objects.

    This is a generic function that uses singledispatch to handle different
    protobuf message types. It converts protobuf messages from Unreal Engine
    into Python/Gymnasium-compatible objects like spaces, observations, and states.

    Parameters
    ----------
    msg : Any
        The protobuf message to deserialize.

    Returns
    -------
    Any
        The deserialized Python object (e.g., gym.Space, np.ndarray, Dict, etc.).

    Notes
    -----
    This function has multiple registered implementations for different
    protobuf message types. See the individual @from_proto.register
    implementations for specific type conversions.
    """
    ...


# Space deserialization


@from_proto.register
def _(msg: proto_spaces.BoxSpace) -> spaces.Box:
    low = []
    high = []
    for dimension in msg.dimensions:
        low.append(dimension.low)
        high.append(dimension.high)
    if len(msg.shape_dimensions) == 0:
        shape = [len(low)]
    else:
        shape = tuple(msg.shape_dimensions)

    dt = dtype_from_proto(msg.dtype)
    low_arr = np.asarray(low, dtype=dt).reshape(shape)
    high_arr = np.asarray(high, dtype=dt).reshape(shape)

    return spaces.Box(low=low_arr, high=high_arr, shape=shape, dtype=cast(Any, dt))


@from_proto.register
def _(msg: proto_spaces.MultiBinarySpace) -> spaces.MultiBinary:
    return spaces.MultiBinary(n=msg.shape)


@from_proto.register
def _(msg: proto_spaces.DiscreteSpace) -> spaces.Discrete[Any]:
    return spaces.Discrete(msg.high)


@from_proto.register
def _(msg: proto_spaces.MultiDiscreteSpace) -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete(list(msg.high))


@from_proto.register
def _(msg: proto_spaces.TextSpace) -> spaces.Text:
    return spaces.Text(
        max_length=msg.max_length,
        min_length=msg.min_length,
        charset=msg.charset,
    )


@from_proto.register
def _(msg: proto_spaces.DictSpace) -> spaces.Dict:
    space_dict = {key: from_proto(value) for key, value in msg.spaces.items()}
    return spaces.Dict(spaces=space_dict)


@from_proto.register
def _(msg: proto_spaces.Space) -> gym.Space[Any]:
    which = msg.WhichOneof("space")
    if which is None:
        raise ValueError(
            "Received Space proto with no 'space' oneof field set. Upstream serialization likely passed an uninitialized FSpace / TInstancedStruct."
        )
    return from_proto(getattr(msg, which))


# Point Deserialization


@from_proto.register
def _(msg: proto_points.BoxPoint) -> NDArray[Any]:
    shape = msg.shape if len(msg.shape) > 0 else None
    return np.array(msg.values, dtype=dtype_from_proto(msg.dtype)).reshape(shape)


@from_proto.register
def _(msg: proto_points.MultiDiscretePoint) -> NDArray[Any]:
    return np.array(msg.values, dtype=np.int64)


@from_proto.register
def _(msg: proto_points.DiscretePoint) -> int:
    return msg.value


@from_proto.register
def _(msg: proto_points.MultiBinaryPoint) -> NDArray[Any]:
    # np.bool was removed in NumPy 1.24; use bool/np.bool_ for compatibility
    return np.array(msg.values, dtype=np.bool_)


@from_proto.register
def _(msg: proto_points.TextPoint) -> str:
    return msg.value


@from_proto.register
def _(msg: proto_points.DictPoint) -> dict[str, Any]:
    return {key: from_proto(value) for key, value in msg.values.items()}


@from_proto.register
def _(msg: proto_points.Point) -> dict[str, Any] | NDArray[Any]:
    which = msg.WhichOneof("point")
    if which is None:
        raise ValueError(
            "Received Point proto with no 'point' oneof field set. Upstream serialization likely passed an uninitialized FPoint / TInstancedStruct."
        )
    return from_proto(getattr(msg, which))


# Initial State Deserialization
@from_proto.register
def _(msg: state.InitialAgentState) -> tuple[NDArray[Any], dict[str, str]]:
    observations = from_proto(msg.observations)
    infos = dict(msg.info)
    return observations, infos


@from_proto.register
def _(msg: state.InitialEnvironmentState) -> tuple[dict[str, Any], dict[str, str]]:
    observations = {}
    infos = {}
    for agent_id, agent_state in msg.agent_states.items():
        observations[agent_id], infos[agent_id] = from_proto(agent_state)
    return observations, infos


@from_proto.register
def _(
    msg: state.InitialState,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, str]]]:
    observations = {}
    infos = {}
    for env_id, env_state in msg.environment_states.items():
        observations[env_id], infos[env_id] = from_proto(env_state)
    return observations, infos


# State Deserialization


@from_proto.register
def _(msg: state.AgentState) -> tuple[Any, float, bool, bool, dict[str, str]]:
    observations = from_proto(msg.observations)
    infos = dict(msg.info)
    terminated = msg.terminated
    truncated = msg.truncated
    return observations, msg.reward, terminated, truncated, infos


@from_proto.register
def _(
    msg: state.EnvironmentState,
) -> tuple[
    dict[str, Any],
    dict[str, float],
    dict[str, bool],
    dict[str, bool],
    dict[str, dict[str, str]],
]:
    observations = {}
    rewards = {}
    terminateds = {}
    truncateds = {}
    infos = {}
    for agent_id, agent_state in msg.agent_states.items():
        (
            observations[agent_id],
            rewards[agent_id],
            terminateds[agent_id],
            truncateds[agent_id],
            infos[agent_id],
        ) = from_proto(agent_state)
    return observations, rewards, terminateds, truncateds, infos


@from_proto.register
def _(
    msg: state.TrainingState,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, float]],
    list[dict[str, bool]],
    list[dict[str, bool]],
    list[dict[str, dict[str, str]]],
]:
    observations = [{} for _ in range(len(msg.environment_states))]
    rewards = [{} for _ in range(len(msg.environment_states))]
    terminateds = [{} for _ in range(len(msg.environment_states))]
    truncateds = [{} for _ in range(len(msg.environment_states))]
    infos = [{} for _ in range(len(msg.environment_states))]
    for env_id, env_state in enumerate(msg.environment_states):
        (
            observations[env_id],
            rewards[env_id],
            terminateds[env_id],
            truncateds[env_id],
            infos[env_id],
        ) = from_proto(env_state)
    return observations, rewards, terminateds, truncateds, infos


# Definition Deserialization


@from_proto.register
def _(msg: definitions.AgentDefinition) -> tuple[str, gym.Space[Any], gym.Space[Any]]:

    obs_space = from_proto(msg.obs_space)
    act_space = from_proto(msg.action_space)
    agent_type = msg.agent_type

    return agent_type, obs_space, act_space


@from_proto.register
def _(
    msg: definitions.EnvironmentDefinition,
) -> tuple[
    list[str], dict[str, str], dict[str, gym.Space[Any]], dict[str, gym.Space[Any]]
]:
    uids = [agent_id for agent_id in msg.agent_definitions]
    agent_types = {}
    obs_spaces = {}
    act_spaces = {}

    for uid in uids:
        agent_types[uid], obs_spaces[uid], act_spaces[uid] = from_proto(
            msg.agent_definitions[uid]
        )

    return uids, agent_types, obs_spaces, act_spaces


@from_proto.register
def _(
    msg: definitions.TrainingDefinition,
) -> tuple[
    list[list[str]],
    dict[int, dict[str, str]],
    dict[int, dict[str, gym.Space[Any]]],
    dict[int, dict[str, gym.Space[Any]]],
]:

    env_uids = [[] for _ in msg.environment_definitions]
    obs_defns: dict[int, dict[str, gym.Space[Any]]] = {}
    action_defns: dict[int, dict[str, gym.Space[Any]]] = {}
    agent_types: dict[int, dict[str, str]] = {}

    for env_id, env_defn in enumerate(msg.environment_definitions):
        (
            env_uids[env_id],
            agent_types[env_id],
            obs_defns[env_id],
            action_defns[env_id],
        ) = from_proto(env_defn)

    return env_uids, agent_types, obs_defns, action_defns


# Imitation Deserialization


@from_proto.register
def _(
    msg: imitation_state_messages.ImitationAgentState,
) -> tuple[NDArray[Any], float, bool, bool, dict[str, str], Any]:
    observations = from_proto(msg.observations)
    reward = msg.reward
    terminated = msg.terminated
    truncated = msg.truncated
    infos = dict(msg.info)
    actions = from_proto(msg.actions)
    return observations, reward, terminated, truncated, infos, actions


@from_proto.register
def _(
    msg: imitation_state_messages.ImitationEnvironmentState,
) -> tuple[
    dict[str, Any],
    dict[str, float],
    dict[str, bool],
    dict[str, bool],
    dict[str, dict[str, str]],
    dict[str, Any],
]:
    observations = {}
    rewards = {}
    terminateds = {}
    truncateds = {}
    infos = {}
    actions = {}
    for agent_id, agent_state in msg.agent_states.items():
        (
            observations[agent_id],
            rewards[agent_id],
            terminateds[agent_id],
            truncateds[agent_id],
            infos[agent_id],
            actions[agent_id],
        ) = from_proto(agent_state)
    return observations, rewards, terminateds, truncateds, infos, actions


@from_proto.register
def _(
    msg: imitation_state_messages.ImitationTrainingState,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, float]],
    list[dict[str, bool]],
    list[dict[str, bool]],
    list[dict[str, dict[str, str]]],
    list[dict[str, Any]],
]:
    observations = [{} for _ in range(len(msg.environment_states))]
    rewards = [{} for _ in range(len(msg.environment_states))]
    terminateds = [{} for _ in range(len(msg.environment_states))]
    truncateds = [{} for _ in range(len(msg.environment_states))]
    infos = [{} for _ in range(len(msg.environment_states))]
    actions = [{} for _ in range(len(msg.environment_states))]
    for env_id, env_state in enumerate(msg.environment_states):
        (
            observations[env_id],
            rewards[env_id],
            terminateds[env_id],
            truncateds[env_id],
            infos[env_id],
            actions[env_id],
        ) = from_proto(env_state)
    return observations, rewards, terminateds, truncateds, infos, actions


@from_proto.register
def _(
    msg: imitation_state_messages.ImitationState,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, float]],
    list[dict[str, bool]],
    list[dict[str, bool]],
    list[dict[str, dict[str, str]]],
    dict[int, dict[str, Any]],
    dict[int, dict[str, str]],
    list[dict[str, Any]],
]:
    # Deserialize training_state if present
    observations, rewards, terminateds, truncateds, infos, actions = from_proto(
        msg.training_state
    )

    # Deserialize initial_state if present
    initial_observations = {}
    initial_infos = {}
    if msg.HasField("initial_state"):
        initial_observations, initial_infos = from_proto(msg.initial_state)

    return (
        observations,
        rewards,
        terminateds,
        truncateds,
        infos,
        initial_observations,
        initial_infos,
        actions,
    )
