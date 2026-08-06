# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

from typing import Any, overload

import numpy as np
from numpy.typing import NDArray
import gymnasium as gym
import gymnasium.spaces as spaces

import schola.generated.Spaces_pb2 as proto_spaces
import schola.generated.Points_pb2 as proto_points
import schola.generated.State_pb2 as state
import schola.generated.Definitions_pb2 as definitions
import schola.generated.ImitationState_pb2 as imitation_state_messages
import schola.generated.DType_pb2 as proto_dtype

PROTO_DTYPE_TO_NUMPY_DTYPE_MAPPING: dict[int, np.dtype[Any]]

def dtype_from_proto(msg: proto_dtype.DType) -> np.dtype[Any]: ...

# ---------------------------------------------------------------------------
# from_proto – space deserialization
# ---------------------------------------------------------------------------

@overload
def from_proto(msg: proto_spaces.BoxSpace) -> spaces.Box: ...
@overload
def from_proto(msg: proto_spaces.MultiBinarySpace) -> spaces.MultiBinary: ...
@overload
def from_proto(msg: proto_spaces.DiscreteSpace) -> spaces.Discrete[np.int64]: ...
@overload
def from_proto(msg: proto_spaces.MultiDiscreteSpace) -> spaces.MultiDiscrete: ...
@overload
def from_proto(msg: proto_spaces.TextSpace) -> spaces.Text: ...
@overload
def from_proto(msg: proto_spaces.DictSpace) -> spaces.Dict: ...
@overload
def from_proto(msg: proto_spaces.Space) -> gym.Space[Any]: ...

# ---------------------------------------------------------------------------
# from_proto – point deserialization
# ---------------------------------------------------------------------------

@overload
def from_proto(msg: proto_points.BoxPoint) -> NDArray[Any]: ...
@overload
def from_proto(msg: proto_points.MultiDiscretePoint) -> NDArray[Any]: ...
@overload
def from_proto(msg: proto_points.DiscretePoint) -> int: ...
@overload
def from_proto(msg: proto_points.MultiBinaryPoint) -> NDArray[Any]: ...
@overload
def from_proto(msg: proto_points.TextPoint) -> str: ...
@overload
def from_proto(msg: proto_points.DictPoint) -> dict[str, Any]: ...
@overload
def from_proto(msg: proto_points.Point) -> dict[str, Any] | NDArray[Any]: ...

# ---------------------------------------------------------------------------
# from_proto – initial state deserialization
# ---------------------------------------------------------------------------

@overload
def from_proto(
    msg: state.InitialAgentState,
) -> tuple[NDArray[Any] | dict[str, Any], dict[str, str]]: ...
@overload
def from_proto(
    msg: state.InitialEnvironmentState,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]: ...
@overload
def from_proto(
    msg: state.InitialState,
) -> tuple[
    dict[int, dict[str, dict[str, Any]]], dict[int, dict[str, dict[str, str]]]
]: ...

# ---------------------------------------------------------------------------
# from_proto – training state deserialization
# ---------------------------------------------------------------------------

@overload
def from_proto(
    msg: state.AgentState,
) -> tuple[Any, float, bool, bool, dict[str, str]]: ...
@overload
def from_proto(
    msg: state.EnvironmentState,
) -> tuple[
    dict[str, Any],
    dict[str, float],
    dict[str, bool],
    dict[str, bool],
    dict[str, dict[str, str]],
]: ...
@overload
def from_proto(
    msg: state.TrainingState,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, float]],
    list[dict[str, bool]],
    list[dict[str, bool]],
    list[dict[str, dict[str, str]]],
]: ...

# ---------------------------------------------------------------------------
# from_proto – definition deserialization
# ---------------------------------------------------------------------------

@overload
def from_proto(
    msg: definitions.AgentDefinition,
) -> tuple[str, gym.Space[Any], gym.Space[Any]]: ...
@overload
def from_proto(
    msg: definitions.EnvironmentDefinition,
) -> tuple[
    list[str],
    dict[str, str],
    dict[str, gym.Space[Any]],
    dict[str, gym.Space[Any]],
]: ...
@overload
def from_proto(
    msg: definitions.TrainingDefinition,
) -> tuple[
    list[list[str]],
    list[dict[str, str]],
    dict[int, dict[str, gym.Space[Any]]],
    dict[int, dict[str, gym.Space[Any]]],
]: ...

# ---------------------------------------------------------------------------
# from_proto – imitation state deserialization
# ---------------------------------------------------------------------------

@overload
def from_proto(
    msg: imitation_state_messages.ImitationAgentState,
) -> tuple[NDArray[Any], float, bool, bool, dict[str, str], Any]: ...
@overload
def from_proto(
    msg: imitation_state_messages.ImitationEnvironmentState,
) -> tuple[
    dict[str, Any],
    dict[str, float],
    dict[str, bool],
    dict[str, bool],
    dict[str, dict[str, str]],
    dict[str, Any],
]: ...
@overload
def from_proto(
    msg: imitation_state_messages.ImitationTrainingState,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, float]],
    list[dict[str, bool]],
    list[dict[str, bool]],
    list[dict[str, dict[str, str]]],
    list[dict[str, Any]],
]: ...
@overload
def from_proto(
    msg: imitation_state_messages.ImitationState,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, float]],
    list[dict[str, bool]],
    list[dict[str, bool]],
    list[dict[str, dict[str, str]]],
    dict[int, dict[str, Any]],
    dict[int, dict[str, dict[str, str]]],
    list[dict[str, Any]],
]: ...

# ---------------------------------------------------------------------------
# from_proto – fallback
# ---------------------------------------------------------------------------

@overload
def from_proto(msg: object) -> Any: ...
