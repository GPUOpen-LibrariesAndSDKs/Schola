# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

from typing import Any, overload

import numpy as np
from numpy.typing import NDArray

from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary
import gymnasium.spaces as spaces
import schola.generated.Spaces_pb2 as proto_spaces
import schola.generated.Points_pb2 as proto_points
import schola.generated.DType_pb2 as proto_dtype

NUMPY_DTYPE_TO_PROTO_DTYPE_MAPPING: dict[type[np.generic], int]

def dtype_to_proto(dtype: np.dtype[Any]) -> proto_dtype.DType: ...

# ---------------------------------------------------------------------------
# to_proto – action serialization
# Dispatches on the first argument (the space); the action is the second arg.
# ---------------------------------------------------------------------------

@overload
def to_proto(msg: Box, action: NDArray[Any], /) -> proto_points.BoxPoint: ...
@overload
def to_proto(
    msg: MultiDiscrete, action: NDArray[Any] | list[int], /
) -> proto_points.MultiDiscretePoint: ...
@overload
def to_proto(
    msg: MultiBinary, action: NDArray[Any] | list[bool], /
) -> proto_points.MultiBinaryPoint: ...
@overload
def to_proto(msg: spaces.Dict, action: dict[str, Any], /) -> proto_points.DictPoint: ...
@overload
def to_proto(
    msg: spaces.Discrete[np.int64], action: int, /
) -> proto_points.DiscretePoint: ...
@overload
def to_proto(msg: spaces.Text, action: str, /) -> proto_points.TextPoint: ...
@overload
def to_proto(msg: Any, /, *_args: Any) -> Any: ...

# ---------------------------------------------------------------------------
# space_to_proto – Gymnasium space → protobuf Space message
# ---------------------------------------------------------------------------

@overload
def space_to_proto(space: Box) -> proto_spaces.BoxSpace: ...
@overload
def space_to_proto(space: MultiBinary) -> proto_spaces.MultiBinarySpace: ...
@overload
def space_to_proto(space: MultiDiscrete) -> proto_spaces.MultiDiscreteSpace: ...
@overload
def space_to_proto(space: Discrete[np.int64]) -> proto_spaces.DiscreteSpace: ...
@overload
def space_to_proto(space: spaces.Text) -> proto_spaces.TextSpace: ...
@overload
def space_to_proto(space: spaces.Dict) -> proto_spaces.DictSpace: ...
@overload
def space_to_proto(space: Any) -> Any: ...

# ---------------------------------------------------------------------------
# fill_generic – fill a generic Space/Point oneof in-place (always None)
# ---------------------------------------------------------------------------

@overload
def fill_generic(
    obj: proto_spaces.BoxSpace, generic_obj: proto_spaces.Space
) -> None: ...
@overload
def fill_generic(
    obj: proto_spaces.DiscreteSpace, generic_obj: proto_spaces.Space
) -> None: ...
@overload
def fill_generic(
    obj: proto_spaces.MultiDiscreteSpace, generic_obj: proto_spaces.Space
) -> None: ...
@overload
def fill_generic(
    obj: proto_spaces.MultiBinarySpace, generic_obj: proto_spaces.Space
) -> None: ...
@overload
def fill_generic(
    obj: proto_spaces.TextSpace, generic_obj: proto_spaces.Space
) -> None: ...
@overload
def fill_generic(
    obj: proto_spaces.DictSpace, generic_obj: proto_spaces.Space
) -> None: ...
@overload
def fill_generic(
    obj: proto_points.TextPoint, generic_obj: proto_points.Point
) -> None: ...
@overload
def fill_generic(
    obj: proto_points.DictPoint, generic_obj: proto_points.Point
) -> None: ...
@overload
def fill_generic(
    obj: proto_points.BoxPoint, generic_obj: proto_points.Point
) -> None: ...
@overload
def fill_generic(
    obj: proto_points.MultiBinaryPoint, generic_obj: proto_points.Point
) -> None: ...
@overload
def fill_generic(
    obj: proto_points.DiscretePoint, generic_obj: proto_points.Point
) -> None: ...
@overload
def fill_generic(
    obj: proto_points.MultiDiscretePoint, generic_obj: proto_points.Point
) -> None: ...
@overload
def fill_generic(obj: object, generic_obj: object) -> None: ...

# ---------------------------------------------------------------------------
# make_generic – wrap a specific Space/Point message in the generic oneof
# ---------------------------------------------------------------------------

@overload
def make_generic(obj: proto_spaces.BoxSpace) -> proto_spaces.Space: ...
@overload
def make_generic(obj: proto_spaces.DiscreteSpace) -> proto_spaces.Space: ...
@overload
def make_generic(obj: proto_spaces.MultiDiscreteSpace) -> proto_spaces.Space: ...
@overload
def make_generic(obj: proto_spaces.MultiBinarySpace) -> proto_spaces.Space: ...
@overload
def make_generic(obj: proto_spaces.TextSpace) -> proto_spaces.Space: ...
@overload
def make_generic(obj: proto_spaces.DictSpace) -> proto_spaces.Space: ...
@overload
def make_generic(obj: proto_points.BoxPoint) -> proto_points.Point: ...
@overload
def make_generic(obj: proto_points.MultiBinaryPoint) -> proto_points.Point: ...
@overload
def make_generic(obj: proto_points.DiscretePoint) -> proto_points.Point: ...
@overload
def make_generic(obj: proto_points.MultiDiscretePoint) -> proto_points.Point: ...
@overload
def make_generic(obj: proto_points.TextPoint) -> proto_points.Point: ...
@overload
def make_generic(obj: proto_points.DictPoint) -> proto_points.Point: ...
@overload
def make_generic(obj: object) -> Any: ...
