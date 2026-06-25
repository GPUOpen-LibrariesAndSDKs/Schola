# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Serialize Gymnasium spaces, transitions, and numpy data to Schola protobuf messages.
"""

from typing import Any
import numpy as np
from numpy.typing import NDArray

from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary
import schola.generated.Spaces_pb2 as proto_spaces
import schola.generated.Points_pb2 as proto_points
import schola.generated.DType_pb2 as proto_dtype
import gymnasium.spaces as spaces
from functools import singledispatch

NUMPY_DTYPE_TO_PROTO_DTYPE_MAPPING = {
    np.float16: proto_dtype.DType.FLOAT16,
    np.float32: proto_dtype.DType.FLOAT32,
    np.float64: proto_dtype.DType.FLOAT64,
    np.uint8: proto_dtype.DType.UINT8,
    np.uint16: proto_dtype.DType.UINT16,
    np.uint32: proto_dtype.DType.UINT32,
    np.uint64: proto_dtype.DType.UINT64,
    np.int8: proto_dtype.DType.INT8,
    np.int16: proto_dtype.DType.INT16,
    np.int32: proto_dtype.DType.INT32,
    np.int64: proto_dtype.DType.INT64,
    np.bool: proto_dtype.DType.BOOL,
    np.bool_: proto_dtype.DType.BOOL,
}


def dtype_to_proto(dtype: np.dtype[Any]) -> proto_dtype.DType:
    """
    Convert a NumPy dtype to a protobuf DType message.

    Parameters
    ----------
    dtype : np.dtype
        The NumPy data type to convert.

    Returns
    -------
    proto_dtype.DType
        The corresponding protobuf DType message.

    Raises
    ------
    KeyError
        If the NumPy dtype is not recognized or supported.
    """
    if not dtype.type in NUMPY_DTYPE_TO_PROTO_DTYPE_MAPPING:
        raise KeyError(
            f"Numpy dtype {dtype.type} not recognized. Supported Numpy dtypes are {list(NUMPY_DTYPE_TO_PROTO_DTYPE_MAPPING.keys())}"
        )
    return NUMPY_DTYPE_TO_PROTO_DTYPE_MAPPING[dtype.type]


@singledispatch
def to_proto(msg: Any, /, *_args: Any) -> Any:
    """
    Serialize Python objects to protobuf messages.

    This is a generic function that uses singledispatch to handle different
    Python object types. It converts Python/Gymnasium objects like actions
    and spaces into protobuf messages that can be sent to Unreal Engine.

    Parameters
    ----------
    msg : Any
        The Python object to serialize. This can be a
        Note: For action serialization, you must also pass the corresponding
        space as the first argument.

    Returns
    -------
    Any
        The serialized protobuf message.

    Raises
    ------
    ValueError
        If the message type is not supported or if required parameters are missing.

    Notes
    -----
    This function has multiple registered implementations for different
    Python object types. See the individual @to_proto.register
    implementations for specific type conversions.
    To serialize gymnasium spaces use `space_to_proto`.
    """
    raise ValueError(
        f"Unsupported message type: {type(msg)}. To convert a point to protobuf message you need to pass the corresponding space."
    )


@to_proto.register(Box)
def _to_proto_box(space: Box, action: NDArray[Any]) -> proto_points.BoxPoint:
    msg = proto_points.BoxPoint()
    msg.values.extend(action.flatten())
    msg.dtype = dtype_to_proto(space.dtype)
    msg.shape.extend(space.shape)
    return msg


@to_proto.register(MultiDiscrete)
def _to_proto_multi_discrete(
    _space: MultiDiscrete, action: NDArray[Any] | list[int]
) -> proto_points.MultiDiscretePoint:
    msg = proto_points.MultiDiscretePoint()
    msg.values.extend(action)
    return msg


@to_proto.register(MultiBinary)
def _to_proto_multi_binary(
    _space: MultiBinary, action: NDArray[Any] | list[bool]
) -> proto_points.MultiBinaryPoint:
    msg = proto_points.MultiBinaryPoint()
    msg.values.extend(bool(value) for value in np.asarray(action, dtype=np.bool_).flat)
    return msg


@to_proto.register(spaces.Dict)
def _to_proto_dict(
    space: spaces.Dict, action: dict[str, Any]
) -> proto_points.DictPoint:
    msg = proto_points.DictPoint()
    for key in action:
        fill_generic(to_proto(space[key], action[key]), msg.values[key])
    return msg


@to_proto.register(spaces.Discrete)
def _to_proto_discrete(
    _space: spaces.Discrete[Any], action: int
) -> proto_points.DiscretePoint:
    msg = proto_points.DiscretePoint(value=action)
    return msg


@to_proto.register(spaces.Text)
def _to_proto_text(_space: spaces.Text, action: str) -> proto_points.TextPoint:
    msg = proto_points.TextPoint(value=action)
    return msg


@singledispatch
def space_to_proto(space: Any) -> Any:
    """
    Convert a Gymnasium space to a protobuf Space message.

    Parameters
    ----------
    space : gym.Space
        The Gymnasium space to convert (e.g., Box, Discrete, Dict, etc.).

    Returns
    -------
    proto_spaces.Space
        The corresponding protobuf Space message.

    Raises
    ------
    ValueError
        If the space type is not supported.

    Notes
    -----
    This function has multiple registered implementations for different
    Gymnasium space types (Box, Discrete, MultiDiscrete, MultiBinary, Dict).
    """
    raise ValueError(
        f"Unsupported message type: {type(space)}. Could not convert to a Protobuf message representing a space"
    )


@space_to_proto.register(MultiBinary)
def _space_to_proto_multi_binary(space: MultiBinary) -> proto_spaces.MultiBinarySpace:
    shape = space.n if isinstance(space.n, int) else int(np.prod(space.n))
    msg = proto_spaces.MultiBinarySpace(shape=shape)
    return msg


@space_to_proto.register(Box)
def _space_to_proto_box(space: Box) -> proto_spaces.BoxSpace:
    msg = proto_spaces.BoxSpace()
    msg.shape_dimensions.extend(space.shape)

    def box_space_dim_factory(
        dim_bounds: tuple[Any, Any],
    ) -> proto_spaces.BoxSpace.BoxSpaceDimension:
        return proto_spaces.BoxSpace.BoxSpaceDimension(
            low=dim_bounds[0], high=dim_bounds[1]
        )

    msg.dimensions.extend(
        map(box_space_dim_factory, zip(space.low.flat, space.high.flat))
    )
    msg.dtype = dtype_to_proto(space.dtype)
    return msg


@space_to_proto.register(MultiDiscrete)
def _space_to_proto_multi_discrete(
    space: MultiDiscrete,
) -> proto_spaces.MultiDiscreteSpace:
    msg = proto_spaces.MultiDiscreteSpace()
    msg.high.extend(space.nvec.astype(int))
    return msg


@space_to_proto.register(Discrete)
def _space_to_proto_discrete(space: Discrete[Any]) -> proto_spaces.DiscreteSpace:
    msg = proto_spaces.DiscreteSpace(high=int(space.n))
    return msg


@space_to_proto.register(spaces.Text)
def _space_to_proto_text(space: spaces.Text) -> proto_spaces.TextSpace:
    # sorted string of the space's character_set, which may be empty (the empty set).
    return proto_spaces.TextSpace(
        max_length=space.max_length,
        min_length=space.min_length,
        charset="".join(sorted(space.character_set)),
    )


@space_to_proto.register(spaces.Dict)
def _space_to_proto_dict(space: spaces.Dict) -> proto_spaces.DictSpace:
    msg = proto_spaces.DictSpace()
    for key, subspace in space.spaces.items():
        # cant use msg.spaces[key] = ... as it's not supported by protobuf
        fill_generic(space_to_proto(subspace), msg.spaces[key])
    return msg


# fill a generic Point/Space message with the specific type


@singledispatch
def fill_generic(_obj: object, _generic_obj: object) -> None:
    """
    Fill a generic protobuf Point or Space message with a specific typed message.

    This function is used to populate generic protobuf messages (which use
    oneof fields) with specific typed messages (like BoxSpace, DiscretePoint, etc.).

    Parameters
    ----------
    obj : Any
        The specific protobuf message (e.g., BoxSpace, DiscretePoint).
    generic_obj : Any
        The generic container protobuf message to fill (e.g., Space, Point).

    Raises
    ------
    ValueError
        If the message type is not supported.

    Notes
    -----
    This function modifies generic_obj in place using protobuf's CopyFrom method.

    See Also
    --------
    `make_generic` to create a new generic container protobuf message.
    """
    raise ValueError(
        f"Unsupported message type: {type(_obj)}. Could not fill a generic Point/Space protobuf message"
    )


@fill_generic.register
def _(space: proto_spaces.BoxSpace, generic_space: proto_spaces.Space) -> None:
    generic_space.box_space.CopyFrom(space)


@fill_generic.register
def _(space: proto_spaces.DiscreteSpace, generic_space: proto_spaces.Space) -> None:
    generic_space.discrete_space.CopyFrom(space)


@fill_generic.register
def _(
    space: proto_spaces.MultiDiscreteSpace, generic_space: proto_spaces.Space
) -> None:
    generic_space.multi_discrete_space.CopyFrom(space)


@fill_generic.register
def _(space: proto_spaces.MultiBinarySpace, generic_space: proto_spaces.Space) -> None:
    generic_space.multi_binary_space.CopyFrom(space)


@fill_generic.register
def _(space: proto_spaces.TextSpace, generic_space: proto_spaces.Space) -> None:
    generic_space.text_space.CopyFrom(space)


@fill_generic.register
def _(space: proto_spaces.DictSpace, generic_space: proto_spaces.Space) -> None:
    generic_space.dict_space.CopyFrom(space)


# points


@fill_generic.register
def _(point: proto_points.TextPoint, generic_point: proto_points.Point) -> None:
    generic_point.text_point.CopyFrom(point)


@fill_generic.register
def _(point: proto_points.DictPoint, generic_point: proto_points.Point) -> None:
    generic_point.dict_point.CopyFrom(point)


@fill_generic.register
def _(point: proto_points.BoxPoint, generic_point: proto_points.Point) -> None:
    generic_point.box_point.CopyFrom(point)


@fill_generic.register
def _(point: proto_points.MultiBinaryPoint, generic_point: proto_points.Point) -> None:
    generic_point.multi_binary_point.CopyFrom(point)


@fill_generic.register
def _(point: proto_points.DiscretePoint, generic_point: proto_points.Point) -> None:
    generic_point.discrete_point.CopyFrom(point)


@fill_generic.register
def _(
    point: proto_points.MultiDiscretePoint, generic_point: proto_points.Point
) -> None:
    generic_point.multi_discrete_point.CopyFrom(point)


# convert ___Point/___Space to geneirc Point/Space type respectively


@singledispatch
def make_generic(_obj: object) -> Any:
    """
    Convert a specific protobuf message to a generic Point or Space message.

    This function wraps specific typed protobuf messages (like BoxSpace,
    DiscretePoint) into generic protobuf messages that use oneof fields.

    Parameters
    ----------
    obj : Any
        The specific protobuf message (e.g., BoxSpace, DiscretePoint).

    Returns
    -------
    Any
        The generic protobuf message (e.g., Space, Point) containing the
        specific typed message.

    Raises
    ------
    ValueError
        If the message type is not supported.

    See Also
    --------
    `fill_generic` for filling a generic pre-existing container protobuf message with a specific typed message.

    """
    raise ValueError(
        f"Unsupported message type: {type(_obj)}. Could not convert specific protobuf type to a generic Point/Space protobuf message"
    )


@make_generic.register
def _(space: proto_spaces.DictSpace) -> proto_spaces.Space:
    msg = proto_spaces.Space(dict_space=space)
    return msg


@make_generic.register
def _(space: proto_spaces.TextSpace) -> proto_spaces.Space:
    msg = proto_spaces.Space(text_space=space)
    return msg


@make_generic.register
def _(space: proto_spaces.BoxSpace) -> proto_spaces.Space:
    msg = proto_spaces.Space(box_space=space)
    return msg


@make_generic.register
def _(space: proto_spaces.DiscreteSpace) -> proto_spaces.Space:
    msg = proto_spaces.Space(discrete_space=space)
    return msg


@make_generic.register
def _(space: proto_spaces.MultiDiscreteSpace) -> proto_spaces.Space:
    msg = proto_spaces.Space(multi_discrete_space=space)
    return msg


@make_generic.register
def _(space: proto_spaces.MultiBinarySpace) -> proto_spaces.Space:
    msg = proto_spaces.Space(multi_binary_space=space)
    return msg


@make_generic.register
def _(point: proto_points.DictPoint) -> proto_points.Point:
    msg = proto_points.Point(dict_point=point)
    return msg


@make_generic.register
def _(point: proto_points.BoxPoint) -> proto_points.Point:
    msg = proto_points.Point(box_point=point)
    return msg


@make_generic.register
def _(point: proto_points.MultiBinaryPoint) -> proto_points.Point:
    msg = proto_points.Point(multi_binary_point=point)
    return msg


@make_generic.register
def _(point: proto_points.DiscretePoint) -> proto_points.Point:
    msg = proto_points.Point(discrete_point=point)
    return msg


@make_generic.register
def _(point: proto_points.MultiDiscretePoint) -> proto_points.Point:
    msg = proto_points.Point(multi_discrete_point=point)
    return msg


@make_generic.register
def _(point: proto_points.TextPoint) -> proto_points.Point:
    msg = proto_points.Point(text_point=point)
    return msg
