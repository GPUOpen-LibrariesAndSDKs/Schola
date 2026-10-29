# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.
# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnusedImport=false
"""Tests for the protobuf serialization"""

from schola.core.protocols.protobuf.serialize import (
    NUMPY_DTYPE_TO_PROTO_DTYPE_MAPPING,
    to_proto,
    space_to_proto,
)
from schola.core.protocols.protobuf.deserialize import from_proto
from schola.generated.Points_pb2 import *
from schola.generated.Spaces_pb2 import *
from schola.generated.DType_pb2 import *
import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary, Dict, Text
from gymnasium.spaces.text import alphanumeric


class TestDiscretePoint:
    def test_empty(self):
        """Test serialization of 0 to DiscretePoint"""
        space = Discrete(10)
        point = to_proto(space, 0)
        assert isinstance(
            point, DiscretePoint
        ), "Serialized point should be DiscretePoint"
        assert point.value == 0, "DiscretePoint value should be 0"

    def test_value(self):
        """Test serialization of 5 to DiscretePoint"""
        space = Discrete(10)
        point = to_proto(space, 5)
        assert isinstance(
            point, DiscretePoint
        ), "Serialized point should be DiscretePoint"
        assert point.value == 5, "DiscretePoint value should be 5"

    def test_max_value(self):
        """Test serialization of 9 to DiscretePoint"""
        space = Discrete(10)
        point = to_proto(space, 9)
        assert isinstance(
            point, DiscretePoint
        ), "Serialized point should be DiscretePoint"
        assert point.value == 9, "DiscretePoint value should be 9"

    @pytest.mark.parametrize(
        "value",
        [
            0,
            5,
            9,
        ],
        ids=["empty", "value", "max_value"],
    )
    def test_roundtrip(self, value: int):
        """An integer survives a proto roundtrip."""
        space = Discrete(10)
        restored = from_proto(to_proto(space, value))
        assert restored == value


class TestDiscreteSpace:

    def test_simple_shape(self):
        """Test serialization of Discrete(3) to DiscreteSpace"""
        space = Discrete(3)
        proto = space_to_proto(space)
        assert isinstance(
            proto, DiscreteSpace
        ), "Serialized space should be DiscreteSpace"
        assert proto.high == 3, "DiscreteSpace high should be 3"

    @pytest.mark.skip(reason="Discrete start is not supported yet")
    def test_start(self):
        """Test serialization of a non-zero start to DiscreteSpace"""
        space = Discrete(3, start=5)
        proto = space_to_proto(space)
        assert isinstance(
            proto, DiscreteSpace
        ), "Serialized space should be DiscreteSpace"
        assert proto.high == 3, "DiscreteSpace high should be 3"

    @pytest.mark.parametrize(
        "space",
        [
            Discrete(4),
            # Discrete(3, start=5), # not supported yet
        ],
        ids=[
            "simple_shape",
            # "start", # not supported yet
        ],
    )
    def test_roundtrip(self, space: Discrete):
        """A Discrete space survives a proto roundtrip."""
        restored = from_proto(space_to_proto(space))
        assert restored == space


class TestMultiBinaryPoint:
    def test_value(self):
        """Test serialization of [True, False, True] to MultiBinaryPoint"""
        space = MultiBinary(3)
        point = to_proto(space, np.array([True, False, True], dtype=np.bool_))
        assert isinstance(
            point, MultiBinaryPoint
        ), "Serialized point should be MultiBinaryPoint"
        assert len(point.values) == 3, "MultiBinaryPoint should have 3 values"
        assert list(point.values) == [
            True,
            False,
            True,
        ], "MultiBinaryPoint values should be [True, False, True]"


class TestMultiBinarySpace:

    # note we can skip the empty case because MultiBinary(0) is not a valid gymnasium space

    def test_single_value(self):
        """Test serialization of a single binary value to MultiBinarySpace"""
        space = MultiBinary(1)
        proto = space_to_proto(space)
        assert isinstance(
            proto, MultiBinarySpace
        ), "Serialized space should be MultiBinarySpace"
        assert proto.shape == 1, "MultiBinarySpace shape should be 1"

    def test_simple_shape(self):
        """Test serialization of MultiBinary(3) to MultiBinarySpace"""
        space = MultiBinary(3)
        proto = space_to_proto(space)
        assert isinstance(
            proto, MultiBinarySpace
        ), "Serialized space should be MultiBinarySpace"
        assert proto.shape == 3, "MultiBinarySpace shape should be 3"

    @pytest.mark.skip(reason="MultiBinary shape is not supported yet")
    def test_complex_shape(self):
        """Test serialization of a multi-dimensional shape to MultiBinarySpace"""
        space = MultiBinary((2, 3))
        proto = space_to_proto(space)
        assert isinstance(
            proto, MultiBinarySpace
        ), "Serialized space should be MultiBinarySpace"
        assert proto.shape == (2, 3), "MultiBinarySpace shape should be (2, 3)"

    @pytest.mark.parametrize(
        "n",
        [
            1,
            3,
            # (2, 3), # not supported yet
        ],
        ids=[
            "single_value",
            "simple_shape",
            # "complex_shape", # not supported yet
        ],
    )
    def test_roundtrip(self, n: int | tuple[int, ...]):
        """A MultiBinary space survives a proto roundtrip."""
        space = MultiBinary(n)
        restored = from_proto(space_to_proto(space))
        assert restored == space


class TestMultiDiscretePoint:

    def test_empty(self):
        """Test serialization of empty array to MultiDiscretePoint"""
        space = MultiDiscrete([])
        point = to_proto(space, np.array([], dtype=np.int64))
        assert isinstance(
            point, MultiDiscretePoint
        ), "Serialized point should be MultiDiscretePoint"
        assert len(point.values) == 0, "MultiDiscretePoint should have no values"

    def test_value(self):
        """Test serialization of [1, 2, 3] to MultiDiscretePoint"""
        space = MultiDiscrete([5, 5, 5])
        point = to_proto(space, np.array([1, 2, 3], dtype=np.int64))
        assert isinstance(
            point, MultiDiscretePoint
        ), "Serialized point should be MultiDiscretePoint"
        assert len(point.values) == 3, "MultiDiscretePoint should have 3 values"
        assert list(point.values) == [
            1,
            2,
            3,
        ], "MultiDiscretePoint values should be [1, 2, 3]"

    @pytest.mark.parametrize(
        "point",
        [
            np.array([], dtype=np.int64),
            np.array([1, 2, 3], dtype=np.int64),
        ],
        ids=["empty", "value"],
    )
    def test_roundtrip(self, point: np.ndarray):
        """An int64 array survives a proto roundtrip."""
        if point.size == 0:
            space = MultiDiscrete([])
        else:
            space = MultiDiscrete([5] * len(point))
        restored = from_proto(to_proto(space, point))
        assert np.array_equal(restored, point)


class TestMultiDiscreteSpace:

    def test_empty(self):
        """Test serialization of an empty nvec to MultiDiscreteSpace"""
        space = MultiDiscrete([])
        proto = space_to_proto(space)
        assert isinstance(
            proto, MultiDiscreteSpace
        ), "Serialized space should be MultiDiscreteSpace"
        assert len(proto.high) == 0, "MultiDiscreteSpace should have no high values"

    @pytest.mark.skip(reason="MultiDiscrete nested shape is not supported")
    def test_nested_shape(self):
        """Test serialization of [[2, 2], [3, 3]] to MultiDiscreteSpace"""
        space = MultiDiscrete(np.array([[2, 2], [3, 3]], dtype=np.int64))
        proto = space_to_proto(space)
        assert isinstance(
            proto, MultiDiscreteSpace
        ), "Serialized space should be MultiDiscreteSpace"
        assert len(proto.high) == 4, "MultiDiscreteSpace should have 4 high values"
        assert list(proto.high) == [
            2,
            2,
            3,
            3,
        ], "MultiDiscreteSpace high should be flattened [2, 2, 3, 3]"

    def test_simple_shape(self):
        """Test serialization of [2, 3, 4] to MultiDiscreteSpace"""
        space = MultiDiscrete([2, 3, 4])
        proto = space_to_proto(space)
        assert isinstance(
            proto, MultiDiscreteSpace
        ), "Serialized space should be MultiDiscreteSpace"
        assert len(proto.high) == 3, "MultiDiscreteSpace should have 3 high values"
        assert list(proto.high) == [
            2,
            3,
            4,
        ], "MultiDiscreteSpace high should be [2, 3, 4]"

    @pytest.mark.skip(reason="MultiDiscrete start is not supported yet")
    def test_start(self):
        """Test serialization of a non-zero start to MultiDiscreteSpace"""
        space = MultiDiscrete([2, 3, 4], start=[1, 2, 3])
        proto = space_to_proto(space)
        assert isinstance(
            proto, MultiDiscreteSpace
        ), "Serialized space should be MultiDiscreteSpace"
        assert len(proto.high) == 3, "MultiDiscreteSpace should have 3 high values"
        assert list(proto.high) == [
            2,
            3,
            4,
        ], "MultiDiscreteSpace high should be [2, 3, 4]"

    @pytest.mark.parametrize(
        "space",
        [
            MultiDiscrete([]),
            MultiDiscrete([2]),
            # MultiDiscrete(np.array([[2, 2], [3, 3]], dtype=np.int64)), # not supported yet
            MultiDiscrete([2, 3, 4]),
            # MultiDiscrete([2, 3, 4], start=[1, 2, 3]), # not supported yet
        ],
        ids=[
            "empty",
            "single_value",
            # "nested_shape", # not supported yet
            "simple_shape",
            # "start", # not supported yet
        ],
    )
    def test_roundtrip(self, space: MultiDiscrete):
        """A MultiDiscrete space survives a proto roundtrip."""
        restored = from_proto(space_to_proto(space))
        assert restored.dtype == space.dtype
        assert restored.shape == space.shape
        assert np.array_equal(restored.nvec, space.nvec)
        assert np.array_equal(restored.start, space.start)


class TestBoxPoint:

    def test_empty(self):
        """Test serialization of empty array to BoxPoint"""
        space = Box(low=0, high=1, shape=(0,), dtype=np.float32)
        point = to_proto(space, np.array([], dtype=np.float32))
        assert isinstance(point, BoxPoint), "Serialized point should be BoxPoint"
        assert len(point.values) == 0, "BoxPoint should have no values"
        assert point.dtype == DType.FLOAT32, "BoxPoint dtype should be FLOAT32"

    def test_value(self):
        """Test serialization of [1.0, 2.0, 3.0] to BoxPoint"""
        space = Box(low=0, high=10, shape=(3,), dtype=np.float32)
        point = to_proto(space, np.array([1.0, 2.0, 3.0], dtype=np.float32))
        assert isinstance(point, BoxPoint), "Serialized point should be BoxPoint"
        assert len(point.values) == 3, "BoxPoint should have 3 values"
        assert list(point.values) == [
            1.0,
            2.0,
            3.0,
        ], "BoxPoint values should be [1.0, 2.0, 3.0]"
        assert point.dtype == DType.FLOAT32, "BoxPoint dtype should be FLOAT32"
        assert list(point.shape) == [3], "BoxPoint shape should be [3]"

    def test_value_with_shape(self):
        """Test serialization of [[1.0, 2.0], [1.0, 2.0]] to BoxPoint"""
        space = Box(low=0, high=10, shape=(2, 2), dtype=np.float32)
        point = to_proto(space, np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32))
        assert isinstance(point, BoxPoint), "Serialized point should be BoxPoint"
        assert len(point.values) == 4, "BoxPoint should have 4 values (flattened)"
        assert list(point.values) == [
            1.0,
            2.0,
            1.0,
            2.0,
        ], "BoxPoint values should be flattened [1.0, 2.0, 1.0, 2.0]"
        assert point.dtype == DType.FLOAT32, "BoxPoint dtype should be FLOAT32"
        assert list(point.shape) == [2, 2], "BoxPoint shape should be [2, 2]"

    @pytest.mark.parametrize(
        "point",
        [
            np.array([], dtype=np.float32),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32),
        ],
        ids=["empty", "value", "value_with_shape"],
    )
    def test_roundtrip(self, point: np.ndarray):
        """A float32 array survives a proto roundtrip, including empty and nested shapes."""
        space = Box(low=0, high=10, shape=point.shape, dtype=np.float32)
        restored = from_proto(to_proto(space, point))
        assert restored.dtype == point.dtype
        assert restored.shape == point.shape
        assert np.array_equal(restored, point)

    @pytest.mark.parametrize(
        "dtype",
        NUMPY_DTYPE_TO_PROTO_DTYPE_MAPPING,
        ids=lambda dtype: np.dtype(dtype).name,
    )
    def test_dtype_roundtrip(self, dtype: type):
        """A modest value survives a proto roundtrip for every supported dtype."""
        space = Box(0, 1, shape=(1,), dtype=dtype)
        point = np.array([1], dtype=dtype)
        restored = from_proto(to_proto(space, point))
        assert restored.dtype == point.dtype
        assert np.array_equal(restored, point)


class TestBoxSpace:
    def test_finite_bounds(self):
        space = Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        proto = space_to_proto(space)
        assert isinstance(proto, BoxSpace)
        assert len(proto.dimensions) == 2
        for dim in proto.dimensions:
            assert dim.HasField("low")
            assert dim.HasField("high")
            assert dim.low == 0.0
            assert dim.high == 1.0

    def test_unbounded_omits_fields(self):
        space = Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        proto = space_to_proto(space)
        assert len(proto.dimensions) == 2
        for dim in proto.dimensions:
            assert not dim.HasField("low")
            assert not dim.HasField("high")

    def test_mixed_bounds(self):
        space = Box(
            low=np.array([-np.inf, 0.0], dtype=np.float32),
            high=np.array([0.0, np.inf], dtype=np.float32),
        )
        proto = space_to_proto(space)
        assert not proto.dimensions[0].HasField("low")
        assert proto.dimensions[0].HasField("high")
        assert proto.dimensions[1].HasField("low")
        assert not proto.dimensions[1].HasField("high")

    @pytest.mark.parametrize(
        "space",
        [
            Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            Box(low=0.0, high=np.inf, shape=(2, 2), dtype=np.float32),
            Box(low=-np.inf, high=0.0, shape=(2, 2), dtype=np.float32),
            Box(
                low=np.array([-np.inf, 0.0], dtype=np.float32),
                high=np.array([0.0, np.inf], dtype=np.float32),
            ),
        ],
        ids=["finite", "unbounded_high", "unbounded_low", "mixed_inf"],
    )
    def test_roundtrip(self, space: Box):
        restored = from_proto(space_to_proto(space))
        assert restored == space

    @pytest.mark.parametrize(
        "dtype",
        NUMPY_DTYPE_TO_PROTO_DTYPE_MAPPING,
        ids=lambda dtype: np.dtype(dtype).name,
    )
    def test_dtype_roundtrip(self, dtype: type):
        """Modest finite bounds survive a proto roundtrip for every supported dtype."""
        space = Box(0, 1, shape=(1,), dtype=dtype)
        assert from_proto(space_to_proto(space)) == space


class TestTextPoint:

    def test_value(self):
        """Test serialization of a string to TextPoint"""
        space = Text(max_length=16)
        point = to_proto(space, "six seven")
        assert isinstance(point, TextPoint), "Serialized point should be TextPoint"
        assert point.value == "six seven", "TextPoint value should be 'six seven'"

    def test_empty_string(self):
        """Test serialization of an empty string to TextPoint"""
        space = Text(max_length=16)
        point = to_proto(space, "")
        assert isinstance(point, TextPoint), "Serialized point should be TextPoint"
        assert point.value == "", "TextPoint value should be empty"

    @pytest.mark.parametrize(
        "value",
        [
            "six seven",
            "",
            "one23",
        ],
        ids=["numeric", "empty_string", "alphanumeric"],
    )
    def test_roundtrip(self, value: str):
        """A string survives a proto roundtrip, including the empty string."""
        space = Text(max_length=16)
        restored = from_proto(to_proto(space, value))
        assert restored == value


class TestTextSpace:

    def test_defaults(self):
        """All fields are serialized verbatim, including Gymnasium's own defaults."""
        space = Text(max_length=12)
        proto = space_to_proto(space)
        assert isinstance(proto, TextSpace), "Serialized space should be TextSpace"
        assert proto.max_length == 12, "TextSpace max_length should be 12"
        assert proto.min_length == 1, "min_length should be Gymnasium's default of 1"
        assert proto.charset == "".join(
            sorted(alphanumeric)
        ), "charset should be the sorted default alphanumeric set"

    def test_explicit_charset_is_sorted(self):
        """Charset is serialized as a deterministic, deduplicated sorted string."""
        space = Text(max_length=8, min_length=2, charset="cba")
        proto = space_to_proto(space)
        assert proto.min_length == 2, "TextSpace min_length should be 2"
        assert proto.charset == "abc", "Charset should be sorted/deduplicated to 'abc'"

    def test_empty_charset_is_literal(self):
        """An empty character set serializes to an empty charset (the empty set)."""
        space = Text(max_length=8, charset="")
        proto = space_to_proto(space)
        assert (
            proto.charset == ""
        ), "Empty character set should serialize to an empty charset"

    @pytest.mark.parametrize(
        "space",
        [
            Text(max_length=12),
            Text(max_length=8, min_length=2, charset="cba"),
            Text(max_length=8, charset=""),
        ],
        ids=["defaults", "explicit_charset_is_sorted", "empty_charset_is_literal"],
    )
    def test_roundtrip(self, space: Text):
        """A Text space survives a proto roundtrip."""
        restored = from_proto(space_to_proto(space))
        assert restored == space


class TestDictPoint:

    def test_empty(self):
        """Test serialization of empty dict to DictPoint"""
        space = Dict({})
        point = to_proto(space, {})
        assert isinstance(point, DictPoint), "Serialized point should be DictPoint"
        assert len(point.values) == 0, "DictPoint should have no values"

    def test_value(self):
        """Test serialization of {'a': 1, 'b': [1.0, 2.0]} to DictPoint"""
        space = Dict(
            {"a": Discrete(10), "b": Box(low=0, high=10, shape=(2,), dtype=np.float32)}
        )
        point = to_proto(space, {"a": 1, "b": np.array([1.0, 2.0], dtype=np.float32)})
        assert isinstance(point, DictPoint), "Serialized point should be DictPoint"
        assert len(point.values) == 2, "DictPoint should have 2 values"
        assert "a" in point.values, "DictPoint should have 'a' key"
        assert "b" in point.values, "DictPoint should have 'b' key"
        assert (
            point.values["a"].discrete_point.value == 1
        ), "DictPoint['a'] should be DiscretePoint with value 1"
        assert (
            len(point.values["b"].box_point.values) == 2
        ), "DictPoint['b'] should be BoxPoint with 2 values"
        assert list(point.values["b"].box_point.values) == [
            1.0,
            2.0,
        ], "DictPoint['b'] values should be [1.0, 2.0]"

    def test_nested(self):
        """Test serialization of a DictPoint nested inside another DictPoint"""
        space = Dict(
            {
                "a": Discrete(10),
                "inner": Dict({"b": Box(low=0, high=10, shape=(2,), dtype=np.float32)}),
            }
        )
        point = to_proto(
            space,
            {"a": 1, "inner": {"b": np.array([1.0, 2.0], dtype=np.float32)}},
        )
        assert isinstance(point, DictPoint), "Serialized point should be DictPoint"
        assert len(point.values) == 2, "DictPoint should have 2 values"
        assert (
            point.values["a"].discrete_point.value == 1
        ), "DictPoint['a'] should be DiscretePoint with value 1"
        assert (
            point.values["inner"].WhichOneof("point") == "dict_point"
        ), "DictPoint['inner'] should be a DictPoint"
        inner = point.values["inner"].dict_point
        assert len(inner.values) == 1, "Nested DictPoint should have 1 value"
        assert list(inner.values["b"].box_point.values) == [
            1.0,
            2.0,
        ], "Nested DictPoint['b'] values should be [1.0, 2.0]"

    @pytest.mark.parametrize(
        "space, point",
        [
            (Dict({}), {}),
            (
                Dict(
                    {
                        "a": Discrete(10),
                        "b": Box(low=0, high=10, shape=(2,), dtype=np.float32),
                    }
                ),
                {"a": 1, "b": np.array([1.0, 2.0], dtype=np.float32)},
            ),
            (
                Dict(
                    {
                        "a": Discrete(10),
                        "inner": Dict(
                            {"b": Box(low=0, high=10, shape=(2,), dtype=np.float32)}
                        ),
                    }
                ),
                {"a": 1, "inner": {"b": np.array([1.0, 2.0], dtype=np.float32)}},
            ),
        ],
        ids=["empty", "value", "nested"],
    )
    def test_roundtrip(self, space: Dict, point: dict):
        """A dict point survives a proto roundtrip, including an empty or nested dict."""

        def assert_point_equal(restored, value):
            if isinstance(value, dict):
                assert restored.keys() == value.keys()
                for key, item in value.items():
                    assert_point_equal(restored[key], item)
            elif isinstance(value, np.ndarray):
                assert restored.dtype == value.dtype
                assert restored.shape == value.shape
                assert np.array_equal(restored, value)
            else:
                assert restored == value

        assert_point_equal(from_proto(to_proto(space, point)), point)


class TestDictSpace:

    def test_empty(self):
        """Test serialization of an empty Dict to DictSpace"""
        space = Dict({})
        proto = space_to_proto(space)
        assert isinstance(proto, DictSpace), "Serialized space should be DictSpace"
        assert len(proto.spaces) == 0, "DictSpace should have no subspaces"

    def test_value(self):
        """Test serialization of a Dict of Discrete and Box to DictSpace"""
        space = Dict(
            {"a": Discrete(10), "b": Box(low=0, high=10, shape=(2,), dtype=np.float32)}
        )
        proto = space_to_proto(space)
        assert isinstance(proto, DictSpace), "Serialized space should be DictSpace"
        assert len(proto.spaces) == 2, "DictSpace should have 2 subspaces"
        assert "a" in proto.spaces, "DictSpace should have 'a' key"
        assert "b" in proto.spaces, "DictSpace should have 'b' key"
        assert (
            proto.spaces["a"].WhichOneof("space") == "discrete_space"
        ), "DictSpace['a'] should be a DiscreteSpace"
        assert (
            proto.spaces["a"].discrete_space.high == 10
        ), "DictSpace['a'] high should be 10"
        assert (
            proto.spaces["b"].WhichOneof("space") == "box_space"
        ), "DictSpace['b'] should be a BoxSpace"
        box = proto.spaces["b"].box_space
        assert list(box.shape_dimensions) == [2], "DictSpace['b'] shape should be [2]"
        assert len(box.dimensions) == 2, "DictSpace['b'] should have 2 dimensions"
        assert box.dtype == DType.FLOAT32, "DictSpace['b'] dtype should be FLOAT32"
        for dim in box.dimensions:
            assert dim.low == 0.0, "DictSpace['b'] low should be 0.0"
            assert dim.high == 10.0, "DictSpace['b'] high should be 10.0"

    def test_nested(self):
        """Test serialization of a DictSpace nested inside another DictSpace"""
        space = Dict(
            {
                "a": Discrete(10),
                "inner": Dict({"b": Box(low=0, high=10, shape=(2,), dtype=np.float32)}),
            }
        )
        proto = space_to_proto(space)
        assert isinstance(proto, DictSpace), "Serialized space should be DictSpace"
        assert len(proto.spaces) == 2, "DictSpace should have 2 subspaces"
        assert (
            proto.spaces["a"].WhichOneof("space") == "discrete_space"
        ), "DictSpace['a'] should be a DiscreteSpace"
        assert (
            proto.spaces["a"].discrete_space.high == 10
        ), "DictSpace['a'] high should be 10"
        assert (
            proto.spaces["inner"].WhichOneof("space") == "dict_space"
        ), "DictSpace['inner'] should be a DictSpace"
        inner = proto.spaces["inner"].dict_space
        assert len(inner.spaces) == 1, "Nested DictSpace should have 1 subspace"
        assert (
            inner.spaces["b"].WhichOneof("space") == "box_space"
        ), "Nested DictSpace['b'] should be a BoxSpace"
        assert list(inner.spaces["b"].box_space.shape_dimensions) == [
            2
        ], "Nested DictSpace['b'] shape should be [2]"

    @pytest.mark.parametrize(
        "space",
        [
            Dict({}),
            Dict(
                {
                    "a": Discrete(10),
                    "b": Box(low=0, high=10, shape=(2,), dtype=np.float32),
                }
            ),
            Dict(
                {
                    "a": Discrete(10),
                    "inner": Dict(
                        {"b": Box(low=0, high=10, shape=(2,), dtype=np.float32)}
                    ),
                }
            ),
        ],
        ids=["empty", "value", "nested"],
    )
    def test_roundtrip(self, space: Dict):
        """A Dict space survives a proto roundtrip."""
        restored = from_proto(space_to_proto(space))
        assert restored == space


class TestPoint:
    """Test serialization through the generic Point wrapper"""

    def test_box_point(self):
        """Test serialization of [1.0, 2.0, 3.0] produces BoxPoint in Point"""
        space = Box(low=0, high=10, shape=(3,), dtype=np.float32)
        point = to_proto(space, np.array([1.0, 2.0, 3.0], dtype=np.float32))
        assert isinstance(point, BoxPoint), "Serialized point should be BoxPoint"
        assert list(point.values) == [
            1.0,
            2.0,
            3.0,
        ], "BoxPoint values should be [1.0, 2.0, 3.0]"

    def test_discrete_point(self):
        """Test serialization of 1 produces DiscretePoint in Point"""
        space = Discrete(10)
        point = to_proto(space, 1)
        assert isinstance(
            point, DiscretePoint
        ), "Serialized point should be DiscretePoint"
        assert point.value == 1, "DiscretePoint value should be 1"

    def test_multi_discrete_point(self):
        """Test serialization of [1, 2, 3] produces MultiDiscretePoint in Point"""
        space = MultiDiscrete([5, 5, 5])
        point = to_proto(space, np.array([1, 2, 3], dtype=np.int64))
        assert isinstance(
            point, MultiDiscretePoint
        ), "Serialized point should be MultiDiscretePoint"
        assert list(point.values) == [
            1,
            2,
            3,
        ], "MultiDiscretePoint values should be [1, 2, 3]"

    def test_text_point(self):
        """Test serialization of a string produces TextPoint"""
        space = Text(max_length=16)
        point = to_proto(space, "abc")
        assert isinstance(point, TextPoint), "Serialized point should be TextPoint"
        assert point.value == "abc", "TextPoint value should be 'abc'"

    def test_multi_binary_point(self):
        """Test serialization of [True, False, True] produces MultiBinaryPoint in Point"""
        space = MultiBinary(3)
        point = to_proto(space, np.array([True, False, True], dtype=np.bool_))
        assert isinstance(
            point, MultiBinaryPoint
        ), "Serialized point should be MultiBinaryPoint"
        assert list(point.values) == [
            True,
            False,
            True,
        ], "MultiBinaryPoint values should be [True, False, True]"

    def test_dict_point(self):
        """Test serialization of {'a': 1, 'b': [1.0, 2.0]} produces DictPoint in Point"""
        space = Dict(
            {"a": Discrete(10), "b": Box(low=0, high=10, shape=(2,), dtype=np.float32)}
        )
        point = to_proto(space, {"a": 1, "b": np.array([1.0, 2.0], dtype=np.float32)})
        assert isinstance(point, DictPoint), "Serialized point should be DictPoint"
        assert len(point.values) == 2, "DictPoint should have 2 values"
        assert (
            point.values["a"].discrete_point.value == 1
        ), "DictPoint['a'] should be DiscretePoint with value 1"
        assert list(point.values["b"].box_point.values) == [
            1.0,
            2.0,
        ], "DictPoint['b'] values should be [1.0, 2.0]"
