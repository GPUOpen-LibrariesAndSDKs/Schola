# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for the protobuf serialization"""

import pytest

from schola.core.protocols.protobuf.serialize import to_proto, space_to_proto
from schola.generated.Points_pb2 import *
from schola.generated.Spaces_pb2 import *
from schola.generated.DType_pb2 import *
import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary, Dict, Text


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


class TestTextPoint:

    def test_value(self):
        """Test serialization of a string to TextPoint"""
        space = Text(max_length=16)
        point = to_proto(space, "six seven")
        assert isinstance(point, TextPoint), "Serialized point should be TextPoint"
        assert point.value == "six seven", "TextPoint value should be 'hello'"

    def test_empty_string(self):
        """Test serialization of an empty string to TextPoint"""
        space = Text(max_length=16)
        point = to_proto(space, "")
        assert isinstance(point, TextPoint), "Serialized point should be TextPoint"
        assert point.value == "", "TextPoint value should be empty"


class TestTextSpace:

    def test_defaults(self):
        """Only max_length set; min_length/charset left unset so Gymnasium defaults apply."""
        space = Text(max_length=12)
        proto = space_to_proto(space)
        assert isinstance(proto, TextSpace), "Serialized space should be TextSpace"
        assert proto.max_length == 12, "TextSpace max_length should be 12"
        assert not proto.HasField("min_length"), "min_length should be left unset for default (1)"
        assert not proto.HasField("charset"), "charset should be left unset for default (alphanumeric)"

    def test_explicit_charset_is_sorted(self):
        """Charset is serialized as a deterministic, deduplicated sorted string."""
        space = Text(max_length=8, min_length=2, charset="cba")
        proto = space_to_proto(space)
        assert proto.min_length == 2, "TextSpace min_length should be 2"
        assert proto.charset == "abc", "Charset should be sorted/deduplicated to 'abc'"


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
