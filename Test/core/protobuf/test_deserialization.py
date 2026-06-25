# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnusedImport=false

"""Tests for protobuf-to-numpy / Gymnasium deserialization helpers."""

from schola.core.protocols.protobuf.deserialize import from_proto
from schola.generated.Points_pb2 import *
from schola.generated.Spaces_pb2 import *
from schola.generated.DType_pb2 import *
import numpy as np
import gymnasium.spaces as spaces


class TestDiscretePoint:
    def test_empty(self):
        point = DiscretePoint()
        assert from_proto(point) == 0, "Empty DiscretePoint should deserialize to 0"

    def test_value(self):
        point = DiscretePoint(value=5)
        assert (
            from_proto(point) == 5
        ), "DiscretePoint with value 5 should deserialize to 5"


class TestMultiBinaryPoint:
    def test_empty(self):
        point = MultiBinaryPoint()
        assert isinstance(
            from_proto(point), np.ndarray
        ), "Empty MultiBinaryPoint should deserialize to numpy array"
        assert from_proto(point).shape == (
            0,
        ), "Empty MultiBinaryPoint should deserialize to numpy array with shape (0,)"

    def test_value(self):
        point = MultiBinaryPoint(values=[True, False, True])
        assert np.all(
            from_proto(point) == np.array([True, False, True], dtype=np.bool_)
        ), "MultiBinaryPoint with values [True, False, True] should deserialize to [True, False, True]"


class TestMultiDiscretePoint:

    def test_empty(self):
        point = MultiDiscretePoint()
        assert isinstance(
            from_proto(point), np.ndarray
        ), "Empty MultiDiscretePoint should deserialize to numpy array"
        assert from_proto(point).shape == (
            0,
        ), "Empty MultiDiscretePoint should deserialize to numpy array with shape (0,)"

    def test_value(self):
        point = MultiDiscretePoint(values=[1, 2, 3])
        assert np.all(
            from_proto(point) == np.array([1, 2, 3], dtype=np.int64)
        ), "MultiDiscretePoint with values [1, 2, 3] should deserialize to [1, 2, 3]"


class TestBoxPoint:

    def test_empty(self):
        point = BoxPoint()
        assert isinstance(
            from_proto(point), np.ndarray
        ), "Empty BoxPoint should deserialize to numpy array"
        assert from_proto(point).shape == (
            0,
        ), "Empty BoxPoint should deserialize to numpy array with shape (0,)"

    def test_value(self):
        point = BoxPoint(values=[1.0, 2.0, 3.0], dtype=DType.FLOAT32, shape=[3])
        assert np.all(
            from_proto(point) == np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ), "BoxPoint with values [1.0, 2.0, 3.0] should deserialize to [1.0, 2.0, 3.0]"

    def test_value_with_shape(self):
        point = BoxPoint(values=[1.0, 2.0, 1.0, 2.0], dtype=DType.FLOAT32, shape=[2, 2])
        assert np.all(
            from_proto(point) == np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32)
        ), "BoxPoint with values [1.0, 2.0, 1.0, 2.0] and shape [2, 2] should deserialize to np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32)"


class TestTextPoint:
    def test_value(self):
        point = TextPoint(value="one two")
        assert (
            from_proto(point) == "one two"
        ), "TextPoint with value 'one two' should deserialize to 'one two'"

    def test_empty(self):
        point = TextPoint()
        assert from_proto(point) == "", "Empty TextPoint should deserialize to ''"


class TestTextSpace:
    def test_defaults(self):
        """All fields are copied verbatim; an empty charset maps to an empty set."""
        space = from_proto(TextSpace(max_length=10))
        assert isinstance(space, spaces.Text), "Should deserialize to a Text space"
        assert space.max_length == 10, "max_length should be 10"
        assert space.min_length == 0, "min_length should be copied verbatim"
        assert (
            space.character_set == frozenset()
        ), "empty charset should map to the empty set"

    def test_empty_charset_is_literal(self):
        """An explicit empty charset is the empty set (only the empty string is valid)."""
        space = from_proto(TextSpace(max_length=5, min_length=0, charset=""))
        assert space.character_set == frozenset(), "charset should be the empty set"
        assert space.contains(""), "empty string should be a valid point"
        assert not space.contains("a"), "no non-empty string should be valid"

    def test_explicit_fields(self):
        space = from_proto(TextSpace(max_length=8, min_length=2, charset="abc"))
        assert space.max_length == 8, "max_length should be 8"
        assert space.min_length == 2, "min_length should be 2"
        assert space.character_set == frozenset("abc"), "charset should be {a, b, c}"


class TestDictPoint:

    def test_empty(self):
        point = DictPoint()
        assert (
            from_proto(point) == {}
        ), "Empty DictPoint should deserialize to empty dict"

    def test_value(self):
        point = DictPoint(
            values={
                "a": Point(discrete_point=DiscretePoint(value=1)),
                "b": Point(
                    box_point=BoxPoint(
                        values=[1.0, 2.0], dtype=DType.FLOAT32, shape=[2]
                    )
                ),
            }
        )
        deserialized_point = from_proto(point)
        assert isinstance(
            deserialized_point, dict
        ), "DictPoint should deserialize to dict"
        assert (
            len(deserialized_point) == 2
        ), "DictPoint with values {'a': 1, 'b': [1.0, 2.0]} should deserialize to dict with 2 keys"
        assert (
            deserialized_point["a"] == 1
        ), "DictPoint with values {'a': 1, 'b': [1.0, 2.0]} should deserialize to dict with 'a' key equal to 1"
        assert np.all(
            deserialized_point["b"] == np.array([1.0, 2.0], dtype=np.float32)
        ), "DictPoint with values {'a': 1, 'b': [1.0, 2.0]} should deserialize to dict with 'b' key equal to np.array([1.0, 2.0], dtype=np.float32)"


class TestPoint:

    def test_box_point(self):
        point = Point(
            box_point=BoxPoint(values=[1.0, 2.0, 3.0], dtype=DType.FLOAT32, shape=[3])
        )
        assert np.all(
            from_proto(point) == np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ), "Point with box_point [1.0, 2.0, 3.0] should deserialize to np.array([1.0, 2.0, 3.0], dtype=np.float32)"

    def test_discrete_point(self):
        point = Point(discrete_point=DiscretePoint(value=1))
        assert (
            from_proto(point) == 1
        ), "Point with discrete_point 1 should deserialize to 1"

    def test_multi_discrete_point(self):
        point = Point(multi_discrete_point=MultiDiscretePoint(values=[1, 2, 3]))
        assert np.all(
            from_proto(point) == np.array([1, 2, 3], dtype=np.int64)
        ), "Point with multi_discrete_point [1, 2, 3] should deserialize to np.array([1, 2, 3], dtype=np.int64)"

    def test_multi_binary_point(self):
        point = Point(multi_binary_point=MultiBinaryPoint(values=[True, False, True]))
        assert np.all(
            from_proto(point) == np.array([True, False, True], dtype=np.bool_)
        ), "Point with multi_binary_point [True, False, True] should deserialize to np.array([True, False, True], dtype=np.bool_)"

    def test_text_point(self):
        point = Point(text_point=TextPoint(value="hello"))
        assert (
            from_proto(point) == "hello"
        ), "Point with text_point 'hello' should deserialize to 'hello'"

    def test_dict_point(self):
        point = Point(
            dict_point=DictPoint(
                values={
                    "a": Point(discrete_point=DiscretePoint(value=1)),
                    "b": Point(
                        box_point=BoxPoint(
                            values=[1.0, 2.0], dtype=DType.FLOAT32, shape=[2]
                        )
                    ),
                }
            )
        )
        deserialized_point = from_proto(point)
        assert isinstance(
            deserialized_point, dict
        ), "DictPoint should deserialize to dict"
        assert (
            len(deserialized_point) == 2
        ), "DictPoint with values {'a': 1, 'b': [1.0, 2.0]} should deserialize to dict with 2 keys"
        assert (
            deserialized_point["a"] == 1
        ), "DictPoint with values {'a': 1, 'b': [1.0, 2.0]} should deserialize to dict with 'a' key equal to 1"
        assert np.all(
            deserialized_point["b"] == np.array([1.0, 2.0], dtype=np.float32)
        ), "DictPoint with values {'a': 1, 'b': [1.0, 2.0]} should deserialize to dict with 'b' key equal to np.array([1.0, 2.0], dtype=np.float32)"
