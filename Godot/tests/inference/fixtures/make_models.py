#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Generates the tiny ONNX models used by the inference tests.

Each model follows the contract of Schola's Python exporter (schola/core/model.py):
inputs are named "obs" or by observation Dict key, outputs "action" or by action Dict key,
recurrent memory is "state_in_<key>" -> "state_out_<key>", and dimension 0 is a dynamic batch.

The generated files are committed, so the tests do not need Python. Regenerate them with:
    pip install onnx onnxruntime
    python Godot/tests/inference/fixtures/make_models.py
The script then runs every model with onnxruntime and checks the expected outputs.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

FIXTURES_DIRECTORY = Path(__file__).resolve().parent
OPSET = 17
IR_VERSION = 8  # Readable by every ONNX Runtime release since 1.10.
BATCH = "batch_size"  # Dynamic batch dimension, as named by the exporter.
SEQUENCE = "seq_len"  # Dynamic sequence dimension, as named by the exporter.


def tensor(name, element_type, shape):
    return helper.make_tensor_value_info(name, element_type, shape)


def constant(name, values, element_type=TensorProto.FLOAT):
    dtype = helper.tensor_dtype_to_np_dtype(element_type)
    return numpy_helper.from_array(np.array(values, dtype=dtype), name)


def model(name, nodes, inputs, outputs, initializers=()):
    graph = helper.make_graph(nodes, name, inputs, outputs, list(initializers))
    result = helper.make_model(
        graph,
        producer_name="schola-test-fixtures",
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    result.ir_version = IR_VERSION
    onnx.checker.check_model(result, full_check=True)
    return result


def box_scale():
    # Box -> Box: action = 2 * obs, so tests can check values are really computed.
    return model(
        "box_scale",
        [helper.make_node("Mul", ["obs", "two"], ["action"])],
        [tensor("obs", TensorProto.FLOAT, [BATCH, 3])],
        [tensor("action", TensorProto.FLOAT, [BATCH, 3])],
        [constant("two", 2.0)],
    )


def discrete():
    # Box -> Discrete: action = argmax(obs), int64 with one value per batch entry.
    return model(
        "discrete",
        [helper.make_node("ArgMax", ["obs"], ["action"], axis=1, keepdims=0)],
        [tensor("obs", TensorProto.FLOAT, [BATCH, 4])],
        [tensor("action", TensorProto.INT64, [BATCH])],
    )


def multi_discrete():
    # MultiDiscrete -> MultiDiscrete: int64 in and out.
    return model(
        "multi_discrete",
        [helper.make_node("Identity", ["obs"], ["action"])],
        [tensor("obs", TensorProto.INT64, [BATCH, 2])],
        [tensor("action", TensorProto.INT64, [BATCH, 2])],
    )


def multi_binary():
    # MultiBinary -> MultiBinary: int8 in, bool out, as the exporter produces.
    return model(
        "multi_binary",
        [helper.make_node("Cast", ["obs"], ["action"], to=TensorProto.BOOL)],
        [tensor("obs", TensorProto.INT8, [BATCH, 3])],
        [tensor("action", TensorProto.BOOL, [BATCH, 3])],
    )


def dict_obs():
    # Dict observation {"a": Box(2), "b": Discrete} -> Box(3) = [a0, a1, b].
    return model(
        "dict_obs",
        [
            helper.make_node("Cast", ["b"], ["b_float"], to=TensorProto.FLOAT),
            helper.make_node("Unsqueeze", ["b_float", "axis_1"], ["b_column"]),
            helper.make_node("Concat", ["a", "b_column"], ["action"], axis=1),
        ],
        [
            tensor("a", TensorProto.FLOAT, [BATCH, 2]),
            tensor("b", TensorProto.INT64, [BATCH]),
        ],
        [tensor("action", TensorProto.FLOAT, [BATCH, 3])],
        [constant("axis_1", [1], TensorProto.INT64)],
    )


def dict_action():
    # Box(2) -> Dict action {"jump": Discrete = argmax(obs), "move": Box(2) = obs}.
    return model(
        "dict_action",
        [
            helper.make_node("ArgMax", ["obs"], ["jump"], axis=1, keepdims=0),
            helper.make_node("Identity", ["obs"], ["move"]),
        ],
        [tensor("obs", TensorProto.FLOAT, [BATCH, 2])],
        [
            tensor("jump", TensorProto.INT64, [BATCH]),
            tensor("move", TensorProto.FLOAT, [BATCH, 2]),
        ],
    )


def counter():
    # Recurrent without a sequence: action = state, new state = state + 1.
    # Starting from a zeroed state, the action counts 0, 1, 2, ... per step.
    return model(
        "counter",
        [
            helper.make_node("Identity", ["state_in_h"], ["action"]),
            helper.make_node("Add", ["state_in_h", "one"], ["state_out_h"]),
        ],
        [
            tensor("obs", TensorProto.FLOAT, [BATCH, 1]),
            tensor("state_in_h", TensorProto.FLOAT, [BATCH, 2]),
        ],
        [
            tensor("action", TensorProto.FLOAT, [BATCH, 2]),
            tensor("state_out_h", TensorProto.FLOAT, [BATCH, 2]),
        ],
        [constant("one", 1.0)],
    )


def counter_seq():
    # Recurrent with a sequence: state_in_h is [batch, seq, 2]. action = the newest entry,
    # state_out_h = newest + 1 (one entry, which the policy appends to the window).
    result = model(
        "counter_seq",
        [
            helper.make_node("Gather", ["state_in_h", "last"], ["newest"], axis=1),
            helper.make_node("Identity", ["newest"], ["action"]),
            helper.make_node("Add", ["newest", "one"], ["state_out_h"]),
        ],
        [
            tensor("obs", TensorProto.FLOAT, [BATCH, 1]),
            tensor("state_in_h", TensorProto.FLOAT, [BATCH, SEQUENCE, 2]),
        ],
        [
            tensor("action", TensorProto.FLOAT, [BATCH, 2]),
            tensor("state_out_h", TensorProto.FLOAT, [BATCH, 2]),
        ],
        [constant("last", -1, TensorProto.INT64), constant("one", 1.0)],
    )
    # The exporter also writes this metadata on the state input; ONNX Runtime cannot read it.
    for value_info in result.graph.input:
        if value_info.name == "state_in_h":
            helper.set_metadata_props(
                value_info, {"has_seq_dim": "True", "max_seq_len": "3", "seq_dim": "1"}
            )
    return result


def bad_unpaired_state():
    # state_in_h without a matching state_out_h.
    return model(
        "bad_unpaired_state",
        [helper.make_node("Identity", ["state_in_h"], ["action"])],
        [
            tensor("obs", TensorProto.FLOAT, [BATCH, 1]),
            tensor("state_in_h", TensorProto.FLOAT, [BATCH, 2]),
        ],
        [tensor("action", TensorProto.FLOAT, [BATCH, 2])],
    )


def bad_state_int():
    # State tensors must be float; this one is int64.
    return model(
        "bad_state_int",
        [
            helper.make_node("Identity", ["obs"], ["action"]),
            helper.make_node("Identity", ["state_in_h"], ["state_out_h"]),
        ],
        [
            tensor("obs", TensorProto.FLOAT, [BATCH, 1]),
            tensor("state_in_h", TensorProto.INT64, [BATCH, 2]),
        ],
        [
            tensor("action", TensorProto.FLOAT, [BATCH, 1]),
            tensor("state_out_h", TensorProto.INT64, [BATCH, 2]),
        ],
    )


MODELS = {
    "box_scale": box_scale,
    "discrete": discrete,
    "multi_discrete": multi_discrete,
    "multi_binary": multi_binary,
    "dict_obs": dict_obs,
    "dict_action": dict_action,
    "counter": counter,
    "counter_seq": counter_seq,
    "bad_unpaired_state": bad_unpaired_state,
    "bad_state_int": bad_state_int,
}


def check_outputs():
    import onnxruntime

    def run(name, feeds):
        session = onnxruntime.InferenceSession(str(FIXTURES_DIRECTORY / f"{name}.onnx"))
        return dict(
            zip([o.name for o in session.get_outputs()], session.run(None, feeds))
        )

    f32, i64, i8 = np.float32, np.int64, np.int8
    expected = {
        "box_scale": (
            {"obs": np.array([[1, 2, 3]], f32)},
            {"action": np.array([[2, 4, 6]], f32)},
        ),
        "discrete": (
            {"obs": np.array([[0, 5, 1, 2]], f32)},
            {"action": np.array([1], i64)},
        ),
        "multi_discrete": (
            {"obs": np.array([[3, 1]], i64)},
            {"action": np.array([[3, 1]], i64)},
        ),
        "multi_binary": (
            {"obs": np.array([[1, 0, 1]], i8)},
            {"action": np.array([[True, False, True]])},
        ),
        "dict_obs": (
            {"a": np.array([[1, 2]], f32), "b": np.array([4], i64)},
            {"action": np.array([[1, 2, 4]], f32)},
        ),
        "dict_action": (
            {"obs": np.array([[0, 9]], f32)},
            {"jump": np.array([1], i64), "move": np.array([[0, 9]], f32)},
        ),
        "counter": (
            {"obs": np.zeros((1, 1), f32), "state_in_h": np.array([[2, 2]], f32)},
            {"action": np.array([[2, 2]], f32), "state_out_h": np.array([[3, 3]], f32)},
        ),
        "counter_seq": (
            {
                "obs": np.zeros((1, 1), f32),
                "state_in_h": np.array([[[0, 0], [1, 1], [2, 2]]], f32),
            },
            {"action": np.array([[2, 2]], f32), "state_out_h": np.array([[3, 3]], f32)},
        ),
    }
    for name, (feeds, outputs) in expected.items():
        actual = run(name, feeds)
        for output_name, value in outputs.items():
            np.testing.assert_array_equal(actual[output_name], value, err_msg=name)
    print(f"Checked {len(expected)} models with onnxruntime {onnxruntime.__version__}.")


def main():
    for name, build in MODELS.items():
        onnx.save(build(), FIXTURES_DIRECTORY / f"{name}.onnx")
    print(f"Wrote {len(MODELS)} models to {FIXTURES_DIRECTORY}")
    check_outputs()


if __name__ == "__main__":
    main()
