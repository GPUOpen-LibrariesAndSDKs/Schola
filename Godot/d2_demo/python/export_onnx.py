#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Exports the trainer's saved Q-table policy to ONNX, so Godot runs it with ONNX Runtime (US7).

The model rounds the Box observation to grid cells, looks up that state's Q-values, and returns
the Discrete action with the highest value. Unvisited states have all-zero Q-values, so they pick
action 0, matching the Q-table policy in Godot.

    python export_onnx.py --policy ../policies/q_policy.json --out ../policies/q_policy.onnx
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 17
IR_VERSION = 8


def state_index(observation, low, sizes):
    cells = np.clip(np.round(observation) - low, 0, sizes - 1).astype(np.int64)
    strides = np.cumprod(np.concatenate(([1], sizes[::-1][:-1])))[::-1]
    return int(np.dot(cells, strides))


def build_table(q_table, low, sizes, n_actions):
    table = np.zeros((int(np.prod(sizes)), n_actions), np.float64)
    for key, values in q_table.items():
        observation = np.array([float(part) for part in key.split(",")])
        table[state_index(observation, low, sizes)] = values
    return table


def build_model(table, low, sizes):
    strides = np.cumprod(np.concatenate(([1], sizes[::-1][:-1])))[::-1]
    constants = [
        numpy_helper.from_array(low.astype(np.float32), "low"),
        numpy_helper.from_array(np.zeros_like(low, np.float32), "zero"),
        numpy_helper.from_array((sizes - 1).astype(np.float32), "max_cell"),
        numpy_helper.from_array(strides.astype(np.int64), "strides"),
        numpy_helper.from_array(np.array([1], np.int64), "axis_1"),
        numpy_helper.from_array(table, "q_values"),
    ]
    nodes = [
        helper.make_node("Round", ["obs"], ["rounded"]),
        helper.make_node("Sub", ["rounded", "low"], ["offset"]),
        helper.make_node("Max", ["offset", "zero"], ["above_min"]),
        helper.make_node("Min", ["above_min", "max_cell"], ["cell"]),
        helper.make_node("Cast", ["cell"], ["cell_index"], to=TensorProto.INT64),
        helper.make_node("Mul", ["cell_index", "strides"], ["weighted"]),
        helper.make_node("ReduceSum", ["weighted", "axis_1"], ["state"], keepdims=0),
        helper.make_node("Gather", ["q_values", "state"], ["state_q_values"], axis=0),
        helper.make_node("ArgMax", ["state_q_values"], ["action"], axis=1, keepdims=0),
    ]
    graph = helper.make_graph(
        nodes,
        "q_policy",
        [
            helper.make_tensor_value_info(
                "obs", TensorProto.FLOAT, ["batch_size", len(low)]
            )
        ],
        [helper.make_tensor_value_info("action", TensorProto.INT64, ["batch_size"])],
        constants,
    )
    model = helper.make_model(
        graph,
        producer_name="schola-d2-demo",
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model, full_check=True)
    return model


def check_model(path, q_table, n_actions):
    import onnxruntime

    session = onnxruntime.InferenceSession(str(path))
    for key, values in q_table.items():
        observation = np.array([[float(part) for part in key.split(",")]], np.float32)
        expected = int(np.argmax(values))
        actual = int(session.run(None, {"obs": observation})[0][0])
        if values[actual] != values[expected]:
            raise RuntimeError(
                f"state {key}: ONNX chose {actual}, Q-table chose {expected}"
            )
    return len(q_table)


def export(policy_path, out_path):
    data = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    observation_space = data["observation_space"]
    action_space = data["action_space"]
    if observation_space.get("type") != "Box" or action_space.get("type") != "Discrete":
        raise ValueError("expected a Box observation space and a Discrete action space")
    low = np.array(observation_space["low"], np.float64)
    high = np.array(observation_space["high"], np.float64)
    sizes = (high - low).astype(np.int64) + 1
    n_actions = int(action_space["n"])

    table = build_table(data["q"], low, sizes, n_actions)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(table, low, sizes), out_path)
    checked = check_model(out_path, data["q"], n_actions)
    return checked


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--policy", type=Path, default=here.parent / "policies" / "q_policy.json"
    )
    parser.add_argument(
        "--out", type=Path, default=here.parent / "policies" / "q_policy.onnx"
    )
    args = parser.parse_args()
    checked = export(args.policy, args.out)
    print(
        f"exported {args.out}; ONNX matches the Q-table on all {checked} trained states"
    )


if __name__ == "__main__":
    main()
