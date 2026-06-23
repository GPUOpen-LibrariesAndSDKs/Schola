# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Unit tests for ONNX export helpers in ``schola.core.model``."""

from onnx import helper, TensorProto
import pytest
import torch as th
import numpy as np

from schola.core.model import (
    assert_shapes_fully_resolved,
    emulate_nne_seq_dim,
    fix_lstm_output_shapes_for_onnx,
    fix_slice_nodes_for_onnx,
    fold_static_shape_gather_constants,
    normalize_shape_slices_to_gather,
    patch_lstm_layers_for_onnx_export,
    reshape_lstm_output_hook,
    validate_exported_onnx_state_shapes,
    StateMetadata,
)
import onnx_ir as ir
import onnx_ir.passes.common.shape_inference  # noqa: F401


def test_emulate_nne_seq_dim_lstm_state():
    assert emulate_nne_seq_dim(["batch_size", 1, 64]) == -1
    assert emulate_nne_seq_dim([-1, 1, 64]) == -1
    assert emulate_nne_seq_dim([-1, -1, 64]) == 1
    assert emulate_nne_seq_dim([-1, 10, 32]) == -1
    assert emulate_nne_seq_dim([-1, -1, 32]) == 1


def test_validate_exported_onnx_state_shapes_rejects_leaked_dynamic_dim():
    state_in = helper.make_tensor_value_info(
        "state_in_actor_h", TensorProto.FLOAT, ["batch_size", None, 64]
    )
    state_out = helper.make_tensor_value_info(
        "state_out_actor_h", TensorProto.FLOAT, ["batch_size", None, 64]
    )
    graph = helper.make_graph([], "test_graph", [state_in], [state_out])
    model_proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model = ir.from_proto(model_proto)
    metadata = {
        "state_in_actor_h": StateMetadata(has_seq_dim=False),
    }
    with pytest.raises(ValueError, match="has_seq_dim=False"):
        validate_exported_onnx_state_shapes(model, metadata)


def test_reshape_lstm_output_hook_uses_input_batch_axis():
    lstm = th.nn.LSTM(input_size=4, hidden_size=8, batch_first=True)
    lstm.eval()
    x = th.randn(3, 2, 4)
    h0 = th.zeros(1, 3, 8)
    c0 = th.zeros(1, 3, 8)
    out, state = lstm(x, (h0, c0))
    _, (hn, cn) = reshape_lstm_output_hook(lstm, (x, (h0, c0)), (out, state))
    assert hn.shape == (1, 3, 8)
    assert cn.shape == (1, 3, 8)


def test_patch_lstm_layers_for_onnx_export_registers_all_lstms():
    module = th.nn.Sequential(th.nn.LSTM(4, 8), th.nn.Linear(8, 2))
    handles = patch_lstm_layers_for_onnx_export(module)
    try:
        assert len(handles) == 1
        assert isinstance(handles[0], th.utils.hooks.RemovableHandle)
    finally:
        for handle in handles:
            handle.remove()


def test_fix_slice_nodes_for_onnx_promotes_scalar_initializer():
    from unittest.mock import MagicMock

    model = MagicMock()
    node = MagicMock()
    node.op_type = "Slice"
    initializer = MagicMock()
    initializer.is_initializer.return_value = True
    initializer.shape = MagicMock(rank=MagicMock(return_value=0))
    initializer.name = "starts"
    initializer.const_value = MagicMock()
    initializer.const_value.numpy.return_value = np.array(0, dtype=np.int64)
    node.inputs = [initializer]
    model.graph.all_nodes.return_value = [node]

    fix_slice_nodes_for_onnx(model)

    assert initializer.shape == ir.Shape((1,))
    assert initializer.const_value.shape == ir.Shape((1,))


def _make_shape_slice_reshape_model(use_constant_nodes: bool = False):
    x = helper.make_tensor_value_info(
        "x", TensorProto.FLOAT, [1, "batch_size", 1, 64]
    )
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, None)

    def make_slice_tensors(name: str, start: int, end: int):
        return (
            helper.make_tensor(f"{name}_starts", TensorProto.INT64, [1], [start]),
            helper.make_tensor(f"{name}_ends", TensorProto.INT64, [1], [end]),
        )

    batch_st, batch_en = make_slice_tensors("batch", 1, 2)
    one_st, one_en = make_slice_tensors("one", 2, 3)
    hid_st, hid_en = make_slice_tensors("hid", 3, 4)
    c1 = helper.make_tensor("c1", TensorProto.INT64, [1], [1])
    if use_constant_nodes:
        nodes = [
            helper.make_node("Shape", ["x"], ["shape"]),
            helper.make_node(
                "Constant", [], ["batch_starts"], value=batch_st
            ),
            helper.make_node("Constant", [], ["batch_ends"], value=batch_en),
            helper.make_node("Constant", [], ["one_starts"], value=one_st),
            helper.make_node("Constant", [], ["one_ends"], value=one_en),
            helper.make_node("Constant", [], ["hid_starts"], value=hid_st),
            helper.make_node("Constant", [], ["hid_ends"], value=hid_en),
            helper.make_node(
                "Slice", ["shape", "batch_starts", "batch_ends"], ["batch"]
            ),
            helper.make_node("Slice", ["shape", "one_starts", "one_ends"], ["one"]),
            helper.make_node("Slice", ["shape", "hid_starts", "hid_ends"], ["hid"]),
            helper.make_node("Mul", ["one", "hid"], ["mul"]),
            helper.make_node("Concat", ["c1", "batch", "mul"], ["concat"], axis=0),
            helper.make_node("Reshape", ["x", "concat"], ["out"]),
        ]
        initializers = [c1]
    else:
        nodes = [
            helper.make_node("Shape", ["x"], ["shape"]),
            helper.make_node("Slice", ["shape", "batch_starts", "batch_ends"], ["batch"]),
            helper.make_node("Slice", ["shape", "one_starts", "one_ends"], ["one"]),
            helper.make_node("Slice", ["shape", "hid_starts", "hid_ends"], ["hid"]),
            helper.make_node("Mul", ["one", "hid"], ["mul"]),
            helper.make_node("Concat", ["c1", "batch", "mul"], ["concat"], axis=0),
            helper.make_node("Reshape", ["x", "concat"], ["out"]),
        ]
        initializers = [batch_st, batch_en, one_st, one_en, hid_st, hid_en, c1]
    graph = helper.make_graph(
        nodes,
        "shape_slice_graph",
        [x],
        [out],
        initializer=initializers,
    )
    return ir.from_proto(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    )


def test_normalize_shape_slices_to_gather_rewrites_constant_shape_slices():
    model = _make_shape_slice_reshape_model(use_constant_nodes=True)
    assert sum(node.op_type == "Slice" for node in model.graph.all_nodes()) == 3

    normalize_shape_slices_to_gather(model)

    assert not any(node.op_type == "Slice" for node in model.graph.all_nodes())
    assert sum(node.op_type == "Gather" for node in model.graph.all_nodes()) == 3


def test_fold_static_shape_gather_constants_replaces_known_dims():
    model = _make_shape_slice_reshape_model(use_constant_nodes=True)
    normalize_shape_slices_to_gather(model)

    fold_static_shape_gather_constants(model)

    active_gathers = [
        node
        for node in model.graph.all_nodes()
        if node.op_type == "Gather"
        and any(
            user_input is not None and user_input.producer() is node
            for user_node in model.graph.all_nodes()
            for user_input in user_node.inputs
        )
    ]
    assert len(active_gathers) == 1
    shaped_input = active_gathers[0].inputs[0].producer().inputs[0]
    assert shaped_input.shape.dims[1] == "batch_size"


def test_normalize_shape_slices_to_gather_rewrites_shape_slices():
    model = _make_shape_slice_reshape_model()
    assert sum(node.op_type == "Slice" for node in model.graph.all_nodes()) == 3

    normalize_shape_slices_to_gather(model)

    assert not any(node.op_type == "Slice" for node in model.graph.all_nodes())
    assert sum(node.op_type == "Gather" for node in model.graph.all_nodes()) == 3
    shape_node = next(node for node in model.graph.all_nodes() if node.op_type == "Shape")
    for gather_node in model.graph.all_nodes():
        if gather_node.op_type != "Gather":
            continue
        data_input = gather_node.inputs[0]
        assert data_input is not None
        assert data_input.producer() is shape_node
        indices = gather_node.inputs[1]
        assert indices is not None and indices.is_initializer()


def test_assert_shapes_fully_resolved_accepts_allowed_dynamic_dims():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch_size", 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch_size", 4])
    graph = helper.make_graph([], "test_graph", [x], [y])
    model = ir.from_proto(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    )
    assert_shapes_fully_resolved(model, {"batch_size"})


def test_assert_shapes_fully_resolved_rejects_unknown_dim():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["unk__1", 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["unk__1", 4])
    graph = helper.make_graph([], "test_graph", [x], [y])
    model = ir.from_proto(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    )
    with pytest.raises(ValueError, match="unk__1"):
        assert_shapes_fully_resolved(model, {"batch_size"})


def test_assert_shapes_fully_resolved_checks_intermediate_tensors():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch_size", 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch_size", 4])
    mid = helper.make_tensor_value_info("mid", TensorProto.FLOAT, ["unk__1", 4])
    relu = helper.make_node("Relu", ["x"], ["mid"])
    graph = helper.make_graph([relu], "test_graph", [x], [y], value_info=[mid])
    model = ir.from_proto(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    )
    with pytest.raises(ValueError, match="mid"):
        assert_shapes_fully_resolved(model, {"batch_size"})


def test_fix_lstm_output_shapes_raises_on_unexpected_output_count():
    lstm_node = helper.make_node("LSTM", ["x", "w", "r"], ["Y", "Y_h"])
    graph = helper.make_graph([lstm_node], "lstm_graph", [], [])
    model = ir.from_proto(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    )
    with pytest.raises(ValueError, match="exactly 3 outputs"):
        fix_lstm_output_shapes_for_onnx(model)


def test_fix_lstm_output_shapes_raises_on_unexpected_state_rank():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, "batch", 4])
    w = helper.make_tensor_value_info("w", TensorProto.FLOAT, [1, 32, 4])
    r = helper.make_tensor_value_info("r", TensorProto.FLOAT, [1, 32, 8])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, "batch", 8])
    y_h = helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, [1, "batch", 8])
    y_c = helper.make_tensor_value_info("Y_c", TensorProto.FLOAT, [1, 1, "batch", 8])
    lstm_node = helper.make_node("LSTM", ["x", "w", "r"], ["Y", "Y_h", "Y_c"])
    graph = helper.make_graph(
        [lstm_node],
        "lstm_graph",
        [x, w, r],
        [y],
        value_info=[y_h, y_c],
    )
    model = ir.from_proto(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    )
    with pytest.raises(ValueError, match="unexpected Y_h rank"):
        fix_lstm_output_shapes_for_onnx(model)


def test_fix_lstm_output_shapes_unblocks_shape_inference():
    # Build a minimal graph with an LSTM whose Y_h/Y_c are mis-annotated as
    # rank-4, which makes onnx shape inference abort and leaves the downstream
    # tensor with no resolved shape.
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, "batch", 4])
    w = helper.make_tensor_value_info("w", TensorProto.FLOAT, [1, 32, 4])
    r = helper.make_tensor_value_info("r", TensorProto.FLOAT, [1, 32, 8])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, "batch", 8])
    y_h = helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, [1, 1, "batch", 8])
    y_c = helper.make_tensor_value_info("Y_c", TensorProto.FLOAT, [1, 1, "batch", 8])
    relu_out = helper.make_tensor_value_info("relu", TensorProto.FLOAT, None)

    lstm_node = helper.make_node(
        "LSTM", ["x", "w", "r"], ["Y", "Y_h", "Y_c"], hidden_size=8
    )
    relu_node = helper.make_node("Relu", ["Y_h"], ["relu"])
    graph = helper.make_graph(
        [lstm_node, relu_node],
        "lstm_graph",
        [x, w, r],
        [y, relu_out],
        value_info=[y_h, y_c],
    )
    model_proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model = ir.from_proto(model_proto)

    fix_lstm_output_shapes_for_onnx(model)

    lstm = next(n for n in model.graph.all_nodes() if n.op_type == "LSTM")
    # Y (rank-4, spec-correct) is preserved; Y_h/Y_c rank-4 annotations cleared.
    assert lstm.outputs[0].shape is not None
    assert lstm.outputs[1].shape is None
    assert lstm.outputs[2].shape is None

    # Shape inference should now succeed instead of aborting.
    ir.passes.common.shape_inference.infer_shapes(model)
