# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""Unit tests for ONNX export helpers in ``schola.core.model``."""

import numpy as np
import onnx
from onnx import helper, TensorProto
import pytest
import torch as th

from schola.core.model import (
    emulate_nne_seq_dim,
    fix_slice_nodes_for_onnx,
    patch_lstm_layers_for_onnx_export,
    reshape_lstm_output_hook,
    validate_exported_onnx_state_shapes,
    StateMetadata,
)
import onnx_ir as ir


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
