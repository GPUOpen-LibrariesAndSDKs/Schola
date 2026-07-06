# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""ONNX IR post-export passes and LSTM export hooks for Unreal NNE compatibility."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import onnx_ir as ir
import onnx_ir.passes.common.shape_inference
import onnxscript.optimizer
import torch as th

from schola.core.onnx_validation import (
    StateMetadataLike,
    check_resolved_dims,
    dim_to_repr,
    ir_tensor_dims,
    is_dynamic_dim_repr,
    validate_exported_onnx_state_shapes,
)

logger = logging.getLogger(__name__)


def _flatten_int64(values: object) -> npt.NDArray[np.int64]:
    """Coerce ONNX tensor payload values to a flattened int64 vector."""
    return np.asarray(values, dtype=np.int64).ravel()


def _scalar_int64(values: object) -> npt.NDArray[np.int64]:
    """Coerce ONNX tensor payload values to a length-1 int64 vector."""
    return np.array(values, dtype=np.int64, ndmin=1)


def postprocess_exported_onnx(
    model: ir.Model,
    *,
    state_metadata: Mapping[str, StateMetadataLike],
    allowed_dynamic_dims: set[str],
) -> None:
    """
    Run the ONNX post-export repair pipeline for Unreal NNE compatibility.

    Order matters: ``torch.onnx.export`` is called with ``optimize=False`` so
    Shape/Slice/Gather rewrites can run before ``onnxscript.optimizer.optimize``.
    Do not enable torch's built-in optimizer without re-running this pipeline.

    The ``normalize_shape_slices_to_gather`` -> ``fold_static_shape_gather_constants``
    -> ``onnxscript.optimizer.optimize`` stages are load-bearing: they are what
    resolve the auto-generated ``unk__*`` symbolic dims left by the dynamo export
    (from LSTM decomposition and output slicing) down to concrete integers, leaving
    only ``batch_size``/``seq_len`` dynamic. Removing them will cause
    ``assert_shapes_fully_resolved`` to fail regardless of how LSTM state is reshaped.

    Stages
    ------
    1. Fix invalid 0-D Slice initializer shapes.
    2. Clear mis-ranked LSTM ``Y_h`` / ``Y_c`` annotations.
    3. Shape inference.
    4. Rewrite ``Slice(Shape(x), k, k+1)`` to ``Gather`` for constant folding.
    5. Fold static ``Gather(Shape(x), [k])`` to constants.
    6. ``onnxscript.optimizer.optimize``.
    7. Shape inference again.
    8. Assert all tensors are fully resolved.
    """
    fix_slice_nodes_for_onnx(model)
    fix_lstm_output_shapes_for_onnx(model)
    _ = onnx_ir.passes.common.shape_inference.infer_shapes(model)
    normalize_shape_slices_to_gather(model)
    fold_static_shape_gather_constants(model)
    _ = onnxscript.optimizer.optimize(model)
    _ = onnx_ir.passes.common.shape_inference.infer_shapes(model)
    assert_shapes_fully_resolved(model, allowed_dynamic_dims)
    if state_metadata:
        validate_exported_onnx_state_shapes(model, state_metadata)


def patch_lstm_layers_for_onnx_export(
    module: th.nn.Module,
) -> list[th.utils.hooks.RemovableHandle]:
    """
    Attach forward hooks so ONNX-exported LSTM hidden states match PyTorch.

    Returns hook handles that must be removed after export (even on failure).
    """
    handles: list[th.utils.hooks.RemovableHandle] = []
    for sub_module in module.modules():
        if isinstance(sub_module, th.nn.LSTM):
            handles.append(sub_module.register_forward_hook(reshape_lstm_output_hook))
    return handles


def reshape_lstm_output_hook(
    lstm: th.nn.LSTM,
    args: tuple[th.Tensor, tuple[th.Tensor, th.Tensor]],
    output: tuple[th.Tensor, tuple[th.Tensor, th.Tensor]],
) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
    """
    Reshape LSTM hidden states during ONNX export so ``hn`` / ``cn`` match PyTorch.

    Only for use with :func:`patch_lstm_layers_for_onnx_export`. The batch axis is
    read from the traced input (axis 0 when ``batch_first=True``, otherwise axis 1)
    rather than using ``-1``. This is not what guarantees fully-resolved shapes --
    ``postprocess_exported_onnx`` does that via its fold pipeline for either reshape
    form. Tying the batch axis to the input is kept because it (a) is correct for
    ``batch_first=False`` LSTMs and (b) yields a cleaner graph that the optimizer
    folds more aggressively (fewer residual Reshape nodes).
    """
    x_in, _ = args
    x_out, (hn, cn) = output

    bidirectional_modifier = 2 if lstm.bidirectional else 1
    layer_dim = bidirectional_modifier * lstm.num_layers

    batch_axis = 0 if lstm.batch_first else 1
    batch_size = x_in.shape[batch_axis]

    hn = hn.reshape(layer_dim, batch_size, lstm.hidden_size)
    cn = cn.reshape(layer_dim, batch_size, lstm.hidden_size)

    return x_out, (hn, cn)


def _promote_initializer_scalar(node_input: ir.Value) -> None:
    """Promote a 0-D initializer Slice input to a length-1 vector in place."""
    node_input.shape = ir.Shape((1,))
    if node_input.const_value is not None:
        node_input.const_value = ir.tensor(
            _scalar_int64(node_input.const_value.numpy()),
            name=node_input.const_value.name,
        )


def _promote_constant_scalar(producer: ir.Node, node_input: ir.Value) -> bool:
    """Promote a 0-D ``Constant``-produced Slice input to a length-1 vector in place."""
    attr = producer.attributes.get("value")
    if attr is None:
        return False
    tensor = attr.as_tensor()
    if tensor.shape is not None and tensor.shape.rank() != 0:
        return False
    promoted = ir.tensor(_scalar_int64(tensor.numpy()), name=tensor.name)
    producer.attributes["value"] = ir.AttrTensor("value", promoted)
    node_input.shape = ir.Shape((1,))
    if node_input.const_value is not None:
        node_input.const_value = promoted
    return True


def fix_slice_nodes_for_onnx(model: ir.Model) -> None:
    """
    Fix Slice nodes that use invalid 0-D tensor inputs.

    ONNX requires Slice ``starts``/``ends``/``axes``/``steps`` to be 1-D. The dynamo
    exporter can emit these as 0-D (scalar) values, either as graph initializers or as
    the output of a ``Constant`` node; both forms are promoted to length-1 vectors here.
    """
    fixed_values: set[str] = set()
    for node in model.graph.all_nodes():
        if node.op_type != "Slice":
            continue
        for node_input in node.inputs:
            if node_input is None:
                continue
            if node_input.shape is not None and node_input.shape.rank() != 0:
                continue
            fixed = False
            if node_input.is_initializer():
                _promote_initializer_scalar(node_input)
                fixed = True
            else:
                producer = node_input.producer()
                if producer is not None and producer.op_type == "Constant":
                    fixed = _promote_constant_scalar(producer, node_input)
            if fixed and node_input.name is not None:
                fixed_values.add(node_input.name)

    if fixed_values:
        logger.info(
            "Fixed %s scalar slice input(s): %s", len(fixed_values), fixed_values
        )


def _static_int_values(value: ir.Value | None) -> npt.NDArray[np.int64] | None:
    """Return flattened integer values from an initializer or Constant output."""
    if value is None:
        return None
    if value.const_value is not None:
        return _flatten_int64(value.const_value.numpy())
    producer = value.producer()
    if producer is None or producer.op_type != "Constant":
        return None
    constant_attr = producer.attributes.get("value")
    if constant_attr is None:
        return None
    constant_tensor = constant_attr.as_tensor()
    return _flatten_int64(constant_tensor.numpy())


def _slice_step_is_one(node: ir.Node) -> bool:
    """Return whether a Slice node uses step 1 (or leaves step unspecified)."""
    if len(node.inputs) <= 4:
        return True
    steps = node.inputs[4]
    if steps is None:
        return True
    step_values = _static_int_values(steps)
    if step_values is None:
        return False
    return len(step_values) == 1 and step_values[0] == 1


def normalize_shape_slices_to_gather(model: ir.Model) -> None:
    """
    Rewrite ``Slice(Shape(x), k, k + 1)`` into ``Gather(Shape(x), [k])``.

    onnxscript's symbolic constant folder handles ``Gather`` on ``Shape`` outputs but
    not ``Slice``, so this rewrite lets a later ``optimizer.optimize`` fold shape
    arithmetic such as ``1 * 64`` and resolve auto-generated ``unk__*`` dims.
    """
    rewritten = 0
    for node in list(model.graph.all_nodes()):
        if node.op_type != "Slice":
            continue
        data_input = node.inputs[0]
        producer = data_input.producer() if data_input is not None else None
        if producer is None or producer.op_type != "Shape":
            continue
        if len(node.inputs) < 3:
            continue
        starts_input, ends_input = node.inputs[1], node.inputs[2]
        if starts_input is None or ends_input is None:
            continue
        start_values = _static_int_values(starts_input)
        end_values = _static_int_values(ends_input)
        if start_values is None or end_values is None:
            continue
        if len(start_values) != 1 or len(end_values) != 1:
            continue
        if end_values[0] != start_values[0] + 1:
            continue
        if not _slice_step_is_one(node):
            continue

        slice_output = node.outputs[0]
        if slice_output.name is None:
            continue
        indices_name = f"{slice_output.name}_gather_idx"
        indices_value = ir.val(
            indices_name,
            const_value=ir.tensor(
                start_values[:1],
                name=indices_name,
            ),
        )
        model.graph.register_initializer(indices_value)
        node.op_type = "Gather"
        node.resize_inputs(2)
        node.replace_input_with(1, indices_value)
        rewritten += 1

    if rewritten:
        logger.info(
            "Rewrote %s Shape slice(s) to Gather for ONNX optimization",
            rewritten,
        )


def fold_static_shape_gather_constants(model: ir.Model) -> None:
    """
    Replace ``Gather(Shape(x), [k])`` with scalar constants when ``x``'s
    ``k``-th dimension is a known integer.
    """
    folded = 0
    for node in list(model.graph.all_nodes()):
        if node.op_type != "Gather" or len(node.inputs) < 2:
            continue
        shape_value, indices_value = node.inputs[0], node.inputs[1]
        shape_node = shape_value.producer() if shape_value is not None else None
        if shape_node is None or shape_node.op_type != "Shape":
            continue
        shaped_input = shape_node.inputs[0]
        if shaped_input is None or shaped_input.shape is None:
            continue
        indices = _static_int_values(indices_value)
        if indices is None or len(indices) != 1:
            continue
        dim_index = indices.item()
        if dim_index < 0 or dim_index >= shaped_input.shape.rank():
            continue
        dim_repr = dim_to_repr(shaped_input.shape.dims[dim_index])
        if is_dynamic_dim_repr(dim_repr):
            continue

        gather_output = node.outputs[0]
        if gather_output.name is None:
            continue
        constant_name = f"{gather_output.name}_folded"
        constant_value = ir.val(
            constant_name,
            const_value=ir.tensor(
                np.array([int(dim_repr)], dtype=np.int64),
                name=constant_name,
            ),
        )
        model.graph.register_initializer(constant_value)
        gather_output.replace_all_uses_with(constant_value)
        folded += 1

    if folded:
        logger.info("Folded %s static Shape gather(s) to constants", folded)


def _producer_desc(value: ir.Value) -> str | None:
    """Describe the node that produced ``value`` for diagnostic error messages."""
    producer = value.producer()
    if producer is None:
        return None
    node_name = producer.name or "<unnamed>"
    return f"op_type={producer.op_type} name={node_name}"


def assert_shapes_fully_resolved(
    model: ir.Model, allowed_dynamic_dims: set[str]
) -> None:
    """Raise if any graph tensor shape still contains unknown or disallowed symbolic dims."""
    checked: set[str] = set()

    def check_value(value: ir.Value) -> None:
        value_name = value.name
        if value_name is None or value_name in checked:
            return
        checked.add(value_name)
        producer_desc = _producer_desc(value)
        if value.shape is None:
            context = f" produced by {producer_desc}" if producer_desc else ""
            raise ValueError(
                f"Exported ONNX tensor '{value_name}'{context} has no resolved shape"
            )
        check_resolved_dims(
            ir_tensor_dims(value.shape),
            value_name,
            allowed_dynamic_dims,
            producer_desc,
        )

    for graph_input in model.graph.inputs:
        check_value(graph_input)
    for graph_output in model.graph.outputs:
        check_value(graph_output)
    for node in model.graph.all_nodes():
        for value in node.outputs:
            check_value(value)


def fix_lstm_output_shapes_for_onnx(model: ir.Model) -> None:
    """
    Drop conflicting rank-4 annotations on ONNX ``LSTM`` hidden/cell outputs.

    Rank-3 annotations are already spec-correct and are left unchanged.
    """
    cleared = 0
    for node in model.graph.all_nodes():
        if node.op_type != "LSTM":
            continue
        if len(node.outputs) != 3:
            raise ValueError(
                f"Expected LSTM node {node.name!r} to have exactly 3 outputs, got {len(node.outputs)}"
            )
        _y_output, y_h_output, y_c_output = node.outputs
        for state_name, state_output in (("Y_h", y_h_output), ("Y_c", y_c_output)):
            if state_output.shape is None:
                continue
            rank = state_output.shape.rank()
            if rank == 3:
                continue
            if rank != 4:
                raise ValueError(
                    f"LSTM node {node.name!r} has unexpected {state_name} rank {rank}; "
                    + "expected rank 3 (spec-correct) or rank 4 (mis-ranked annotation to clear)"
                )
            state_output.shape = None
            cleared += 1

    if cleared:
        logger.info("Cleared %s mis-ranked LSTM state output annotation(s)", cleared)
