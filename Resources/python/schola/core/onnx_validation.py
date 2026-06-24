# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Shared ONNX dimension helpers and state-shape validation for export and tests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Protocol

import onnx_ir as ir
from onnx.onnx_ml_pb2 import ValueInfoProto


class StateMetadataLike(Protocol):
    """Minimal state-metadata surface used by ONNX validation."""

    has_seq_dim: bool
    seq_dim: int | None


def dim_to_repr(dim: int | ir.SymbolicDim) -> str | int:
    """Normalize an ONNX / IR dimension to a string symbolic name or integer."""
    if isinstance(dim, int):
        return dim
    return str(dim)


def is_dynamic_dim_repr(dim_repr: str | int) -> bool:
    """Return whether a dimension is unresolved (symbolic or -1)."""
    return isinstance(dim_repr, str) or dim_repr < 0


def tensor_dims_from_value_info(tensor: ValueInfoProto) -> list[str | int]:
    """Read symbolic or integer dimensions from an ONNX ``ValueInfoProto``."""
    dims: list[str | int] = []
    for dim in tensor.type.tensor_type.shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)
        elif dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        else:
            dims.append(-1)
    return dims


def ir_tensor_dims(shape: ir.Shape | None) -> list[str | int]:
    """Read symbolic or integer dimensions from an ONNX IR shape."""
    if shape is None:
        return []
    return [dim_to_repr(dim) for dim in shape.dims]


def state_out_name_for_input(input_name: str) -> str:
    """Map a flattened ONNX state input name to its matching output name."""
    return input_name.replace("state_in_", "state_out_", 1)


def emulate_nne_seq_dim(
    shape: Iterable[str | int], *, fix_batch_to_1: bool = True
) -> int:
    """
    Emulate ``FNNEStateBuffer`` sequence-axis inference via ``FindLast(-1)``.

    Unreal fixes ``Shape[0] = 1`` then searches backward for the last
    unresolved (-1 / symbolic) dimension.
    """
    resolved = list(shape)
    if fix_batch_to_1 and resolved:
        resolved[0] = 1
    for index in range(len(resolved) - 1, -1, -1):
        if is_dynamic_dim_repr(resolved[index]):
            return index
    return -1


def allowed_dynamic_dims_for_state_metadata(
    input_state_metadata: Mapping[str, StateMetadataLike],
) -> set[str]:
    """Return symbolic dimension names permitted for the given state metadata."""
    allowed = {"batch_size"}
    if any(metadata.has_seq_dim for metadata in input_state_metadata.values()):
        allowed.add("seq_len")
    return allowed


def check_resolved_dims(
    dims: list[str | int],
    tensor_name: str,
    allowed_dynamic_dims: set[str],
) -> None:
    """
    Raise if any dimension is unresolved outside ``allowed_dynamic_dims``.

    Parameters
    ----------
    dims : list of str or int
        Tensor shape dimensions from proto or IR.
    tensor_name : str
        Name used in error messages.
    allowed_dynamic_dims : set of str
        Symbolic names that may remain dynamic.
    """
    for dim_index, dim_repr in enumerate(dims):
        if not is_dynamic_dim_repr(dim_repr):
            continue
        if isinstance(dim_repr, str) and dim_repr.startswith("unk"):
            raise ValueError(
                f"ONNX tensor '{tensor_name}' dimension {dim_index} uses "
                + f"auto-generated symbolic name {dim_repr!r}"
            )
        if dim_repr not in allowed_dynamic_dims:
            raise ValueError(
                f"ONNX tensor '{tensor_name}' dimension {dim_index} is unresolved "
                + f"({dim_repr!r}); allowed dynamic dims are "
                + f"{sorted(allowed_dynamic_dims)}"
            )


def validate_onnx_state_shape_rules(
    input_shapes: dict[str, list[str | int]],
    output_shapes: dict[str, list[str | int]],
    input_state_metadata: Mapping[str, StateMetadataLike],
) -> None:
    """
    Validate recurrent state tensor shapes against ``StateMetadata``.

    Pure function over dimension lists; shared by IR export validation and
    protobuf-based integration tests.

    Raises
    ------
    ValueError
        If state I/O shapes are inconsistent or would be misread by Unreal NNE.
    """
    if set(input_shapes) != set(input_state_metadata):
        missing = set(input_state_metadata) - set(input_shapes)
        raise ValueError(
            "Exported ONNX model is missing state inputs declared in metadata: "
            + f"{sorted(missing)}"
        )

    for input_name, metadata in input_state_metadata.items():
        in_shape = input_shapes[input_name]
        out_name = state_out_name_for_input(input_name)
        out_shape = output_shapes.get(out_name)
        if out_shape is None:
            raise ValueError(
                f"Exported ONNX model is missing matching state output '{out_name}' "
                + f"for input '{input_name}'"
            )
        if in_shape != out_shape:
            raise ValueError(
                f"State input/output shapes must match for recurrent round-trip. "
                + f"'{input_name}' has {in_shape} but '{out_name}' has {out_shape}"
            )

        inferred_seq_dim = emulate_nne_seq_dim(in_shape)
        if metadata.has_seq_dim:
            if metadata.seq_dim is None:
                raise ValueError(
                    f"State metadata for '{input_name}' sets has_seq_dim=True "
                    + "but omits seq_dim"
                )
            if inferred_seq_dim != metadata.seq_dim:
                raise ValueError(
                    f"State tensor '{input_name}' shape {in_shape} would be read by "
                    + f"Unreal NNE with seq_dim={inferred_seq_dim}, but metadata "
                    + f"declares seq_dim={metadata.seq_dim}"
                )
        elif inferred_seq_dim != -1:
            raise ValueError(
                f"State tensor '{input_name}' shape {in_shape} leaves dynamic axis "
                + f"{inferred_seq_dim} after fixing batch to 1, but has_seq_dim=False. "
                + "Only the batch dimension may be dynamic for LSTM state I/O."
            )


def validate_exported_onnx_state_shapes(
    onnx_model: ir.Model,
    input_state_metadata: Mapping[str, StateMetadataLike],
) -> None:
    """Validate recurrent state tensor shapes on an ONNX IR model."""
    input_shapes: dict[str, list[str | int]] = {}
    for graph_input in onnx_model.graph.inputs:
        input_name = graph_input.name
        if input_name is None or input_name not in input_state_metadata:
            continue
        input_shapes[input_name] = ir_tensor_dims(graph_input.shape)

    output_shapes: dict[str, list[str | int]] = {}
    for graph_output in onnx_model.graph.outputs:
        output_name = graph_output.name
        if output_name is None or not output_name.startswith("state_out_"):
            continue
        output_shapes[output_name] = ir_tensor_dims(graph_output.shape)

    validate_onnx_state_shape_rules(input_shapes, output_shapes, input_state_metadata)
