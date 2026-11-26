# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# pyright: reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportAny=false

"""ONNX export functions and helpers (Slice node fixes) for ``ScholaModel``."""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import onnx_ir as ir
import torch as th

from schola.core.model import ScholaModel
from schola.core.export.export_common import (
    build_export_example,
    lstm_export_patch,
)

logger = logging.getLogger(__name__)


def export_onnx_program(model: ScholaModel, onnx_opset: int = 21) -> th.onnx.ONNXProgram:
    """
    Export a ``ScholaModel`` as an ONNX program.

    The model has the following properties:
    - Inputs are named based on they key in the observation space.
    - Outputs are named based on they key in the action space.
    - State inputs have metadata that contains the sequence dimension and max sequence length if they have a sequence dimension.
    - Those same fields are copied onto the model metadata as ``{state_input_name}.{field}`` (for example ``state_in_actor_h.has_seq_dim``).

    Parameters
    ----------
    model : ScholaModel
        Model to export.
    onnx_opset : int, optional
        The ONNX opset version to use for the export.

    Returns
    -------
    th.onnx.ONNXProgram
        The ONNX program generated with torch dynamo export.
    """
    example = build_export_example(model, batch_size=2, fixed_batch=False)

    # State inputs have shape: [batch_size, *state_dim] (state_dim may include a sequence dimension)
    # Observation inputs have shape: [batch_size, *obs_dim]
    with th.no_grad(), lstm_export_patch(model):
        onnx_program = th.onnx.export(
            model,
            args=example.args,
            input_names=example.input_names,
            opset_version=onnx_opset,
            output_names=example.output_names,
            dynamic_shapes=(example.dynamic_shapes,),
            dynamo=True,
            report=False,
            optimize=True,
            verbose=False,
        )

        assert (
            onnx_program is not None
        ), "Expected ONNX program to be generated after calling th.onnx.export"
        fix_slice_nodes_for_onnx(onnx_program.model)

        ir_passes: Any = ir.passes
        ir_passes.common.shape_inference.infer_shapes(onnx_program.model)
        # Embed state metadata on each state input, and copy those fields
        # onto the model metadata as "{input_name}.{field}".
        for inp in onnx_program.model.graph.inputs:
            if inp.name in model.input_state_metadata:
                props = model.input_state_metadata[inp.name].to_dict()
                inp.metadata_props.update(props)
                for key, value in props.items():
                    onnx_program.model.metadata_props[f"{inp.name}.{key}"] = value
        return onnx_program


def save_as_onnx(
    model: ScholaModel, export_path: str | pathlib.Path, onnx_opset: int = 21
) -> None:
    """
    Export a ``ScholaModel`` to an ``.onnx`` file on disk.

    Parameters
    ----------
    model : ScholaModel
        Model to export.
    export_path : str or pathlib.Path
        Output file path; parent directories are created if missing.
    onnx_opset : int, optional
        ONNX opset passed to :func:`export_onnx_program`.

    Returns
    -------
    None
    """
    dir_path = pathlib.Path(export_path).parent
    dir_path.mkdir(parents=True, exist_ok=True)
    onnx_program = export_onnx_program(model, onnx_opset)
    onnx_program.save(export_path)


def fix_slice_nodes_for_onnx(model: ir.Model) -> None:
    """
    Fix Slice nodes produced by ``torch.onnx.export`` that use invalid 0-D tensor inputs.

    Parameters
    ----------
    model : onnx_ir.ir.Model
        ONNX IR model to mutate in place.

    """
    # Create a mapping of initializer names to their values
    fixed_values: set[str] = set()
    # Find all Slice nodes
    for node in model.graph.all_nodes():
        if node.op_type == "Slice":
            for _i, node_input in enumerate(node.inputs):
                if node_input is None:
                    continue

                if node_input.is_initializer() and (
                    node_input.shape is None or node_input.shape.rank() == 0
                ):
                    node_input.shape = ir.Shape((1,))
                    if node_input.const_value is not None:
                        node_input.const_value = ir.tensor(
                            node_input.const_value.numpy().reshape((1,)),
                            name=node_input.const_value.name,
                        )
                    if node_input.name is not None:
                        fixed_values.add(node_input.name)

    logger.warning("Fixed %s slice nodes: %s", len(fixed_values), fixed_values)
