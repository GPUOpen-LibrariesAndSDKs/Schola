# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# pyright: reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportAny=false

"""Shared helpers for ONNX and IREE export of ``ScholaModel`` instances."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch as th
from torch.export import Dim

from schola.core.model import ScholaModel


@dataclass
class ExportExample:
    """Example tensors and names used by both ONNX and IREE exporters."""

    args: tuple[th.Tensor, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    dynamic_shapes: tuple[dict[int, Dim], ...]
    input_dtypes: list[str]
    output_dtypes: list[str]
    input_shapes: list[list[int]]
    output_shapes: list[list[int]]
    batch_size: int


def reshape_lstm_output_hook(
    lstm: th.nn.LSTM,
    args: tuple[th.Tensor, tuple[th.Tensor, th.Tensor]],
    output: tuple[th.Tensor, tuple[th.Tensor, th.Tensor]],
) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
    """
    Reshape LSTM hidden states during ONNX export so ``hn`` / ``cn`` match PyTorch layouts.

    Only for use with :func:`patch_lstm_layers_for_export`; requires batched LSTM inputs.

    Parameters
    ----------
    lstm : torch.nn.LSTM
        Layer instance receiving the hook.
    args : tuple
        Forward inputs ``(input, (h_0, c_0))`` as passed to ``LSTM.forward``.
    output : tuple
        Forward outputs ``(output, (h_n, c_n))``.

    Returns
    -------
    tuple
        ``(output, (h_n, c_n))`` with ``h_n`` and ``c_n`` reshaped for ONNX compatibility.

    Notes
    -----
    This hook is only used during export (ONNX / IREE), and is not used for normal forward passes.
    """
    _x_in, _ = args
    x_out, (hn, cn) = output

    bidirectional_modifier = 2 if lstm.bidirectional else 1
    layer_dim = bidirectional_modifier * lstm.num_layers

    # This is a no-op on pytorch, but on ONNX the output has an extra dimension that we need to squeeze to match the pytorch output
    hn = hn.reshape(layer_dim, -1, lstm.hidden_size)
    cn = cn.reshape(layer_dim, -1, lstm.hidden_size)

    return x_out, (hn, cn)


def patch_lstm_layers_for_export(
    module: th.nn.Module,
) -> list[th.utils.hooks.RemovableHandle]:
    """
    Attach forward hooks so exported (ONNX / IREE) LSTM hidden states match PyTorch.

    Parameters
    ----------
    module : torch.nn.Module
        Root module; every nested ``torch.nn.LSTM`` receives
        ``reshape_lstm_output_hook``.

    Returns
    -------
    list of torch.utils.hooks.RemovableHandle
        Forward-hook handles registered on each nested LSTM (pass to
        ``RemovableHandle.remove()`` after export).

    Notes
    -----
    Prefer :func:`lstm_export_patch`, which removes the hooks even if export fails.
    """
    handles = []
    for sub_module in module.modules():
        if isinstance(sub_module, th.nn.LSTM):
            handles.append(sub_module.register_forward_hook(reshape_lstm_output_hook))
    return handles


@contextmanager
def lstm_export_patch(module: th.nn.Module) -> Iterator[None]:
    """
    Temporarily patch nested LSTMs for export, removing the hooks on exit.

    The hooks are removed even if the body raises, so a failed export never
    leaves hooks attached to the caller's model.

    Parameters
    ----------
    module : torch.nn.Module
        Root module whose nested ``torch.nn.LSTM`` layers are patched.
    """
    handles = patch_lstm_layers_for_export(module)
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _torch_dtype_name(dtype: th.dtype) -> str:
    mapping = {
        th.float32: "float32",
        th.float64: "float64",
        th.float16: "float16",
        th.int64: "int64",
        th.int32: "int32",
        th.int8: "int8",
        th.bool: "bool",
    }
    return mapping.get(dtype, str(dtype).replace("torch.", ""))


def build_export_example(
    model: ScholaModel, *, batch_size: int = 2, fixed_batch: bool = False
) -> ExportExample:
    """
    Build example inputs, names, and dynamic-shape specs for export.

    Parameters
    ----------
    model : ScholaModel
        Model to export.
    batch_size : int, optional
        Example batch size. Use a non-singleton batch for ONNX so PyTorch does
        not specialize the batch dim to 1 before applying dynamic shapes.
    fixed_batch : bool, optional
        When True, omit dynamic batch/seq dims (used for IREE llvm-cpu).
    """
    _ = model.eval()
    obs_inputs: list[th.Tensor] = []
    for obs_space in model.observation_space.spaces.values():
        obs_inputs.append(
            th.as_tensor(np.stack([obs_space.sample() for _ in range(batch_size)]))
        )

    batch_dim = Dim("batch_size")
    seq_dim = Dim("seq_len")

    if fixed_batch:
        obs_input_shapes: tuple[dict[int, Dim], ...] = tuple({} for _ in obs_inputs)
    else:
        obs_input_shapes = tuple({0: batch_dim} for _ in obs_inputs)

    state_input_shapes_generator: tuple[dict[int, Dim], ...] = ()
    state_input_generator: tuple[th.Tensor, ...] = ()

    if model.is_stateful:
        state_input_generator = tuple(
            v.reshape(1, *v.shape).repeat(batch_size, *[1 for _ in v.shape])
            for v in model.input_state_dict.values()
        )
        if fixed_batch:
            state_input_shapes_generator = tuple({} for _ in model.input_state_metadata)
        else:
            state_dynamic_shapes_fn = lambda _k, v: (
                {0: batch_dim, v.seq_dim: seq_dim} if v.has_seq_dim else {0: batch_dim}
            )
            state_input_shapes_generator = tuple(
                state_dynamic_shapes_fn(k, metadata)
                for k, metadata in model.input_state_metadata.items()
            )

    input_args = (*obs_inputs, *state_input_generator)
    input_names = (*model.input_obs_keys, *model.input_state_keys)
    input_shapes = (*obs_input_shapes, *state_input_shapes_generator)
    output_names = (*model.output_action_keys, *model.output_state_keys)

    if set(input_names).intersection(set(output_names)):
        raise ValueError(
            "Input and output names must be unique. Reused Names: "
            f"{set(input_names).intersection(set(output_names))}"
        )

    with th.no_grad():
        outputs = model(*input_args)
    outputs = tuple(outputs) if isinstance(outputs, (tuple, list)) else (outputs,)

    return ExportExample(
        args=input_args,
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=input_shapes,
        input_dtypes=[_torch_dtype_name(t.dtype) for t in input_args],
        output_dtypes=[_torch_dtype_name(t.dtype) for t in outputs],
        input_shapes=[list(t.shape) for t in input_args],
        output_shapes=[list(t.shape) for t in outputs],
        batch_size=batch_size,
    )
