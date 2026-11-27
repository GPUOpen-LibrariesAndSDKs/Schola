# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# pyright: reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportAny=false

"""IREE (iree-turbine) export helpers for ``ScholaModel`` and loading exported modules."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, replace
import pathlib
from typing import Any

import torch as th

from schola.core.model import ScholaModel, StateMetadata
from schola.core.export.export_common import (
    ExportExample,
    build_export_example,
    lstm_export_patch,
)

logger = logging.getLogger(__name__)

#: IREE target backends used when none are specified.
DEFAULT_TARGET_BACKENDS: tuple[str, ...] = ("llvm-cpu",)

#: Key inside a function's ``iree.reflection`` attribute that holds the Schola metadata JSON.
SCHOLA_REFLECTION_KEY = "schola.metadata"


#: Prefixes that turn a state's ``name`` into its model input / output tensor name.
STATE_IN_PREFIX = "state_in_"
STATE_OUT_PREFIX = "state_out_"


@dataclass
class TensorSpec:
    """One observation input or action output tensor."""

    name: str
    dtype: str
    shape: list[int]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TensorSpec":
        return cls(name=data["name"], dtype=data["dtype"], shape=list(data["shape"]))


@dataclass
class StateSpec:
    """
    A recurrent state, passed in as ``state_in_<name>`` and returned as ``state_out_<name>``.

    The input and output tensors share the same dtype and shape.
    """

    name: str
    dtype: str
    shape: list[int]
    state_metadata: StateMetadata

    @property
    def in_name(self) -> str:
        return STATE_IN_PREFIX + self.name

    @property
    def out_name(self) -> str:
        return STATE_OUT_PREFIX + self.name

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateSpec":
        return cls(
            name=data["name"],
            dtype=data["dtype"],
            shape=list(data["shape"]),
            state_metadata=StateMetadata(**data["state_metadata"]),
        )


@dataclass
class ScholaIREEMetadata:
    """Model metadata embedded in an IREE ``.vmfb`` as an ``iree.reflection`` attribute."""

    entry_function: str
    #: Observation inputs, in call order (before the state inputs).
    inputs: list[TensorSpec]
    #: Action outputs, in return order (before the state outputs).
    outputs: list[TensorSpec]
    #: Recurrent states: one input (after the observations) and one output (after the actions) each.
    state: list[StateSpec]
    target_backends: list[str]
    iree_version: str | None
    batch_size: int

    @property
    def call_inputs(self) -> list[TensorSpec]:
        """All entry-function inputs in call order: observations, then states."""
        return [
            *self.inputs,
            *(TensorSpec(s.in_name, s.dtype, s.shape) for s in self.state),
        ]

    @property
    def call_outputs(self) -> list[TensorSpec]:
        """All entry-function outputs in return order: actions, then states."""
        return [
            *self.outputs,
            *(TensorSpec(s.out_name, s.dtype, s.shape) for s in self.state),
        ]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to the compact JSON string stored in the ``iree.reflection`` attribute."""
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "ScholaIREEMetadata":
        data = json.loads(text)
        return cls(
            entry_function=data["entry_function"],
            inputs=[TensorSpec.from_dict(t) for t in data["inputs"]],
            outputs=[TensorSpec.from_dict(t) for t in data["outputs"]],
            state=[StateSpec.from_dict(s) for s in data["state"]],
            target_backends=list(data["target_backends"]),
            iree_version=data.get("iree_version"),
            batch_size=data["batch_size"],
        )


def _normalized_state_metadata(metadata: StateMetadata) -> StateMetadata:
    """Copy ``metadata``, dropping sequence fields that are meaningless without ``has_seq_dim``."""
    if metadata.has_seq_dim:
        return replace(metadata)
    return StateMetadata(has_seq_dim=False, max_seq_len=None, seq_dim=None)


def build_iree_metadata(
    model: ScholaModel,
    example: ExportExample,
    *,
    entry_function: str = "main",
    target_backends: list[str] | None = None,
    iree_version: str | None = None,
) -> ScholaIREEMetadata:
    """
    Build the embedded model metadata.

    Names and state metadata come from ``model``; dtypes, shapes and batch size
    come from the traced ``example`` (they depend on the example inputs).
    """
    n_obs = len(model.input_obs_keys)
    n_actions = len(model.output_action_keys)

    inputs = [
        TensorSpec(name, dtype, list(shape))
        for name, dtype, shape in zip(
            model.input_obs_keys, example.input_dtypes, example.input_shapes
        )
    ]
    outputs = [
        TensorSpec(name, dtype, list(shape))
        for name, dtype, shape in zip(
            model.output_action_keys, example.output_dtypes, example.output_shapes
        )
    ]

    state: list[StateSpec] = []
    for i, (in_key, out_key) in enumerate(
        zip(model.input_state_keys, model.output_state_keys)
    ):
        if not in_key.startswith(STATE_IN_PREFIX) or not out_key.startswith(
            STATE_OUT_PREFIX
        ):
            raise ValueError(
                f"State tensors must be named '{STATE_IN_PREFIX}<name>' / "
                f"'{STATE_OUT_PREFIX}<name>'; got '{in_key}' / '{out_key}'."
            )
        name = in_key[len(STATE_IN_PREFIX) :]
        if out_key[len(STATE_OUT_PREFIX) :] != name:
            raise ValueError(
                f"State input '{in_key}' does not match state output '{out_key}'."
            )
        dtype = example.input_dtypes[n_obs + i]
        shape = list(example.input_shapes[n_obs + i])
        if (
            example.output_dtypes[n_actions + i] != dtype
            or list(example.output_shapes[n_actions + i]) != shape
        ):
            raise ValueError(
                f"State '{name}' must have the same dtype and shape as input and output."
            )
        state.append(
            StateSpec(
                name=name,
                dtype=dtype,
                shape=shape,
                state_metadata=_normalized_state_metadata(
                    model.input_state_metadata[in_key]
                ),
            )
        )

    return ScholaIREEMetadata(
        entry_function=entry_function,
        inputs=inputs,
        outputs=outputs,
        state=state,
        target_backends=list(target_backends or DEFAULT_TARGET_BACKENDS),
        iree_version=iree_version,
        batch_size=example.batch_size,
    )


def iree_version() -> str | None:
    try:
        import iree.compiler as ireec  # type: ignore[import-not-found]

        return getattr(ireec, "__version__", None)
    except Exception:
        try:
            import iree.runtime as ireert  # type: ignore[import-not-found]

            return getattr(ireert, "__version__", None)
        except Exception:
            return None


def embed_metadata(
    export_output,
    entry_function: str,
    metadata: ScholaIREEMetadata,
) -> None:
    """Attach ``metadata`` to ``entry_function`` as ``iree.reflection["schola.metadata"]``."""
    from iree.compiler import ir  # type: ignore[import-not-found]

    module = export_output.mlir_module
    # ``mlir_module`` may be an ``ir.Module`` or its underlying ``ir.Operation``.
    module_op = getattr(module, "operation", module)
    with module_op.context:
        for op in module_op.regions[0].blocks[0].operations:
            attrs = op.operation.attributes
            if op.operation.name not in ("func.func", "util.func"):
                continue
            if ir.StringAttr(attrs["sym_name"]).value != entry_function:
                continue

            # Preserve any reflection entries the frontend already attached.
            entries: dict[str, ir.Attribute] = {}
            if "iree.reflection" in attrs:
                existing = ir.DictAttr(attrs["iree.reflection"])
                entries.update({named.name: named.attr for named in existing})
            entries[SCHOLA_REFLECTION_KEY] = ir.StringAttr.get(metadata.to_json())
            attrs["iree.reflection"] = ir.DictAttr.get(entries)
            return
    raise RuntimeError(
        f"Could not find exported function '{entry_function}' to attach metadata to."
    )


def read_embedded_metadata(vm_module) -> ScholaIREEMetadata:
    """Read the Schola metadata embedded in a loaded IREE ``VmModule``."""
    for name in vm_module.function_names:
        reflection = vm_module.lookup_function(name).reflection
        if SCHOLA_REFLECTION_KEY in reflection:
            return ScholaIREEMetadata.from_json(reflection[SCHOLA_REFLECTION_KEY])
    raise ValueError(
        f"No '{SCHOLA_REFLECTION_KEY}' iree.reflection attribute found in module."
    )


def save_as_iree(
    model: ScholaModel,
    export_path: str | pathlib.Path,
    *,
    target_backends: list[str] | None = None,
    entry_function: str = "main",
    batch_size: int = 1,
) -> None:
    """
    Export a ``ScholaModel`` to an IREE ``.vmfb`` file with embedded Schola metadata.

    The metadata JSON is stored in the entry function's ``iree.reflection``
    attribute under the key ``schola.metadata``. Uses ``iree.turbine.aot``
    with a fixed batch size (default 1) so the compiled artifact matches
    Unreal's single-agent inference path.

    Parameters
    ----------
    model : ScholaModel
        Model to export.
    export_path : str or pathlib.Path
        Output ``.vmfb`` path; parent directories are created if missing.
    target_backends : list of str, optional
        IREE target backend(s) passed to ``ExportOutput.compile``.
        Defaults to ``["llvm-cpu"]``.
    entry_function : str, optional
        Exported entry function name (default ``main``).
    batch_size : int, optional
        Fixed example batch size used for tracing (default 1).
    """
    try:
        import iree.turbine.aot as aot  # type: ignore[import-not-found]
        from iree.turbine.aot.decompositions import (  # type: ignore[import-not-found]
            extend_aot_decompositions,
        )
    except ImportError as exc:
        raise ImportError(
            "IREE export requires the optional 'iree' extra. "
            "Install with: pip install 'schola[iree]'"
        ) from exc

    export_path = pathlib.Path(export_path)
    if export_path.suffix.lower() != ".vmfb":
        requested_path = export_path
        export_path = export_path.with_suffix(".vmfb")
        logger.warning(
            "IREE export path '%s' does not end in '.vmfb'; writing to '%s' instead.",
            requested_path,
            export_path,
        )
    export_path.parent.mkdir(parents=True, exist_ok=True)

    backends = list(target_backends or DEFAULT_TARGET_BACKENDS)

    # Export requires eval mode (build_export_example also switches to it), so
    # remember each submodule's mode and restore it afterwards, even on failure.
    training_modes = {m: m.training for m in model.modules()}
    try:
        example = build_export_example(model, batch_size=batch_size, fixed_batch=True)
        _ = model.eval()
        # IREE has no lowering for aten.lstm, and the mkldnn LSTM decomposition emits
        # mkldnn_rnn_layer, which IREE cannot lower either.
        with (
            th.no_grad(),
            extend_aot_decompositions(add_ops=[th.ops.aten.lstm.input]),
            th.backends.mkldnn.flags(enabled=False),
            lstm_export_patch(model),
        ):
            # Non-strict tracing, like torch.onnx.export. Strict (Dynamo) tracing
            # rejects SB3's one_hot preprocessing and ScholaModel's vmap outputs.
            export_output = aot.export(
                model,
                args=example.args,
                function_name=entry_function,
                # Fixed shapes for llvm-cpu Unreal inference (batch=1).
                dynamic_shapes=None,
                strict_export=False,
            )
            metadata = build_iree_metadata(
                model,
                example,
                entry_function=entry_function,
                target_backends=backends,
                iree_version=iree_version(),
            )
            embed_metadata(export_output, entry_function, metadata)
            export_output.compile(save_to=str(export_path), target_backends=backends)
    finally:
        for module, was_training in training_modes.items():
            module.training = was_training

    logger.info("Wrote IREE model with embedded metadata to %s", export_path)


def load_iree_module(
    vmfb_path: str | pathlib.Path,
    *,
    device: str = "local-task",
):
    """
    Load a compiled ``.vmfb`` with the IREE Python runtime.

    Parameters
    ----------
    vmfb_path : str or pathlib.Path
        Path to the ``.vmfb`` file.
    device : str, optional
        IREE device URI (default ``local-task``).

    Returns
    -------
    tuple
        ``(vm_module, metadata)`` where ``vm_module`` exposes the entry
        function and ``metadata`` is a :class:`ScholaIREEMetadata`.
    """
    try:
        import iree.runtime as ireert  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Loading IREE modules requires the optional 'iree' extra. "
            "Install with: pip install 'schola[iree]'"
        ) from exc

    vmfb_path = pathlib.Path(vmfb_path)
    config = ireert.Config(device)
    flatbuffer = vmfb_path.read_bytes()
    raw_module = ireert.VmModule.copy_buffer(config.vm_instance, flatbuffer)
    metadata = read_embedded_metadata(raw_module)
    vm_module = ireert.load_vm_module(raw_module, config)
    return vm_module, metadata
