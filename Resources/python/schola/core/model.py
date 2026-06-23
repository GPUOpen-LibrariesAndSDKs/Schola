# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
ONNX export metadata, ``ScholaModel``, and related helpers for policies trained with Schola.
"""

from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
import onnx
import torch as th

import gymnasium as gym
from gymnasium.spaces import Box, flatdim
import numpy as np
from torch.export import Dim
from functools import cached_property
from schola.core.utils.dict_helpers import *
from itertools import accumulate
import numpy as np
import onnx_ir as ir
import onnx_ir.passes.common.shape_inference
import onnxscript.optimizer

logger = logging.getLogger(__name__)


# bit overkill for now but we can extend later if we need more metadata
@dataclass
class StateMetadata:
    """
    Metadata for recurrent or sequential state tensors in ONNX export.
    """

    has_seq_dim: bool = False  # whether the state input has a sequence dimension
    max_seq_len: Optional[int] = None  # maximum sequence length for the state input
    seq_dim: Optional[int] = None  # index of the sequence dimension

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize state metadata to string values for ONNX ``metadata_props``.

        Returns
        -------
        Dict[str, Any]
            Keys ``has_seq_dim`` and, when sequential, ``max_seq_len`` and ``seq_dim``.
        """
        output_dict = {"has_seq_dim": str(self.has_seq_dim)}
        if self.has_seq_dim:
            output_dict["max_seq_len"] = str(self.max_seq_len)
            output_dict["seq_dim"] = str(self.seq_dim)
        return output_dict


def _dim_to_repr(dim: Any) -> Union[str, int]:
    """Normalize an ONNX / IR dimension to a string symbolic name or integer."""
    if isinstance(dim, int):
        return dim
    return str(dim)


def _is_dynamic_dim_repr(dim_repr: Union[str, int]) -> bool:
    """Return whether a dimension is unresolved (symbolic or -1)."""
    return isinstance(dim_repr, str) or dim_repr < 0


def _proto_tensor_dims(tensor: Any) -> List[Union[str, int]]:
    """Read symbolic or integer dimensions from an ONNX ``ValueInfoProto``."""
    dims: List[Union[str, int]] = []
    for dim in tensor.type.tensor_type.shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)
        elif dim.dim_value:
            dims.append(dim.dim_value)
        else:
            dims.append(-1)
    return dims


def _ir_tensor_dims(shape: Optional[ir.Shape]) -> List[Union[str, int]]:
    """Read symbolic or integer dimensions from an ONNX IR shape."""
    if shape is None:
        return []
    return [_dim_to_repr(dim) for dim in shape.dims]


def emulate_nne_seq_dim(
    shape: Iterable[Union[str, int]], *, fix_batch_to_1: bool = True
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
        if _is_dynamic_dim_repr(resolved[index]):
            return index
    return -1


def validate_exported_onnx_state_shapes(
    onnx_model: ir.Model,
    input_state_metadata: Dict[str, StateMetadata],
) -> None:
    """
    Validate recurrent state tensor shapes against embedded ``StateMetadata``.

    Raises
    ------
    ValueError
        If state I/O shapes are inconsistent or would be misread by Unreal NNE.
    """
    input_shapes: Dict[str, List[Union[str, int]]] = {}
    for graph_input in onnx_model.graph.inputs:
        input_name = graph_input.name
        if input_name is None or input_name not in input_state_metadata:
            continue
        input_shapes[input_name] = _ir_tensor_dims(graph_input.shape)

    output_shapes: Dict[str, List[Union[str, int]]] = {}
    for graph_output in onnx_model.graph.outputs:
        output_name = graph_output.name
        if output_name is None or not output_name.startswith("state_out_"):
            continue
        output_shapes[output_name] = _ir_tensor_dims(graph_output.shape)

    if set(input_shapes) != {name for name in input_state_metadata}:
        missing = set(input_state_metadata) - set(input_shapes)
        raise ValueError(
            "Exported ONNX model is missing state inputs declared in metadata: "
            f"{sorted(missing)}"
        )

    for input_name, metadata in input_state_metadata.items():
        in_shape = input_shapes[input_name]
        out_name = input_name.replace("state_in_", "state_out_", 1)
        out_shape = output_shapes.get(out_name)
        if out_shape is None:
            raise ValueError(
                f"Exported ONNX model is missing matching state output '{out_name}' "
                f"for input '{input_name}'"
            )
        if in_shape != out_shape:
            raise ValueError(
                f"State input/output shapes must match for recurrent round-trip. "
                f"'{input_name}' has {in_shape} but '{out_name}' has {out_shape}"
            )

        inferred_seq_dim = emulate_nne_seq_dim(in_shape)
        if metadata.has_seq_dim:
            if metadata.seq_dim is None:
                raise ValueError(
                    f"State metadata for '{input_name}' sets has_seq_dim=True "
                    "but omits seq_dim"
                )
            if inferred_seq_dim != metadata.seq_dim:
                raise ValueError(
                    f"State tensor '{input_name}' shape {in_shape} would be read by "
                    f"Unreal NNE with seq_dim={inferred_seq_dim}, but metadata "
                    f"declares seq_dim={metadata.seq_dim}"
                )
        elif inferred_seq_dim != -1:
            raise ValueError(
                f"State tensor '{input_name}' shape {in_shape} leaves dynamic axis "
                f"{inferred_seq_dim} after fixing batch to 1, but has_seq_dim=False. "
                "Only the batch dimension may be dynamic for LSTM state I/O."
            )


class StatefulModelMixin:
    """
    Mixin for models that expose non-observation internal state (e.g. RNN hidden state).

    Subclasses should override ``initial_state_dict`` and related cached properties.

    See Also
    --------
    ScholaModel
    """

    @cached_property
    def initial_state_dict(self) -> NestedDict[str, th.Tensor]:
        """
        Nested structure of state tensors **without** batch dimensions.

        Returns
        -------
        NestedDict[str, torch.Tensor]
            Nested structure of state tensors **without** batch dimensions.
            Stateless models use the default empty mapping.
        """
        # implement this in stateful subclasses, tensors should not include any batch dimensions
        return {}

    @cached_property
    def is_stateful(self) -> bool:
        """
        Whether this model uses non-observation internal state (e.g. RNN hidden state).

        Returns
        -------
        bool
            ``True`` if ``initial_state_dict`` is non-empty.
        """
        return len(self.initial_state_dict) > 0

    @cached_property
    def state_metadata(self) -> NestedDict[str, StateMetadata]:
        """
        Metadata aligned with ``initial_state_dict`` keys (default: empty
        :class:`StateMetadata` for each leaf).

        Returns
        -------
        NestedDict[str, StateMetadata]
            Metadata aligned with ``initial_state_dict`` keys (default: empty
            :class:`StateMetadata` for each leaf).
        """
        return (
            DIterator(self.initial_state_dict).map(lambda x: StateMetadata()).to_dict()
        )

    @cached_property
    def input_state_dict(self) -> Dict[str, th.Tensor]:
        """
        Flattened state inputs keyed as ``state_in/...`` for export and ONNX naming.

        Returns
        -------
        Dict[str, torch.Tensor]
            Flattened state inputs keyed as ``state_in/...`` for export and ONNX naming.
        """
        return flatten_dict(self.initial_state_dict, "state_in")

    @cached_property
    def input_state_keys(self) -> List[str]:
        """
        Keys of ``input_state_dict`` for export and ONNX naming.

        Returns
        -------
        list of str
            Keys of ``input_state_dict``.
        """
        return list(self.input_state_dict.keys())

    @cached_property
    def output_state_keys(self) -> List[str]:
        """
        Keys of ``output_state_dict`` for export and ONNX naming.

        Returns
        -------
        list of str
            Flattened output state names (``state_out/...``) derived from ``initial_state_dict``.
        """
        return list(flattened_key_iterator(self.initial_state_dict, "state_out"))

    @cached_property
    def input_state_metadata(self) -> Dict[str, StateMetadata]:
        """
        Flattened metadata for each ONNX state input name.

        Returns
        -------
        Dict[str, StateMetadata]
            Flattened metadata for each ONNX state input name.
        """
        return flatten_dict(self.state_metadata, "state_in")


class ScholaModel(th.nn.Module, StatefulModelMixin):
    """
    A PyTorch Module that is compatible with Schola inference. All Models have the following properties to allow for easy conversion to ONNX.

    - Observation and action spaces are wrapped in a Dict with keys "obs" and "action" respectively if they are not already.
    - Inputs to ``__call__`` follow the observation space (shape and dtype), typically batched.
    - Outputs from ``__call__`` follow the action space (shape and dtype), typically batched.

    Subclasses must implement :meth:`forward`. Its positional tensor arguments are
    observation tensors in ``observation_space`` key order, followed when
    :attr:`~StatefulModelMixin.is_stateful` is true by state tensors in
    ``input_state_keys`` order. Returned tensors must match ``output_action_keys``
    order (and emit updated state tensors matching ``output_state_keys`` when stateful).

    Parameters
    ----------
    observation_space : gym.Space
        The observation space of the model. If not a gym.spaces.Dict, it will be wrapped in a Dict with a single key "obs".
    action_space : gym.Space
        The action space of the model. If not a gym.spaces.Dict, it will be wrapped in a Dict with a single key "action".

    Attributes
    ----------
    observation_space : gym.spaces.Dict
        The observation space of the model.
    action_space : gym.spaces.Dict
        The action space of the ScholaModel.
    flat_dims : Dict[str, int]
        A dictionary of the flat dimensions of the action spaces. Used to convert logits outputs to the correct output shapes.
    """

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        super().__init__()
        self.observation_space_is_natively_dict = True
        # The Schola model operates on named inputs/outputs so we need to wrap the observation/action spaces in a Dict if they are not already
        if not isinstance(observation_space, gym.spaces.Dict):
            self.observation_space_is_natively_dict = False
            observation_space = gym.spaces.Dict({"obs": observation_space})

        if not isinstance(action_space, gym.spaces.Dict):
            action_space = gym.spaces.Dict({"action": action_space})

        self.observation_space = observation_space
        self.action_space = action_space
        self.flat_dims = self.get_logit_dimensions()

    @cached_property
    def input_obs_keys(self) -> List[str]:
        """
        Keys of ``observation_space`` in forward / export input order.

        Returns
        -------
        list of str
            Keys of ``observation_space`` in forward / export input order.
        """
        return list(self.observation_space.keys())

    @cached_property
    def output_action_keys(self) -> List[str]:
        """
        Returns
        -------
        list of str
            Keys of ``action_space`` in forward / export output order.
        """
        return list(self.action_space.keys())

    def forward(self, *args: th.Tensor) -> Tuple[th.Tensor, ...]:
        raise NotImplementedError("forward method must be implemented in subclass")

    def get_logit_dimensions(self) -> Dict[str, int]:
        """
        Get the flat dimensions of the action spaces.
        Returns
        -------
        Dict[str, int]
            Flat size per action dict key (``gymnasium.spaces.flatdim`` on each subspace).
        """
        return {k: flatdim(v) for k, v in self.action_space.items()}

    # utility functions for converting logits to outputs
    def make_box_output(
        self, logits: th.Tensor, space_name: str = "action"
    ) -> th.Tensor:
        """
        Map logits to a :class:`gymnasium.spaces.Box` action slice (identity for Box).

        Parameters
        ----------
        logits : torch.Tensor
            Logits slice for ``space_name`` (typically shaped for one fundamental space).
        space_name : str, optional
            Key in ``action_space`` used only for symmetry with other ``make_*`` helpers.

        Returns
        -------
        torch.Tensor
            Box action tensor (unchanged logits).
        """
        return logits

    def make_discrete_output(
        self, logits: th.Tensor, space_name: str = "action"
    ) -> th.Tensor:
        """
        Map logits to a :class:`gymnasium.spaces.Discrete` action (argmax).

        Parameters
        ----------
        logits : torch.Tensor
            Logits for the discrete branch.
        space_name : str, optional
            Key in ``action_space`` (unused for Discrete; kept for API uniformity).

        Returns
        -------
        torch.Tensor
            Discrete action index from :meth:`torch.Tensor.argmax`.
        """
        return logits.argmax()

    def make_multi_binary_output(
        self, logits: th.Tensor, space_name: str = "action"
    ) -> th.Tensor:
        """
        Map logits to a :class:`gymnasium.spaces.MultiBinary` action.

        Parameters
        ----------
        logits : torch.Tensor
            Logits for the multi-binary branch.
        space_name : str, optional
            Key in ``action_space`` (unused; kept for API uniformity).

        Returns
        -------
        torch.Tensor
            Boolean tensor from rounded logits.
        """
        return logits.round().to(th.bool)

    def make_multi_discrete_output(
        self, logits: th.Tensor, space_name: str = "action"
    ) -> th.Tensor:
        """
        Map logits to a :class:`gymnasium.spaces.MultiDiscrete` action (per-section argmax).

        Parameters
        ----------
        logits : torch.Tensor
            Concatenated logits aligned with ``action_space[space_name].nvec``.
        space_name : str, optional
            Key of the :class:`~gymnasium.spaces.MultiDiscrete` subspace in ``action_space``.

        Returns
        -------
        torch.Tensor
            One integer index per discrete component, stacked on dimension 0.
        """
        # take max over each section of the Multidiscrete space
        nvec = self.action_space.spaces[space_name].nvec  # type: ignore
        indices = list(accumulate(nvec[:-1]))
        index_tensors = []
        for tensor in logits.tensor_split(indices):
            max_indices = tensor.argmax()
            index_tensors.append(max_indices)
        return th.stack(index_tensors, dim=0)

    def make_fundamental_output(
        self, logits: th.Tensor, space_name: str = "action"
    ) -> th.Tensor:
        """
        Dispatch to the appropriate ``make_*_output`` helper for ``space_name``.

        Parameters
        ----------
        logits : torch.Tensor
            Logits slice for the fundamental space at ``space_name``.
        space_name : str, optional
            Key in ``action_space``.

        Returns
        -------
        torch.Tensor
            Action tensor for Box, Discrete, MultiDiscrete, or MultiBinary subspaces.

        Raises
        ------
        ValueError
            If the subspace type is not supported.
        """
        space = self.action_space.spaces[space_name]
        # space name is so that things can be looked up later when implementing in a child class
        if isinstance(space, Box):
            return self.make_box_output(logits, space_name=space_name)
        if isinstance(space, gym.spaces.Discrete):
            return self.make_discrete_output(logits, space_name=space_name)
        elif isinstance(space, gym.spaces.MultiDiscrete):
            return self.make_multi_discrete_output(logits, space_name=space_name)
        elif isinstance(space, gym.spaces.MultiBinary):
            return self.make_multi_binary_output(logits, space_name=space_name)
        else:
            raise ValueError(f"Unsupported space type: {type(space)}")

    def make_outputs(self, logits: th.Tensor) -> List[th.Tensor]:
        """
        Split concatenated logits and produce one output tensor per action key.

        Parameters
        ----------
        logits : torch.Tensor
            Concatenated logits over action branches (sequence dimensions flattened to batch).

        Returns
        -------
        list of torch.Tensor
            One tensor per ``output_action_key``, from :meth:`make_fundamental_output`
            applied along the batch dimension via :func:`torch.vmap`.
        """
        logits = logits.flatten(start_dim=1)  # get rid of any sequence dimensions
        idx_list = list(accumulate(list(self.flat_dims.values())[:-1]))
        # vmap the make_fundamental_output function over the logits tensor, keeping the space_name constant
        batched_fn = th.vmap(self.make_fundamental_output, in_dims=0)
        outputs = []
        for name, chunk in zip(
            self.output_action_keys, logits.tensor_split(idx_list, dim=-1)
        ):
            # use kwargs here to make it explicitly not a batchable parameter
            output_chunk = batched_fn(chunk, space_name=name)
            outputs.append(output_chunk)
        return outputs

    def export_onnx_program(self, onnx_opset: int = 21) -> th.onnx.ONNXProgram:
        """
        Export the model as an ONNX program.

        The model has the following properties:
        - Inputs are named based on they key in the observation space.
        - Outputs are named based on they key in the action space.
        - State Inputs have metadata that contains the sequence dimension and max sequence length if they have a sequence dimension.

        Parameters
        ----------
        onnx_opset : int, optional
            The ONNX opset version to use for the export.

        Returns
        -------
        th.onnx.ONNXProgram
            The ONNX program generated with torch dynamo export.
        """
        self.eval()
        # make directories if they don't exist
        obs_inputs = []
        batch_dim = Dim("batch_size")
        seq_dim = Dim("seq_len")

        for obs_space_name, obs_space in self.observation_space.spaces.items():
            # Just flatten discrete and boolean spaces
            # add the batch dimension to the sample
            obs_inputs.append(th.as_tensor(obs_space.sample()).unsqueeze(0))

        obs_input_shapes = ({0: batch_dim} for _ in obs_inputs)

        # default to empty iterators
        state_input_shapes_generator = ()
        state_input_generator = ()

        # add the state input, we could just plug the values directly but this is more flexible in the case someone does something weird
        # setting is_stateful to False and adding a state_input_dict
        if self.is_stateful:
            # add batch and sequence dimensions to the state inputs
            state_input_generator = (
                v.reshape(1, *v.shape) for v in self.input_state_dict.values()
            )
            state_dynamic_shapes_fn = lambda k, v: (
                {0: batch_dim, v.seq_dim: seq_dim} if v.has_seq_dim else {0: batch_dim}
            )
            state_input_shapes_generator = (
                state_dynamic_shapes_fn(k, metadata)
                for k, metadata in self.input_state_metadata.items()
            )

        input_args = (*obs_inputs, *state_input_generator)
        input_names = (*self.input_obs_keys, *self.input_state_keys)
        input_shapes = (*obs_input_shapes, *state_input_shapes_generator)

        output_names = (*self.output_action_keys, *self.output_state_keys)

        if set(input_names).intersection(set(output_names)):
            raise ValueError(
                f"Input and output names must be unique. Reused Names: {set(input_names).intersection(set(output_names))}"
            )

        # State inputs have shape: [batch_size, *state_dim] (state_dim may include a sequence dimension)
        # Observation inputs have shape: [batch_size, *obs_dim]
        with th.no_grad():
            handles = patch_lstm_layers_for_onnx_export(self)
            onnx_program = th.onnx.export(
                self,
                args=input_args,
                input_names=input_names,
                opset_version=onnx_opset,
                output_names=output_names,
                dynamic_shapes=(input_shapes,),
                dynamo=True,
                report=False,
                optimize=False,
                verbose=False,
            )

            for handle in handles:
                handle.remove()

            assert (
                onnx_program is not None
            ), "Expected ONNX program to be generated after calling th.onnx.export"
            fix_slice_nodes_for_onnx(onnx_program.model)
            fix_lstm_output_shapes_for_onnx(onnx_program.model)
            onnx_ir.passes.common.shape_inference.infer_shapes(
                onnx_program.model
            )
            normalize_shape_slices_to_gather(onnx_program.model)
            fold_static_shape_gather_constants(onnx_program.model)
            onnxscript.optimizer.optimize(onnx_program.model)
            onnx_ir.passes.common.shape_inference.infer_shapes(
                onnx_program.model
            )
            allowed_dynamic_dims = {"batch_size"}
            if any(
                metadata.has_seq_dim
                for metadata in self.input_state_metadata.values()
            ):
                allowed_dynamic_dims.add("seq_len")
            assert_shapes_fully_resolved(
                onnx_program.model, allowed_dynamic_dims
            )
            # Embed state metadata on each state input's doc_string
            for inp in onnx_program.model.graph.inputs:
                if inp.name in self.input_state_metadata:
                    inp.metadata_props.update(
                        self.input_state_metadata[inp.name].to_dict()
                    )
            if self.input_state_metadata:
                validate_exported_onnx_state_shapes(
                    onnx_program.model, self.input_state_metadata
                )
            return onnx_program

    def save_as_onnx(
        self, export_path: str | pathlib.Path, onnx_opset: int = 21
    ) -> None:
        """
        Export this model to an ``.onnx`` file on disk.

        Parameters
        ----------
        export_path : str or pathlib.Path
            Output file path; parent directories are created if missing.
        onnx_opset : int, optional
            ONNX opset passed to :meth:`export_onnx_program`.

        Returns
        -------
        None
        """
        dir_path = pathlib.Path(export_path).parent
        dir_path.mkdir(parents=True, exist_ok=True)
        onnx_program = self.export_onnx_program(onnx_opset)
        onnx_program.save(export_path)


def patch_lstm_layers_for_onnx_export(
    module: th.nn.Module,
) -> List[th.utils.hooks.RemovableHandle]:
    """
    Attach forward hooks so ONNX-exported LSTM hidden states match PyTorch.

    Parameters
    ----------
    module : torch.nn.Module
        Root module; every nested ``torch.nn.LSTM`` receives
        ``reshape_lstm_output_hook``.

    Returns
    -------
    list of torch.utils.hooks.RemovableHandle
        Hook handles to remove after export (from ``register_forward_hook`` on each
        nested ``torch.nn.LSTM``).

    Notes
    -----
    Intended for use only around ``torch.onnx.export``; hooks reshape ``hn``/``cn``.
    """
    handles = []
    for sub_module in module.modules():
        if isinstance(sub_module, th.nn.LSTM):
            handles.append(sub_module.register_forward_hook(reshape_lstm_output_hook))
    return handles


def reshape_lstm_output_hook(
    lstm: th.nn.LSTM,
    args: Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]],
    output: Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]],
) -> Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
    """
    Reshape LSTM hidden states during ONNX export so ``hn`` / ``cn`` match PyTorch layouts.

    Only for use with :func:`patch_lstm_layers_for_onnx_export`; requires batched LSTM inputs.

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
    This hook is only used for ONNX export, and is not used for normal forward passes.
    """
    x_in, _ = args
    x_out, (hn, cn) = output

    bidirectional_modifier = 2 if lstm.bidirectional else 1
    layer_dim = bidirectional_modifier * lstm.num_layers

    # Tie the batch axis to the traced input batch size so the
    # exported graph does not leak an extra unresolved dimension that Unreal NNE
    # would misread as a sequence axis via FindLast(-1).
    batch_size = x_in.shape[0]
    hn = hn.reshape(layer_dim, batch_size, lstm.hidden_size)
    cn = cn.reshape(layer_dim, batch_size, lstm.hidden_size)

    return x_out, (hn, cn)


def fix_slice_nodes_for_onnx(model: ir.Model) -> None:
    """
    Fix Slice nodes produced by ``torch.onnx.export`` that use invalid 0-D tensor inputs.

    Parameters
    ----------
    model : onnx_ir.ir.Model
        ONNX IR model to mutate in place.
    """
    fixed_values: Set[str] = set()
    for node in model.graph.all_nodes():
        if node.op_type != "Slice":
            continue
        for node_input in node.inputs:
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

    if fixed_values:
        logger.info("Fixed %s slice initializer(s): %s", len(fixed_values), fixed_values)


def _static_int_values(value: ir.Value) -> Optional[np.ndarray]:
    """Return flattened integer values from an initializer or Constant output."""
    if value is None:
        return None
    if value.const_value is not None:
        return value.const_value.numpy().reshape(-1)
    producer = value.producer()
    if producer is None or producer.op_type != "Constant":
        return None
    constant_attr = producer.attributes.get("value")
    if constant_attr is None:
        return None
    constant_tensor = constant_attr.as_tensor()
    if constant_tensor is None:
        return None
    return constant_tensor.numpy().reshape(-1)


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

        gather_index = int(start_values[0])
        slice_output = node.outputs[0]
        if slice_output is None or slice_output.name is None:
            continue
        indices_name = f"{slice_output.name}_gather_idx"
        indices_value = ir.val(
            indices_name,
            const_value=ir.tensor(
                np.array([gather_index], dtype=np.int64),
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

    onnxscript's optimizer does not always fold these gathers before reshape
    shape tensors are built, which leaves ``unk__*`` on LSTM collapse reshapes.
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
        dim_index = int(indices[0])
        if dim_index < 0 or dim_index >= shaped_input.shape.rank():
            continue
        dim_repr = _dim_to_repr(shaped_input.shape.dims[dim_index])
        if _is_dynamic_dim_repr(dim_repr):
            continue

        gather_output = node.outputs[0]
        if gather_output is None or gather_output.name is None:
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


def assert_shapes_fully_resolved(
    model: ir.Model, allowed_dynamic_dims: Set[str]
) -> None:
    """
    Raise if any graph tensor shape still contains unknown or disallowed symbolic dims.

    Checks graph I/O and every intermediate node output so partially-resolved
    tensors such as LSTM reshape intermediates cannot slip through validation.

    Parameters
    ----------
    model : onnx_ir.ir.Model
        ONNX IR model to validate.
    allowed_dynamic_dims : set of str
        Symbolic dimension names that may remain dynamic (for example
        ``batch_size`` or ``seq_len``).
    """
    checked: Set[str] = set()

    def check_value(value: ir.Value) -> None:
        value_name = value.name
        if value_name is None or value_name in checked:
            return
        checked.add(value_name)
        if value.shape is None:
            raise ValueError(
                f"Exported ONNX tensor '{value_name}' has no resolved shape"
            )
        for dim_index, dim in enumerate(value.shape.dims):
            dim_repr = _dim_to_repr(dim)
            if not _is_dynamic_dim_repr(dim_repr):
                continue
            if isinstance(dim_repr, str) and dim_repr.startswith("unk"):
                raise ValueError(
                    "Exported ONNX tensor "
                    f"'{value_name}' dimension {dim_index} uses auto-generated "
                    f"symbolic name {dim_repr!r}"
                )
            if dim_repr not in allowed_dynamic_dims:
                raise ValueError(
                    "Exported ONNX tensor "
                    f"'{value_name}' dimension {dim_index} is unresolved "
                    f"({dim_repr!r}); allowed dynamic dims are "
                    f"{sorted(allowed_dynamic_dims)}"
                )

    for graph_input in model.graph.inputs:
        check_value(graph_input)
    for graph_output in model.graph.outputs:
        check_value(graph_output)
    for node in model.graph.all_nodes():
        for value in node.outputs:
            if value is not None:
                check_value(value)


def fix_lstm_output_shapes_for_onnx(model: ir.Model) -> None:
    """
    Drop conflicting rank-4 annotations on ONNX ``LSTM`` hidden/cell outputs.

    ``torch.onnx.export`` annotates the LSTM ``Y_h``/``Y_c`` outputs as rank-4
    ``[num_layers, num_directions, batch, hidden]``, but the ONNX ``LSTM`` spec
    requires them to be rank-3 ``[num_directions * num_layers, batch, hidden]``.
    That rank mismatch makes ``onnx.shape_inference`` abort on the whole graph,
    leaving every tensor downstream of the LSTM with an unresolved shape. Clearing the bad annotations
    lets shape inference recompute spec-correct shapes and propagate them.

    Parameters
    ----------
    model : onnx_ir.ir.Model
        ONNX IR model to mutate in place.

    Raises
    ------
    ValueError
        If an ``LSTM`` node does not expose the expected ``(Y, Y_h, Y_c)`` outputs
        or if ``Y_h`` / ``Y_c`` carry an unexpected rank annotation.
    """
    cleared = 0
    for node in model.graph.all_nodes():
        if node.op_type != "LSTM":
            continue
        if len(node.outputs) != 3:
            raise ValueError(
                f"Expected LSTM node {node.name!r} to have exactly 3 outputs, "
                f"got {len(node.outputs)}"
            )
        y_output, y_h_output, y_c_output = node.outputs
        if y_output is None or y_h_output is None or y_c_output is None:
            raise ValueError(
                f"LSTM node {node.name!r} must expose Y, Y_h, and Y_c outputs"
            )
        for state_name, state_output in (("Y_h", y_h_output), ("Y_c", y_c_output)):
            if state_output.shape is None:
                continue
            if state_output.shape.rank() != 4:
                raise ValueError(
                    f"LSTM node {node.name!r} has unexpected {state_name} rank "
                    f"{state_output.shape.rank()}; expected rank 4 before clearing "
                    "mis-ranked annotations"
                )
            state_output.shape = None
            cleared += 1

    if cleared:
        logger.info("Cleared %s mis-ranked LSTM state output annotation(s)", cleared)
