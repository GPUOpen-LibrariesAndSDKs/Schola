# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
ONNX export metadata, ``ScholaModel``, and related helpers for policies trained with Schola.
"""

from dataclasses import dataclass
import logging
import pathlib
from collections.abc import Iterator
from functools import cached_property
from itertools import accumulate
from typing import cast

from typing_extensions import override

import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete, Space, flatdim
import torch as th
from torch.export import Dim

from schola.core.onnx_postprocess import (
    patch_lstm_layers_for_onnx_export,
    postprocess_exported_onnx,
)
from schola.core.onnx_validation import allowed_dynamic_dims_for_state_metadata
from schola.core.utils.dict_helpers import *

logger = logging.getLogger(__name__)


# bit overkill for now but we can extend later if we need more metadata
@dataclass
class StateMetadata:
    """
    Metadata for recurrent or sequential state tensors in ONNX export.
    """

    has_seq_dim: bool = False  # whether the state input has a sequence dimension
    max_seq_len: int | None = None  # maximum sequence length for the state input
    seq_dim: int | None = None  # index of the sequence dimension

    def to_dict(self) -> dict[str, str]:
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
    def input_state_dict(self) -> dict[str, th.Tensor]:
        """
        Flattened state inputs keyed as ``state_in/...`` for export and ONNX naming.

        Returns
        -------
        Dict[str, torch.Tensor]
            Flattened state inputs keyed as ``state_in/...`` for export and ONNX naming.
        """
        return flatten_dict(self.initial_state_dict, "state_in")

    @cached_property
    def input_state_keys(self) -> list[str]:
        """
        Keys of ``input_state_dict`` for export and ONNX naming.

        Returns
        -------
        list of str
            Keys of ``input_state_dict``.
        """
        return list(self.input_state_dict.keys())

    @cached_property
    def output_state_keys(self) -> list[str]:
        """
        Keys of ``output_state_dict`` for export and ONNX naming.

        Returns
        -------
        list of str
            Flattened output state names (``state_out/...``) derived from ``initial_state_dict``.
        """
        return list(flattened_key_iterator(self.initial_state_dict, "state_out"))

    @cached_property
    def input_state_metadata(self) -> dict[str, StateMetadata]:
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

    def __init__(
        self, observation_space: Space[object], action_space: Space[object]
    ) -> None:
        super().__init__()
        self.observation_space_is_natively_dict: bool = True
        # The Schola model operates on named inputs/outputs so we need to wrap the observation/action spaces in a Dict if they are not already
        if not isinstance(observation_space, gym.spaces.Dict):
            self.observation_space_is_natively_dict = False
            observation_space = gym.spaces.Dict({"obs": observation_space})

        if not isinstance(action_space, gym.spaces.Dict):
            action_space = gym.spaces.Dict({"action": action_space})

        self.observation_space: gym.spaces.Dict = observation_space
        self.action_space: gym.spaces.Dict = action_space
        self.flat_dims: dict[str, int] = self.get_logit_dimensions()

    @cached_property
    def input_obs_keys(self) -> list[str]:
        """
        Keys of ``observation_space`` in forward / export input order.

        Returns
        -------
        list of str
            Keys of ``observation_space`` in forward / export input order.
        """
        return list(self.observation_space.spaces)

    @cached_property
    def output_action_keys(self) -> list[str]:
        """
        Returns
        -------
        list of str
            Keys of ``action_space`` in forward / export output order.
        """
        return list(self.action_space.spaces)

    @override
    def forward(self, *args: th.Tensor) -> tuple[th.Tensor, ...]:
        raise NotImplementedError("forward method must be implemented in subclass")

    def get_logit_dimensions(self) -> dict[str, int]:
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
        self, logits: th.Tensor, _space_name: str = "action"
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
        self, logits: th.Tensor, _space_name: str = "action"
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
        self, logits: th.Tensor, _space_name: str = "action"
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
        space = self.action_space.spaces[space_name]
        if not isinstance(space, MultiDiscrete):
            raise TypeError(
                f"Expected MultiDiscrete action space for {space_name!r}, got {type(space)}"
            )
        nvec = space.nvec
        split_indices: list[int] = list(
            accumulate(cast(list[int], nvec[:-1].tolist()))
        )
        index_tensors: list[th.Tensor] = []
        for tensor in logits.tensor_split(split_indices):
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
            return self.make_box_output(logits, space_name)
        if isinstance(space, gym.spaces.Discrete):
            return self.make_discrete_output(logits, space_name)
        elif isinstance(space, gym.spaces.MultiDiscrete):
            return self.make_multi_discrete_output(logits, space_name)
        elif isinstance(space, gym.spaces.MultiBinary):
            return self.make_multi_binary_output(logits, space_name)
        else:
            raise ValueError(f"Unsupported space type: {type(space)}")

    def make_outputs(self, logits: th.Tensor) -> list[th.Tensor]:
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
        idx_list: list[int] = list(accumulate(list(self.flat_dims.values())[:-1]))
        # vmap the make_fundamental_output function over the logits tensor, keeping the space_name constant
        batched_fn = th.vmap(self.make_fundamental_output, in_dims=0)
        outputs: list[th.Tensor] = []
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

        ``torch.onnx.export`` is called with ``optimize=False`` so custom
        Shape/Slice/Gather rewrites in :func:`~schola.core.onnx_postprocess.postprocess_exported_onnx`
        can run before ``onnxscript.optimizer.optimize``. Do not enable torch's
        built-in optimizer without re-running the full postprocess pipeline.

        Parameters
        ----------
        onnx_opset : int, optional
            The ONNX opset version to use for the export.

        Returns
        -------
        th.onnx.ONNXProgram
            The ONNX program generated with torch dynamo export.
        """
        _ = self.eval()
        # make directories if they don't exist
        obs_inputs: list[th.Tensor] = []
        batch_dim = Dim("batch_size")
        seq_dim = Dim("seq_len")

        for _, obs_space in self.observation_space.spaces.items():
            # Just flatten discrete and boolean spaces
            # add the batch dimension to the sample
            obs_inputs.append(th.as_tensor(obs_space.sample()).unsqueeze(0))

        obs_input_shapes: tuple[dict[int, Dim], ...] = tuple(
            {0: batch_dim} for _ in obs_inputs
        )

        # default to empty iterators
        state_input_shapes_generator: Iterator[dict[int, Dim]] = iter(())
        state_input_generator: Iterator[th.Tensor] = iter(())

        # add the state input, we could just plug the values directly but this is more flexible in the case someone does something weird
        # setting is_stateful to False and adding a state_input_dict
        if self.is_stateful:

            def state_dynamic_shapes(
                _key: str, metadata: StateMetadata
            ) -> dict[int, Dim]:
                if metadata.has_seq_dim:
                    if metadata.seq_dim is None:
                        raise ValueError(
                            "State metadata sets has_seq_dim=True but omits seq_dim"
                        )
                    return {0: batch_dim, metadata.seq_dim: seq_dim}
                return {0: batch_dim}

            # add batch and sequence dimensions to the state inputs
            state_input_generator = (
                v.reshape(1, *v.shape) for v in self.input_state_dict.values()
            )
            state_input_shapes_generator = (
                state_dynamic_shapes(k, metadata)
                for k, metadata in self.input_state_metadata.items()
            )

        input_args: tuple[th.Tensor, ...] = (
            *obs_inputs,
            *state_input_generator,
        )
        input_names = (*self.input_obs_keys, *self.input_state_keys)
        input_shapes: tuple[dict[int, Dim], ...] = (
            *obs_input_shapes,
            *state_input_shapes_generator,
        )

        output_names = (*self.output_action_keys, *self.output_state_keys)

        if set(input_names).intersection(set(output_names)):
            raise ValueError(
                f"Input and output names must be unique. Reused Names: {set(input_names).intersection(set(output_names))}"
            )

        # State inputs have shape: [batch_size, *state_dim] (state_dim may include a sequence dimension)
        # Observation inputs have shape: [batch_size, *obs_dim]
        with th.no_grad():
            handles = patch_lstm_layers_for_onnx_export(self)
            onnx_program = None
            try:
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
            finally:
                for handle in handles:
                    handle.remove()

            assert (
                onnx_program is not None
            ), "Expected ONNX program to be generated after calling th.onnx.export"
            postprocess_exported_onnx(
                onnx_program.model,
                state_metadata=self.input_state_metadata,
                allowed_dynamic_dims=allowed_dynamic_dims_for_state_metadata(
                    self.input_state_metadata
                ),
            )
            # Embed state metadata on each state input's doc_string
            for inp in onnx_program.model.graph.inputs:
                if inp.name in self.input_state_metadata:
                    inp.metadata_props.update(
                        self.input_state_metadata[inp.name].to_dict()
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


# Re-export ONNX helpers for backward compatibility.
from schola.core.onnx_postprocess import (  # noqa: E402
    assert_shapes_fully_resolved,
    fix_lstm_output_shapes_for_onnx,
    fix_slice_nodes_for_onnx,
    fold_static_shape_gather_constants,
    normalize_shape_slices_to_gather,
    reshape_lstm_output_hook,
)
from schola.core.onnx_validation import (  # noqa: E402
    emulate_nne_seq_dim,
    tensor_dims_from_value_info,
    validate_exported_onnx_state_shapes,
)

__all__ = [
    "StateMetadata",
    "StatefulModelMixin",
    "ScholaModel",
    "assert_shapes_fully_resolved",
    "emulate_nne_seq_dim",
    "fix_lstm_output_shapes_for_onnx",
    "fix_slice_nodes_for_onnx",
    "fold_static_shape_gather_constants",
    "normalize_shape_slices_to_gather",
    "patch_lstm_layers_for_onnx_export",
    "postprocess_exported_onnx",
    "reshape_lstm_output_hook",
    "tensor_dims_from_value_info",
    "validate_exported_onnx_state_shapes",
]
