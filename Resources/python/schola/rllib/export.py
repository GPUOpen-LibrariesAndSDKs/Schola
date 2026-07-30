# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Code for Exporting RLlib Polices to the Schola Inference Interface"""

from collections import defaultdict
from itertools import chain
import pathlib
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.sac import SACTorchPolicy
from ray.rllib.core.rl_module import MultiRLModule
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.models import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.torch_utils import flatten_inputs_to_1d_tensor
import torch.nn as nn
from gymnasium.spaces import Box, flatdim
from functools import singledispatch
from ray.rllib.policy import Policy

import torch as th
from ray.rllib.policy.sample_batch import SampleBatch
import os
import numpy as np
from schola.core.model import *
import gymnasium as gym
from gymnasium import spaces
import copy
from schola.rllib.env import RayVecEnv
from torch.export.dynamic_shapes import Dim
from typing import Any, List, Optional, Set, Tuple, Dict, Union
import logging

logger = logging.getLogger(__name__)


@singledispatch
def export_onnx_from_policy(arg: Any, path: pathlib.Path) -> None:
    """
    Export an RLlib policy to ONNX format.

    This is a generic function that uses singledispatch to handle different
    input types (Policy objects, dictionaries of policies, or checkpoint paths).

    Parameters
    ----------
    arg : Policy or dict or str
        The policy to export. Can be:
        - A Policy object
        - A dictionary mapping policy names to Policy objects
        - A string path to a policy checkpoint
        - A pathlib.Path object to a policy checkpoint
    path : str
        The directory path where the ONNX model will be saved.
    policy_name : str, optional
        The name to use for the exported policy file. If None, uses "Policy" as default.

    Raises
    ------
    TypeError
        If the argument type is not supported.

    Notes
    -----
    This function has multiple registered implementations for different
    argument types. See the individual @export_onnx_from_policy.register
    implementations for specific type handling.
    """
    raise TypeError(
        f"Cannot export ONNX from Policy/Checkpoint stored as {type(arg)}. Pass a Policy, Dictionary of Policies, or a path to a Policy Checkpoint"
    )


@export_onnx_from_policy.register
def _(
    arg: Policy,
    path: pathlib.Path,
    observation_space: gym.Space[Any] | None = None,
    action_space: gym.Space[Any] | None = None,
):
    if path.is_dir():
        path = path / "default_policy.onnx"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    assert isinstance(arg, TorchPolicyV2), f"Expected TorchPolicyV2, got {type(arg)}"

    schola_model = RllibScholaModel(
        arg, observation_space=observation_space, action_space=action_space
    )
    schola_model.save_as_onnx(path)


@export_onnx_from_policy.register
def _(
    arg: dict,
    path: pathlib.Path,
    observation_spaces: Dict[str, gym.Space[Any]] | None = None,
    action_spaces: Dict[str, gym.Space[Any]] | None = None,
):
    observation_spaces = observation_spaces if observation_spaces is not None else {}
    action_spaces = action_spaces if action_spaces is not None else {}
    if path.is_file():
        path = path.parent
        logger.warning(
            "Path is a file but a dictionary of policies was passed, using parent directory: %s",
            path,
        )
    # policy name is ignored, as the dictionary has them already
    for policy_name, policy in arg.items():
        export_onnx_from_policy(
            policy,
            path / f"{policy_name}.onnx",
            observation_space=observation_spaces.get(policy_name, None),
            action_space=action_spaces.get(policy_name, None),
        )


@export_onnx_from_policy.register
def _(arg: str, path: pathlib.Path):
    # dir and other stuff is handled later
    policy = Policy.from_checkpoint(arg)
    export_onnx_from_policy(policy, path)


@export_onnx_from_policy.register
def _(arg: pathlib.Path, path: pathlib.Path):
    # dir and other stuff is handled later
    policy = Policy.from_checkpoint(str(arg))
    export_onnx_from_policy(policy, path)


@export_onnx_from_policy.register
def _(
    arg: TorchRLModule,
    path: pathlib.Path,
    observation_space: gym.Space[Any] | None = None,
    action_space: gym.Space[Any] | None = None,
):
    if path.is_dir():
        path = path / "default_policy.onnx"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    schola_model = ScholaRLModule(
        arg, observation_space=observation_space, action_space=action_space
    )
    schola_model.save_as_onnx(path)


@export_onnx_from_policy.register
def _(
    arg: MultiRLModule,
    path: pathlib.Path,
    observation_spaces: Dict[str, gym.Space] = None,
    action_spaces: Dict[str, gym.Space] = None,
):
    """
    Export a multiagent module to ONNX format.
    Parameters
    ----------
    arg : MultiRLModule
        The multiagent module to export.
    path : pathlib.Path
        The path to save the ONNX model to.
    observation_spaces : gym.spaces.Dict
        The observation spaces defined as per policy_id
    action_spaces : gym.spaces.Dict
        The action spaces defined as per policy_id
    """
    if path.is_file():
        path = path.parent
        logger.warning(
            "Path is a file but a multiagent module was passed, using parent directory: %s",
            path,
        )
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    for k, module in arg.items():
        # remap the observation space which has agents as keys to the policy name
        export_onnx_from_policy(
            module, path / f"{k}.onnx", observation_spaces[k], action_spaces[k]
        )


@export_onnx_from_policy.register
def _(arg: Algorithm, path: pathlib.Path):
    # arg.spaces["__env_single__"] is a tuple (observation_space, action_space)
    if arg.spaces is None:
        raise RuntimeError("No spaces found on Algorithm while exporting to ONNX")
    elif "__env_single__" not in arg.spaces:
        raise RuntimeError(
            "No single environment spaces found on Algorithm while exporting to ONNX. Expected key '__env_single__'"
        )

    obs_space, act_space = arg.spaces["__env_single__"]

    if arg.env_runner is not None:
        module = arg.env_runner.module
    elif arg.env_runner_group is not None:
        module = arg.env_runner_group.foreach_env_runner(
            lambda er: er.module,
            remote_worker_ids=[1],
            local_env_runner=False,
        )[0]
    else:
        raise RuntimeError(
            "No env runner or env runner group found while exporting Algorithm to ONNX"
        )
    # The space will be defined as a Dict Space with agent_ids as the key, so we condense down to policy_ids.
    # All agent's mapping to the same policy will have the same observation space so we can just overwrite the values.
    # We can't use the policy observation space as it is already flattened.
    if isinstance(module, MultiRLModule) and arg.get_config() is not None:
        obs_space = {
            arg.get_config().policy_mapping_fn(k): v for k, v in obs_space.items()
        }
        act_space = {
            arg.get_config().policy_mapping_fn(k): v for k, v in act_space.items()
        }
    export_onnx_from_policy(module, path, obs_space, act_space)


class ScholaRLModule(ScholaModel):
    """
    ``ScholaModel`` adapter around an RLlib ``RLModule`` for ONNX export.

    Parameters
    ----------
    rl_module : ray.rllib.core.rl_module.rl_module.RLModule
        Source module whose forward pass is wrapped.
    observation_space : gymnasium.spaces.Space, optional
        Override observation space; defaults to ``rl_module.observation_space``.
    action_space : gymnasium.spaces.Space, optional
        Override action space; defaults to ``rl_module.action_space``.
    ignored_state_keys : tuple of str, optional
        State dict keys to drop before export (default hides critic-only state).

    See Also
    --------
    ScholaModel
    export_onnx_from_policy
    """

    def __init__(
        self,
        rl_module: TorchRLModule,
        observation_space: gym.Space[Any] | None = None,
        action_space: gym.Space[Any] | None = None,
        ignored_state_keys: tuple[str, ...] = ("critic",),
    ):
        super().__init__(
            observation_space=(
                observation_space
                if observation_space is not None
                else rl_module.observation_space
            ),
            action_space=(
                action_space if action_space is not None else rl_module.action_space
            ),
        )
        self.rl_module = rl_module.to("cpu")
        self.rl_module.eval()
        if observation_space is None and isinstance(
            self.rl_module.observation_space, gym.spaces.Box
        ):
            logger.warning(
                "Potentially flattened space detected in export. This may cause your model to be different than intended when imported into Unreal."
            )
        self.ignored_state_keys = ignored_state_keys

    @property
    def initial_state_dict(self) -> NestedDict[str, th.Tensor]:
        # drop any critic state as we don't need it for onnx export
        return kfilter_dict(
            lambda x: not x.startswith(self.ignored_state_keys),
            self.rl_module.get_initial_state(),
            any_key=False,
        )

    @property
    def num_state_inputs(self) -> int:
        return len(self.state_input_dict.keys())

    # override the default here because ray outputs extra variance dimensions that we don't need for box spaces
    def get_logit_dimensions(self) -> Dict[str, int]:
        custom_flat_dim_func = lambda space: (
            2 * flatdim(space) if isinstance(space, gym.spaces.Box) else flatdim(space)
        )
        return {k: custom_flat_dim_func(v) for k, v in self.action_space.items()}

    def make_box_output(
        self, logits: th.Tensor, _space_name: str = "action"
    ) -> th.Tensor:
        # must always be even so we can just divide here
        return logits[: len(logits) // 2]

    def forward(self, *args: th.Tensor) -> Tuple[th.Tensor]:

        # split the state and observation inputs into two separate dictionaries
        # Note: zip stops on the shortest iterator, so this will eventually read all the inputs
        args_iter = iter(args)
        tensor_dict = {k: v for k, v in zip(self.input_obs_keys, args_iter)}
        flat_state_dict = {
            k[len("state_in_") :]: v for k, v in zip(self.input_state_keys, args_iter)
        }
        # TODO this likely breaks shaped Image inputs being batched with other inputs.
        # ~acann, 2026-02-03
        # we need to be very specific here because it is possible that we are using LSTM but have no state inputs, if everything is filtered
        if self.rl_module.is_stateful():
            # using LSTM, so we need to add an extra dim before flattening

            module_inputs = {
                "obs": flatten_inputs_to_1d_tensor(
                    tensor_dict, self.observation_space.spaces
                ).unsqueeze(
                    1
                )  # type: ignore
            }
            batch_size = module_inputs["obs"].shape[0]

            # these could be empty iterators depending on which keys are filtered
            # these are the state tensors that we don't need during inference so we just pass a default value during export
            internal_state = DIterator(self.rl_module.get_initial_state()).kfilter(
                lambda x: x.startswith(self.ignored_state_keys)
            )
            internal_state = internal_state.map(
                lambda x: x.repeat(batch_size, *[1 for _ in x.shape])
            )
            state_from_args = (
                DIterator(self.rl_module.get_initial_state())
                .kfilter(
                    lambda x: not x.startswith(self.ignored_state_keys), any_key=False
                )
                .unflatten(flat_state_dict)
            )
            module_inputs["state_in"] = state_from_args.chain(internal_state).to_dict()

            # flatten the observation inputs keeping the batch and sequence dimensions
        else:
            module_inputs = {
                "obs": flatten_inputs_to_1d_tensor(
                    tensor_dict, self.observation_space.spaces
                )
            }

        model_out = self.rl_module.forward_inference(module_inputs)

        # Convert logits to actual outputs in the respective action spaces
        outputs = self.make_outputs(model_out["action_dist_inputs"])
        state_outputs = (
            DIterator(model_out.get("state_out", {}))
            .kfilter(lambda x: not x.startswith(self.ignored_state_keys), any_key=False)
            .values()
        )
        if self.is_stateful:
            return tuple((*outputs, *state_outputs))
        else:
            return tuple(outputs)


# The below code is adapted from https://github.com/ray-project/ray/blob/master/rllib/policy/torch_policy_v2.py
"""
Copyright 2023 Ray Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.


class RllibScholaModel(ScholaModel):
    """
    Wrapper for RLlib policies to enable ONNX export for use with Schola.

    This class adapts RLlib policies to the ScholaModel interface, allowing
    them to be exported to ONNX format for inference in Unreal Engine.

    Parameters
    ----------
    policy : ray.rllib.policy.Policy
        The RLlib policy to wrap for ONNX export.

    Attributes
    ----------
    _policy : ray.rllib.policy.Policy
        The wrapped RLlib policy.
    _model : torch.nn.Module
        The underlying PyTorch model from the policy.
    """

    def __init__(
        self,
        policy: TorchPolicyV2,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
    ):
        super().__init__(
            observation_space=(
                observation_space
                if observation_space is not None
                else policy.observation_space
            ),
            action_space=(
                action_space if action_space is not None else policy.action_space
            ),
        )
        self._policy = policy
        # for SAC the model.forward is a no-op, so we need to use the action model instead
        if isinstance(policy, SACTorchPolicy):
            # ignore typing here because rllib erases the type of the model
            self._model: TorchModelV2 = policy.model.action_model.to("cpu") # type: ignore
        else:
            self._model: TorchModelV2 = policy.model.to("cpu") # type: ignore
        # update the dummy batch here
        self._policy._dummy_batch = (
            self._policy._get_dummy_batch_from_view_requirements(1)
        )
        self._policy._lazy_tensor_dict(self._policy._dummy_batch)

    @property
    def initial_state_dict(self) -> NestedDict[str, th.Tensor]:
        # State is a 2D tensor with shape [seq_len, state_dim]
        return dict(
            map(
                lambda x: (str(x[0]), x[1].unsqueeze(0)),
                enumerate(self._model.get_initial_state()),
            )
        )

    @property
    def state_metadata(self) -> NestedDict[str, StateMetadata]:
        return (
            DIterator(self.initial_state_dict)
            .map(
                lambda x: StateMetadata(
                    has_seq_dim=True, max_seq_len=self._policy.max_seq_len, seq_dim=1
                )
            )
            .to_dict()
        )

    def forward(self, *args: th.Tensor) -> Tuple[th.Tensor, ...]:
        """
        Forward pass through the model. Removes variance outputs, to make compatible with Unreal.
        """
        if self.is_stateful:
            logger.warning(
                "Exporting with stateful models can export models with static batch size due to issues with torch.onnx.export and Ray."
            )
        inputs = {}
        seq_lens = [1] * len(self.initial_state_dict)
        args_iter = iter(args)
        obs_dict = {k: v for k, v in zip(self.input_obs_keys, args_iter)}
        state_dict = {f"state_in_{i}": v for i, v in enumerate(args_iter)}
        inputs.update(state_dict)

        # Are we wrapping a dictionary or not?
        if self.observation_space_is_natively_dict:
            inputs["obs"] = obs_dict
        else:
            inputs["obs"] = obs_dict["obs"]

        inputs["obs_flat"] = flatten_inputs_to_1d_tensor(
            obs_dict, self.observation_space
        )

        logits, state_outputs = self._model.forward(
            inputs, list(state_dict.values()), seq_lens
        )
        # model_out[0] is the logits, model_out[1] is the state
        # check if state is 3D meaning a rnn model, if not, view it as 1x1x1
        # Logits output other miscelanous outputs so we need to mask them out
        return self.make_outputs(logits) + state_outputs

    # override the default here because ray outputs extra variance dimensions that we don't need for box spaces
    def get_logit_dimensions(self) -> Dict[str, int]:
        custom_flat_dim_func = lambda space: (
            2 * flatdim(space) if isinstance(space, gym.spaces.Box) else flatdim(space)
        )
        return {k: custom_flat_dim_func(v) for k, v in self.action_space.items()}

    def make_box_output(
        self, logits: th.Tensor, _space_name: str = "action"
    ) -> th.Tensor:
        # must always be even so we can just divide here
        return logits[: len(logits) // 2]


# end of adapted code
