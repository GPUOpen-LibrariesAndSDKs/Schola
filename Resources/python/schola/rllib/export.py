# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Code for Exporting RLlib Polices to the Schola Inference Interface"""

from collections import defaultdict
from itertools import chain
import pathlib
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.algorithms.sac import SACTorchPolicy
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.distribution.torch.torch_distribution import (
    TorchDiagGaussian as NewTorchDiagGaussian,
    TorchMultiDistribution,
    TorchSquashedGaussian as NewTorchSquashedGaussian,
)
from ray.rllib.models.torch.torch_action_dist import (
    TorchSquashedGaussian as OldTorchSquashedGaussian,
)
from ray.rllib.core.rl_module import MultiRLModule
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.models import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.torch_utils import flatten_inputs_to_1d_tensor
from ray.rllib.utils.typing import AgentID, AlgorithmConfigDict, EpisodeType
import torch.nn as nn
from gymnasium.spaces import Box, flatdim
from functools import singledispatch
from ray.rllib.policy import Policy, TorchPolicy

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
from typing import Any, Iterable, Mapping, TypedDict, cast
import logging

from schola.rllib.checkpoint import rl_module_dir_from_algorithm_checkpoint
from schola.rllib.policy_mapping import PolicyMappingFn

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
    path : pathlib.Path
        The ONNX file, or the directory to save one ONNX file per policy in.

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
    assert isinstance(
        arg, (TorchPolicyV2, TorchPolicy)
    ), f"Expected TorchPolicyV2 or SACTorchPolicy, got {type(arg)}"

    schola_model = RllibScholaModel(
        arg, observation_space=observation_space, action_space=action_space
    )
    schola_model.save_as_onnx(path)


@export_onnx_from_policy.register
def _(
    arg: dict,
    path: pathlib.Path,
    observation_spaces: dict[str, gym.Space[Any]] | None = None,
    action_spaces: dict[str, gym.Space[Any]] | None = None,
):
    observation_spaces = _require_policy_spaces(
        observation_spaces, arg.keys(), "observation"
    )
    action_spaces = _require_policy_spaces(action_spaces, arg.keys(), "action")
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
            observation_space=observation_spaces[policy_name],
            action_space=action_spaces[policy_name],
        )


@export_onnx_from_policy.register(str)
@export_onnx_from_policy.register(pathlib.Path)
def _(
    arg: str | pathlib.Path,
    path: pathlib.Path,
    observation_space: gym.Space[Any] | dict[str, gym.Space[Any]] | None = None,
    action_space: gym.Space[Any] | dict[str, gym.Space[Any]] | None = None,
):
    """
    Export a checkpoint path. Spaces are either shared by every policy or given
    as a dict keyed by policy ID, in which case only those policies are exported.

    A single-agent checkpoint is written like an exported Algorithm: to
    ``path / "default_policy.onnx"`` if ``path`` is an existing directory,
    otherwise to ``path``.
    """
    policies = _load_checkpoint_policies(arg)
    if not isinstance(policies, (dict, MultiRLModule)):
        policies = {DEFAULT_MODULE_ID: policies}
    policies = dict(policies.items())
    if isinstance(observation_space, dict):
        policies = _select_policies(policies, observation_space.keys())
    observation_spaces = _spaces_by_policy(observation_space, policies)
    action_spaces = _spaces_by_policy(action_space, policies)

    if list(policies) == [DEFAULT_MODULE_ID]:
        export_onnx_from_policy(
            policies[DEFAULT_MODULE_ID],
            path,
            observation_spaces[DEFAULT_MODULE_ID],
            action_spaces[DEFAULT_MODULE_ID],
        )
    else:
        export_onnx_from_policy(policies, path, observation_spaces, action_spaces)


def _load_checkpoint_policies(checkpoint: str | pathlib.Path) -> Any:
    """Load a new API stack ``MultiRLModule``, or old API stack policies."""
    try:
        module_dir = rl_module_dir_from_algorithm_checkpoint(pathlib.Path(checkpoint))
    except FileNotFoundError:
        return Policy.from_checkpoint(str(checkpoint))
    return MultiRLModule.from_checkpoint(module_dir)


def _select_policies(
    policies: dict[str, Any], policy_ids: Iterable[str]
) -> dict[str, Any]:
    policy_ids = set(policy_ids)
    if unknown_ids := policy_ids - policies.keys():
        logger.warning(
            "Ignoring spaces for policy IDs not in the checkpoint: %s",
            sorted(unknown_ids),
        )
    if skipped_ids := policies.keys() - policy_ids:
        logger.warning(
            "Not exporting checkpoint policies missing from the spaces config: %s",
            sorted(skipped_ids),
        )
    selected = {
        policy_id: p for policy_id, p in policies.items() if policy_id in policy_ids
    }
    if not selected:
        raise ValueError(
            f"None of the policy IDs in the spaces config are in the checkpoint "
            f"(checkpoint has {sorted(policies)})"
        )
    return selected


def _spaces_by_policy(
    space: gym.Space[Any] | dict[str, gym.Space[Any]] | None,
    policy_ids: Iterable[str],
) -> dict[str, gym.Space[Any] | None]:
    if isinstance(space, dict):
        return {policy_id: space[policy_id] for policy_id in policy_ids}
    return {policy_id: space for policy_id in policy_ids}


def _require_policy_spaces(
    spaces: Mapping[str, gym.Space[Any] | None] | None,
    policy_ids: Iterable[str],
    kind: str,
) -> Mapping[str, gym.Space[Any] | None]:
    """
    ``None`` means no override for any policy; otherwise every policy must have
    an entry, so a missing ID can't silently fall back to the module's space.
    """
    policy_ids = list(policy_ids)
    if spaces is None:
        return dict.fromkeys(policy_ids)
    # gym.spaces.Dict overrides ``in`` to test sample membership, so use keys()
    space_ids = set(spaces.keys())
    if missing_ids := [p for p in policy_ids if p not in space_ids]:
        raise KeyError(
            f"No {kind} space given for policy IDs {missing_ids} "
            f"(spaces were given for {sorted(space_ids)})"
        )
    return spaces


@export_onnx_from_policy.register
def _(
    arg: TorchRLModule,
    path: pathlib.Path,
    observation_space: gym.Space[Any] | None = None,
    action_space: gym.Space[Any] | None = None,
    normalize_actions: bool = True,
    clip_actions: bool = False,
):
    if path.is_dir():
        path = path / "default_policy.onnx"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    schola_model = ScholaRLModule(
        arg,
        observation_space=observation_space,
        action_space=action_space,
        normalize_actions=normalize_actions,
        clip_actions=clip_actions,
    )
    schola_model.save_as_onnx(path)


@export_onnx_from_policy.register
def _(
    arg: MultiRLModule,
    path: pathlib.Path,
    observation_spaces: dict[str, gym.Space[Any]] | None = None,
    action_spaces: dict[str, gym.Space[Any]] | None = None,
    normalize_actions: bool = True,
    clip_actions: bool = False,
) -> None:
    """
    Export a multiagent module to ONNX format.
    Parameters
    ----------
    arg : MultiRLModule
        The multiagent module to export.
    path : pathlib.Path
        The path to save the ONNX model to.
    observation_spaces : Mapping[str, gym.Space], optional
        The observation spaces keyed by policy ID. Must cover every policy if
        given; None uses each module's own space.
    action_spaces : Mapping[str, gym.Space], optional
        The action spaces keyed by policy ID. Must cover every policy if
        given; None uses each module's own space.
    normalize_actions : bool, optional
        Match ``NormalizeAndClipActions`` when evaluation has
        ``normalize_actions=True`` (the RLlib default). Applied to every policy.
    clip_actions : bool, optional
        Match ``NormalizeAndClipActions`` when normalization is off and
        ``clip_actions=True``. Applied to every policy.
    """
    if path.is_file():
        path = path.parent
        logger.warning(
            "Path is a file but a multiagent module was passed, using parent directory: %s",
            path,
        )
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    observation_spaces = _require_policy_spaces(
        observation_spaces, arg.keys(), "observation"
    )
    action_spaces = _require_policy_spaces(action_spaces, arg.keys(), "action")
    for k, module in arg.items():
        # remap the observation space which has agents as keys to the policy name
        export_onnx_from_policy(
            module,
            path / f"{k}.onnx",
            observation_spaces[k],
            action_spaces[k],
            normalize_actions=normalize_actions,
            clip_actions=clip_actions,
        )


def _flatten_inputs_to_1d_tensor(
    inputs: dict[str, Any], spaces: dict[str, gym.Space[Any]]
) -> th.Tensor:
    return th.as_tensor(flatten_inputs_to_1d_tensor(inputs, spaces))


def _partial_distribution_kwargs(dist_cls: type) -> dict[str, Any]:
    """Return kwargs captured by ``Distribution.get_partial_dist_cls``."""
    merge = getattr(dist_cls, "_merge_kwargs", None)
    closure = getattr(merge, "__closure__", None)
    if closure is None:
        return {}
    fallback: dict[str, Any] | None = None
    for cell in closure:
        try:
            contents = cell.cell_contents
        except ValueError:
            continue
        if isinstance(contents, dict):
            if "child_distribution_cls_struct" in contents:
                return contents
            fallback = contents
    return fallback or {}


def _eval_action_postprocessing(config: Any) -> tuple[bool, bool]:
    """Return ``(normalize_actions, clip_actions)`` used for evaluation steps.

    Evaluation copies the training config and applies ``evaluation_config``
    overrides. Missing config uses RLlib's defaults: normalize, do not clip.
    """
    if isinstance(config, AlgorithmConfig):
        eval_config = config.get_evaluation_config_object()
        action_config = eval_config if eval_config is not None else config
        return bool(action_config.normalize_actions), bool(action_config.clip_actions)
    return True, False


@export_onnx_from_policy.register
def _(arg: Algorithm, path: pathlib.Path) -> None:
    # arg.spaces["__env_single__"] is a tuple (observation_space, action_space)
    if arg.spaces is None:
        raise RuntimeError("No spaces found on Algorithm while exporting to ONNX")
    elif "__env_single__" not in arg.spaces:
        raise RuntimeError(
            "No single environment spaces found on Algorithm while exporting to ONNX. Expected key '__env_single__'"
        )

    obs_space, act_space = cast(
        tuple[gym.Space[Any], gym.Space[Any]], arg.spaces["__env_single__"]
    )

    if arg.env_runner is not None:
        module = arg.env_runner.module  # type: ignore
    elif arg.env_runner_group is not None:
        module = arg.env_runner_group.foreach_env_runner(
            lambda er: er.module,  # type: ignore
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
    config = arg.get_config()
    if isinstance(module, MultiRLModule) and config is not None:
        assert isinstance(
            obs_space, gym.spaces.Dict
        ), f"Expected MultiAgent Algorithm to have a Dict Observation space with agent_ids as the keys but got {type(obs_space)}"
        assert isinstance(
            act_space, gym.spaces.Dict
        ), f"Expected MultiAgent Algorithm to have a Dict Action space with agent_ids as the keys but got {type(act_space)}"
        policy_mapping_fn: PolicyMappingFn | None = None
        if isinstance(config, AlgorithmConfig):
            # ignore old stack type annotations here
            policy_mapping_fn = cast(PolicyMappingFn | None, config.policy_mapping_fn)
        elif isinstance(config, AlgorithmConfigDict) and "policy_mapping_fn" in config:
            policy_mapping_fn = cast(
                PolicyMappingFn | None, config["policy_mapping_fn"]
            )

        # Can only map if we have a dictionary
        if policy_mapping_fn is not None:
            dummy_episode: EpisodeType = MultiAgentEpisode()
            obs_space = gym.spaces.Dict(
                {policy_mapping_fn(k, dummy_episode): v for k, v in obs_space.items()}
            )
            act_space = gym.spaces.Dict(
                {policy_mapping_fn(k, dummy_episode): v for k, v in act_space.items()}
            )

    normalize_actions, clip_actions = _eval_action_postprocessing(config)
    export_kwargs: dict[str, Any] = {}
    if isinstance(module, (TorchRLModule, MultiRLModule)):
        export_kwargs["normalize_actions"] = normalize_actions
        export_kwargs["clip_actions"] = clip_actions
    export_onnx_from_policy(module, path, obs_space, act_space, **export_kwargs)


class ModelOutputDict(TypedDict):
    action_dist_inputs: th.Tensor
    state_out: NestedDict[str, th.Tensor]


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
    normalize_actions : bool, optional
        Match ``NormalizeAndClipActions`` when evaluation has
        ``normalize_actions=True`` (the RLlib default). Bounded float Box means
        are mapped from ``[-1, 1]`` into ``[low, high]`` and clipped.
    clip_actions : bool, optional
        Match ``NormalizeAndClipActions`` when normalization is off and
        ``clip_actions=True``. Box means are clipped to the space bounds.
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
        normalize_actions: bool = True,
        clip_actions: bool = False,
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
        self.rl_module = cast(TorchRLModule, rl_module.to("cpu"))
        self.rl_module.eval()
        self.normalize_actions = normalize_actions
        self.clip_actions = clip_actions
        self._inference_action_dist_cls = self.rl_module.get_inference_action_dist_cls()
        self._box_output_mode: dict[str, str] = {}
        for key, space in self.action_space.spaces.items():
            if not isinstance(space, Box):
                continue
            mode = self._box_output_mode_for(space, key)
            self._box_output_mode[key] = mode
            if mode != "mean":
                self.register_buffer(f"action_{key}_low", th.as_tensor(space.low))
                self.register_buffer(f"action_{key}_high", th.as_tensor(space.high))

        if observation_space is None and isinstance(
            self.rl_module.observation_space, gym.spaces.Box
        ):
            logger.warning(
                "Potentially flattened space detected in export. This may cause your model to be different than intended when imported into Unreal."
            )
        self.ignored_state_keys = ignored_state_keys

    @cached_property
    def initial_state_dict(self) -> NestedDict[str, th.Tensor]:
        # drop any critic state as we don't need it for onnx export
        return kfilter_dict(
            lambda x: not x.startswith(self.ignored_state_keys),
            self.rl_module.get_initial_state(),
            any_key=False,
        )

    # override the default here because ray outputs extra variance dimensions that we don't need for box spaces
    def get_logit_dimensions(self) -> dict[str, int]:
        custom_flat_dim_func = lambda space: (
            2 * flatdim(space) if isinstance(space, gym.spaces.Box) else flatdim(space)
        )
        return {k: custom_flat_dim_func(v) for k, v in self.action_space.items()}

    def _distribution_class_for_box(self, space_name: str) -> type | None:
        """Return the inference distribution class for one Box action branch."""
        dist_cls = self._inference_action_dist_cls
        # Dict and Tuple spaces use a partial subclass of TorchMultiDistribution.
        # The partial is a class, so ``isinstance(..., TorchMultiDistribution)`` is false.
        if isinstance(dist_cls, type) and issubclass(dist_cls, TorchMultiDistribution):
            struct = _partial_distribution_kwargs(dist_cls).get(
                "child_distribution_cls_struct"
            )
            child = struct.get(space_name) if isinstance(struct, dict) else None
            if not isinstance(child, type):
                logger.warning(
                    "Could not resolve the inference distribution for action '%s'. "
                    "Exporting the distribution mean without environment bound mapping.",
                    space_name,
                )
                return None
            dist_cls = child
        if isinstance(dist_cls, type):
            return dist_cls
        return None

    def _box_output_mode_for(self, space: Box, space_name: str) -> str:
        """Choose how a deterministic Box mean is turned into an env action.

        ``GetActions`` calls ``to_deterministic()`` during evaluation. Both
        diagonal and squashed Gaussians return the raw mean there; ``tanh`` is
        only used while sampling. ``NormalizeAndClipActions`` then unsquashes
        or clips that mean.
        """
        dist_cls = self._distribution_class_for_box(space_name)
        if dist_cls is None or not issubclass(
            dist_cls, (NewTorchSquashedGaussian, NewTorchDiagGaussian)
        ):
            return "mean"
        if self.normalize_actions:
            fully_bounded = bool(
                np.all(space.bounded_below) and np.all(space.bounded_above)
            )
            if not fully_bounded:
                return "mean"
            if space.dtype == np.float32 or space.dtype == np.float64:
                return "unsquash"
            if np.issubdtype(space.dtype, np.integer):
                return "shift"
            return "mean"
        if self.clip_actions and (
            np.any(space.bounded_below) or np.any(space.bounded_above)
        ):
            return "clip"
        return "mean"

    def make_box_output(
        self, logits: th.Tensor, _space_name: str = "action"
    ) -> th.Tensor:
        """Turn box distribution inputs into the action sent to the environment.

        Distribution inputs are ``(mean, log_std)``. Deterministic evaluation
        keeps the mean. With ``normalize_actions``, a bounded float mean is
        mapped from ``[-1, 1]`` with ``low + (mean + 1) * (high - low) / 2``
        and clipped. With only ``clip_actions``, the mean is clipped to the Box.
        """
        mean = logits[: len(logits) // 2]
        mode = self._box_output_mode.get(_space_name, "mean")
        if mode == "mean":
            return mean

        low = self.get_buffer(f"action_{_space_name}_low").to(dtype=mean.dtype)
        high = self.get_buffer(f"action_{_space_name}_high").to(dtype=mean.dtype)
        if mode == "unsquash":
            return th.clamp(low + (mean + 1.0) * (high - low) / 2.0, low, high)
        if mode == "shift":
            return low + mean
        return th.clamp(mean, low, high)

    def forward(self, *args: th.Tensor) -> tuple[th.Tensor, ...]:

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

            module_inputs: dict[str, Any] = {
                "obs": _flatten_inputs_to_1d_tensor(
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
                "obs": _flatten_inputs_to_1d_tensor(
                    tensor_dict, self.observation_space.spaces
                )
            }

        model_out = cast(
            ModelOutputDict, self.rl_module.forward_inference(module_inputs)
        )

        # Convert logits to actual outputs in the respective action spaces
        outputs = self.make_outputs(cast(th.Tensor, model_out["action_dist_inputs"]))
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
        policy: TorchPolicyV2 | TorchPolicy,
        observation_space: gym.Space[Any] | None = None,
        action_space: gym.Space[Any] | None = None,
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
        self.policy_observation_space_is_dict = isinstance(
            policy.observation_space, gym.spaces.Dict
        )
        # for SAC the model.forward is a no-op, so we need to use the action model instead
        if isinstance(policy, SACTorchPolicy):
            # ignore typing here because rllib erases the type of the model
            self._model = cast(TorchModelV2, policy.model.action_model.to("cpu"))  # type: ignore
        else:
            self._model = cast(TorchModelV2, policy.model.to("cpu"))  # type: ignore
        # update the dummy batch here
        self._policy._dummy_batch = (
            self._policy._get_dummy_batch_from_view_requirements(1)
        )
        self._policy._lazy_tensor_dict(self._policy._dummy_batch)
        dist_cls = getattr(policy, "dist_class", None)
        dist_cls = getattr(dist_cls, "func", dist_cls)
        self._squash_box_actions = isinstance(dist_cls, type) and issubclass(
            dist_cls, OldTorchSquashedGaussian
        )
        if self._squash_box_actions:
            for key, space in self.action_space.spaces.items():
                if isinstance(space, Box):
                    self.register_buffer(f"action_{key}_low", th.as_tensor(space.low))
                    self.register_buffer(f"action_{key}_high", th.as_tensor(space.high))

    @cached_property
    def initial_state_dict(self) -> NestedDict[str, th.Tensor]:
        # State is a 2D tensor with shape [seq_len, state_dim]
        return dict(
            map(
                lambda x: (str(x[0]), x[1].unsqueeze(0)),
                enumerate(cast(list[th.Tensor], self._model.get_initial_state())),
            )
        )

    @cached_property
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

    def forward(self, *args: th.Tensor) -> tuple[th.Tensor, ...]:
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
        if self.policy_observation_space_is_dict:
            inputs["obs"] = obs_dict
        else:
            if len(obs_dict) != 1:
                raise ValueError(
                    "A non-dictionary RLlib observation space requires exactly one "
                    "model observation input"
                )
            inputs["obs"] = next(iter(obs_dict.values()))

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
    def get_logit_dimensions(self) -> dict[str, int]:
        custom_flat_dim_func = lambda space: (
            2 * flatdim(space) if isinstance(space, gym.spaces.Box) else flatdim(space)
        )
        return {k: custom_flat_dim_func(v) for k, v in self.action_space.items()}

    def make_box_output(
        self, logits: th.Tensor, _space_name: str = "action"
    ) -> th.Tensor:
        # Distribution inputs are (mean, log_std). Keep the mean.
        mean = logits[: len(logits) // 2]
        if not self._squash_box_actions:
            return mean
        low = self.get_buffer(f"action_{_space_name}_low").to(dtype=mean.dtype)
        high = self.get_buffer(f"action_{_space_name}_high").to(dtype=mean.dtype)
        return low + 0.5 * (th.tanh(mean) + 1.0) * (high - low)


# end of adapted code
