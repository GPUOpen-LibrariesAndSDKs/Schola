# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""ONNX export helpers for LeRobot policies trained or deployed with Schola."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from functools import cached_property
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
import yaml
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.pretrained import PreTrainedPolicy
from schola.core.model import ScholaModel
from schola.core.utils.dict_helpers import NestedDict


def get_schola_lerobot_model(
    policy: PreTrainedPolicy,
    observation_space: gym.Space,
    action_space: gym.Space,
) -> ScholaModel:
    """
    Wrap a LeRobot policy in the matching :class:`ScholaModel` for ONNX export.

    Parameters
    ----------
    policy : lerobot.policies.pretrained.PreTrainedPolicy
        A loaded LeRobot policy (currently only ACT is supported).
    observation_space : gymnasium.Space
        Observation space used as ONNX inputs (from the spaces YAML).
    action_space : gymnasium.Space
        Action space used as ONNX outputs (from the spaces YAML).

    Returns
    -------
    ScholaModel
        Export wrapper for ``policy``.

    Raises
    ------
    TypeError
        If ``policy`` is not a supported LeRobot algorithm.
    """
    if isinstance(policy, ACTPolicy):
        return LeRobotACTScholaModule(policy, observation_space, action_space)
    raise TypeError(
        "Schola ONNX export currently supports only LeRobot ACT "
        f"({ACTPolicy.__name__}), got {type(policy).__name__}."
    )


def get_spaces(spaces_path: str) -> tuple[Dict, Dict]:
    """
    Build observation and action spaces from a nested spaces YAML file.

    Leaves are keyed by their dotted path including the root, so the ONNX
    tensor names are the LeRobot batch keys (``observation.images.wrist``).

    Parameters
    ----------
    spaces_path : str
        Path to a YAML file with ``observation`` and ``action`` mappings.

    Returns
    -------
    tuple of gymnasium.spaces.Dict
        ``(observation_space, action_space)``.
    """
    raw = yaml.safe_load(Path(spaces_path).read_text(encoding="utf-8"))
    return (
        Dict(dict(_iter_leaf_spaces(raw["observation"], "observation"))),
        Dict(dict(_iter_leaf_spaces(raw["action"], "action"))),
    )


def _iter_leaf_spaces(
    node: Mapping[str, Any], path: str
) -> Iterator[tuple[str, gym.Space]]:
    """Yield ``(dotted path, space)`` for every leaf below ``node``."""
    if "type" in node:
        yield path, _leaf_to_space(node, path)
        return
    for key, child in node.items():
        yield from _iter_leaf_spaces(child, f"{path}.{key}")


def _leaf_to_space(leaf: Mapping[str, Any], path: str) -> gym.Space:
    """Build one Gym space from a ``type``/``shape``(/``dtype``) leaf."""
    shape = tuple(leaf["shape"])
    match leaf["type"]:
        case "box":
            dtype = np.dtype(leaf["dtype"])
            # Bounds only need to keep sampling valid; ONNX ignores them.
            low, high = (
                (np.iinfo(dtype).min, np.iinfo(dtype).max)
                if np.issubdtype(dtype, np.integer)
                else (-np.inf, np.inf)
            )
            return Box(low=low, high=high, shape=shape, dtype=dtype.type)
        case "discrete":
            return Discrete(shape[0])
        case "multi_discrete":
            return MultiDiscrete(list(shape))
        case "multi_binary":
            return MultiBinary(list(shape))
        case unsupported:
            raise ValueError(f"{path}: unsupported space type {unsupported!r}")


class LeRobotACTScholaModule(ScholaModel):
    ensemble_weights: th.Tensor
    ensemble_weights_cumsum: th.Tensor

    def __init__(
        self,
        policy: ACTPolicy,
        observation_space: gym.Space,
        action_space: gym.Space,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
        )
        self.policy = policy
        self.policy.eval()
        self.temporal_ensemble_coeff = getattr(
            policy.config, "temporal_ensemble_coeff", None
        )
        if self.temporal_ensemble_coeff is not None:
            chunk_size = int(policy.config.chunk_size)
            if chunk_size < 2:
                raise ValueError(
                    "temporal ensemble requires chunk_size >= 2, " f"got {chunk_size}"
                )
            self.chunk_size = chunk_size
            weights = th.exp(
                -float(self.temporal_ensemble_coeff)
                * th.arange(chunk_size, dtype=th.float32)
            )
            self.register_buffer("ensemble_weights", weights)
            self.register_buffer("ensemble_weights_cumsum", th.cumsum(weights, dim=0))

    @cached_property
    def initial_state_dict(self) -> NestedDict[str, th.Tensor]:
        if self.temporal_ensemble_coeff is None:
            return {}
        # Zeros mean "uninitialized", replacing LeRobot's None buffers.
        tail = self.chunk_size - 1
        return {
            "ensembled_actions": th.zeros(tail, sum(self.flat_dims.values())),
            "ensembled_actions_count": th.zeros(tail, 1, dtype=th.long),
        }

    def _temporal_ensemble_step(
        self,
        chunk: th.Tensor,
        ensembled_actions: th.Tensor,
        ensembled_counts: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Tensor form of LeRobot ``ACTTemporalEnsembler.update``.

        Leftovers are the unconsumed tail ``[B, chunk_size - 1, ...]``.
        A count of 0 is episode start (LeRobot's empty buffers).

        Parameters
        ----------
        chunk : torch.Tensor
            New prediction ``[B, chunk_size, action_dim]``.
        ensembled_actions : torch.Tensor
            Running average for each leftover time.
        ensembled_counts : torch.Tensor
            How many predictions are already in each leftover average.

        Returns
        -------
        action : torch.Tensor
            Command to execute this tick, ``[B, action_dim]``.
        ensembled_actions : torch.Tensor
            Updated leftover averages for next tick,
            ``[B, chunk_size - 1, action_dim]``.
        ensembled_counts : torch.Tensor
            Updated leftover counts for next tick,
            ``[B, chunk_size - 1, 1]``.
        """
        # New chunk vs last tick's leftovers: same times except the farthest step,
        # which has never been predicted before.
        new_overlap = chunk[:, :-1]
        newest_action = chunk[:, -1:]

        # Per leftover slot: how many prior predictions are already in the average.
        n_votes = ensembled_counts.long()
        is_first_step = n_votes == 0
        weight_index = n_votes.clamp(min=1)

        # online weighted mean
        blended = (
            ensembled_actions * self.ensemble_weights_cumsum[weight_index - 1]
            + new_overlap * self.ensemble_weights[weight_index]
        ) / self.ensemble_weights_cumsum[weight_index]

        # First tick: no stored average, so the overlap is the new chunk as-is (counts = 1).
        overlap = th.where(is_first_step, new_overlap, blended)
        overlap_votes = th.where(is_first_step, th.ones_like(n_votes), n_votes + 1)

        # Rebuild a full horizon, then consume index 0 as this tick's action.
        one_vote = th.ones_like(n_votes[:, :1])
        actions = th.cat([overlap, newest_action], dim=1)
        votes = th.cat([overlap_votes, one_vote], dim=1)
        return actions[:, 0], actions[:, 1:], votes[:, 1:]

    def forward(self, *args: th.Tensor) -> tuple[th.Tensor, ...]:
        """
        Pack ONNX observation tensors into a LeRobot batch and predict one action.

        The graph is the policy only. Inputs must already be in the layout
        ``predict_action_chunk`` expects. The output is one command per tick
        (``chunk[:, 0]``, or the ensembled command), matching the YAML action
        Box. With ensemble, leftover buffers are returned as ``state_out_*``.
        """
        args_iter = iter(args)
        batch = {key: next(args_iter) for key in self.input_obs_keys}
        chunk = self.policy.predict_action_chunk(batch)
        if self.temporal_ensemble_coeff is None:
            return (chunk[:, 0],)
        action, ensembled_actions, ensembled_counts = self._temporal_ensemble_step(
            chunk, next(args_iter), next(args_iter)
        )
        return (action, ensembled_actions, ensembled_counts)


def convert_pretrained_to_onnx(
    pretrained_path: str,
    export_path: str,
    spaces_path: str,
) -> None:
    """
    Load a LeRobot pretrained policy and export it to ONNX for Unreal.

    Parameters
    ----------
    pretrained_path : str
        Directory or Hub id loadable with ``from_pretrained``.
    export_path : str
        Destination ``.onnx`` file path.
    spaces_path : str
        YAML file describing observation and action spaces.
    """
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class

    observation_space, action_space = get_spaces(spaces_path)
    config = PreTrainedConfig.from_pretrained(pretrained_path)
    config.device = "cpu"
    policy = get_policy_class(config.type).from_pretrained(
        pretrained_path, config=config
    )
    model = get_schola_lerobot_model(policy, observation_space, action_space)
    model.save_as_onnx(export_path)
