# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Action processors: how Unreal's ``UNNEPolicy`` plays the action chunks an exported ACT policy predicts.

Each processor shapes the exported outputs and adds a marker output named
``action_processor.<unreal_struct>``, which tells ``UNNEPolicy`` which C++ processor to create.

To add a processor, subclass ``ActionProcessor``, set its ``name``, its ``unreal_struct`` and any
``extra_output_names``, implement ``forward`` to turn the chunk into those outputs, and decorate
the class with ``register_action_processor``. Its ``name`` then works with ``--action-processor``.
Unreal needs a C++ struct derived from ``FActionProcessor`` named ``F<unreal_struct>``, see
``ActionProcessor.h``.
"""

from __future__ import annotations

from typing import ClassVar, TypeVar

import numpy as np
import torch as th
from lerobot.policies.act.configuration_act import ACTConfig

MARKER_OUTPUT_PREFIX = "action_processor."
"""Prefix of the output whose name tells ``UNNEPolicy`` which processor plays the chunks."""


class ActionProcessor(th.nn.Module):
    """
    Turns a predicted action chunk into the exported outputs.

    ``forward`` takes the chunk ``[batch, chunk_size, action_dim]`` and returns the action
    output followed by one tensor per name in ``extra_output_names``. Constants it returns,
    such as the temporal ensemble's coefficient, should be buffers so they are exported with the graph.

    The exported model also returns ``marker`` as ``marker_output_name``. Its value is unused;
    the name tells ``UNNEPolicy`` to play the chunks with the C++ struct ``unreal_struct``.
    """

    name: ClassVar[str]
    """Name that selects this processor on the ``lerobot-to-onnx`` command line."""

    unreal_struct: ClassVar[str]
    """C++ struct that plays this processor's outputs in Unreal, without its ``F``, e.g. ``ACTActionProcessor``."""

    extra_output_names: ClassVar[tuple[str, ...]] = ()
    """Names of the outputs ``forward`` returns after the action output."""

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.register_buffer("marker", th.zeros(1))

    @property
    def marker_output_name(self) -> str:
        """Name of the output that names this processor's C++ struct for ``UNNEPolicy``."""
        return f"{MARKER_OUTPUT_PREFIX}{self.unreal_struct}"

    @classmethod
    def is_default_for(cls, config: ACTConfig) -> bool:
        """Return whether this processor is used for the policy's config when none is named."""
        return False


_ActionProcessorType = TypeVar("_ActionProcessorType", bound=type[ActionProcessor])

ACTION_PROCESSORS: dict[str, type[ActionProcessor]] = {}
"""Every registered processor, by name."""


def register_action_processor(cls: _ActionProcessorType) -> _ActionProcessorType:
    """Class decorator adding a processor to ``ACTION_PROCESSORS``."""
    if cls.name in ACTION_PROCESSORS:
        raise ValueError(f"An action processor named {cls.name!r} is already registered.")
    ACTION_PROCESSORS[cls.name] = cls
    return cls


def make_action_processor(config: ACTConfig, name: str | None = None) -> ActionProcessor:
    """
    Create the processor called ``name``, or the one the policy's config uses if ``name`` is ``None``.

    Raises
    ------
    ValueError
        If no processor is called ``name``, or ``name`` is ``None`` and not exactly one
        processor matches the config.
    """
    if name is None:
        defaults = [cls for cls in ACTION_PROCESSORS.values() if cls.is_default_for(config)]
        if len(defaults) != 1:
            raise ValueError(
                f"Expected one action processor to be the default for the policy's config, got "
                f"{sorted(cls.name for cls in defaults)}. Name one of "
                f"{sorted(ACTION_PROCESSORS)}."
            )
        return defaults[0](config)
    if name not in ACTION_PROCESSORS:
        raise ValueError(
            f"Unknown action processor {name!r}, expected one of {sorted(ACTION_PROCESSORS)}."
        )
    return ACTION_PROCESSORS[name](config)


@register_action_processor
class ACTProcessor(ActionProcessor):
    """
    Plays the first ``n_action_steps`` actions of each chunk in order, like LeRobot ACT's action queue.

    The action output is ``[batch, n_action_steps, action_dim]``, which ``UNNEPolicy`` plays
    whole before predicting again.
    """

    name = "act"
    unreal_struct = "ACTActionProcessor"

    def __init__(self, config: ACTConfig):
        super().__init__(config)
        self.n_action_steps = config.n_action_steps

    @classmethod
    def is_default_for(cls, config: ACTConfig) -> bool:
        return config.temporal_ensemble_coeff is None

    def forward(self, chunk: th.Tensor) -> tuple[th.Tensor, ...]:
        return (chunk[:, : self.n_action_steps],)


@register_action_processor
class TemporalEnsembleProcessor(ActionProcessor):
    """
    Averages every prediction for a step, like LeRobot ACT's temporal ensemble.

    The action output is the whole chunk ``[batch, chunk_size, action_dim]``, and the
    ``temporal_ensemble_coeff`` output holds the policy's coefficient ``k`` as one float.
    ``UNNEPolicy`` weights the ``i``-th prediction for a step, counting from the oldest,
    ``exp(-k * i)``, so ``k`` must keep every weight over the chunk finite.
    """

    name = "temporal_ensemble"
    unreal_struct = "TemporalEnsembleActionProcessor"
    extra_output_names = ("temporal_ensemble_coeff",)

    def __init__(self, config: ACTConfig):
        super().__init__(config)
        if config.temporal_ensemble_coeff is None:
            raise ValueError(
                "The temporal_ensemble action processor needs the policy's "
                "temporal_ensemble_coeff, which is not set."
            )
        coeff = config.temporal_ensemble_coeff
        # A negative coefficient weights the newest prediction most, up to exp(-k * (chunk_size - 1))
        largest_weight = np.exp(np.float32(-coeff) * np.float32(config.chunk_size - 1))
        if not np.isfinite(np.float32(coeff)) or not np.isfinite(largest_weight):
            raise ValueError(
                f"temporal_ensemble_coeff is {coeff}, which does not give finite weights over "
                f"a chunk of {config.chunk_size}."
            )
        self.register_buffer(
            "coeff", th.tensor([config.temporal_ensemble_coeff], dtype=th.float32)
        )

    @classmethod
    def is_default_for(cls, config: ACTConfig) -> bool:
        return config.temporal_ensemble_coeff is not None

    def forward(self, chunk: th.Tensor) -> tuple[th.Tensor, ...]:
        return chunk, self.coeff
