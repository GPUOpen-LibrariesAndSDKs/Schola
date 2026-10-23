# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Schola ONNX wrapper for LeRobot ACT policies."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import onnx_ir as ir
import torch as th
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor import TransitionKey, create_transition
from lerobot.processor.normalize_processor import (
    NormalizerProcessorStep,
    UnnormalizerProcessorStep,
)
from schola.core.model import ScholaModel, fix_slice_nodes_for_onnx

from ..feature_mapping import is_image_policy_key


def _lerobot_observation_key(model_input_name: str) -> str:
    """Map a framework-neutral model input name to a LeRobot batch key."""
    if model_input_name.startswith("observation."):
        return model_input_name
    return f"observation.{model_input_name}"


def action_chunking_metadata(config: ACTConfig) -> dict[str, str]:
    """
    Describe how each predicted chunk should be played, as ONNX ``metadata_props``.

    The keys mirror Unreal's ``FActionChunkerSettings``. A ``temporal_ensemble_coeff``
    selects the temporal ensemble; otherwise ``n_action_steps`` actions are played
    from each chunk before predicting again.
    """
    if config.temporal_ensemble_coeff is not None:
        return {
            "schola.action_chunking.chunk_mode": "temporal_ensemble",
            "schola.action_chunking.temporal_ensemble_coeff": str(
                config.temporal_ensemble_coeff
            ),
        }
    return {
        "schola.action_chunking.chunk_mode": "queue",
        "schola.action_chunking.num_action_steps": str(config.n_action_steps),
    }


class LeRobotACTScholaModule(ScholaModel):
    """
    Expose an ACT policy through Schola's named ONNX tensor contract.

    The graph is stateless and outputs the whole action chunk
    ``[batch, chunk_size, action_dim]``. Executing the chunk (action queue or
    temporal ensemble) is left to the runtime; the exported model records the
    policy's playback settings in its metadata (see ``action_chunking_metadata``).

    ``uint8`` image inputs are scaled to ``[0, 1]`` float, matching the
    conversion LeRobot policies see during training.
    """

    def __init__(
        self,
        policy: ACTPolicy,
        observation_space: gym.Space,
        action_space: gym.Space,
        normalizer: NormalizerProcessorStep | None = None,
        unnormalizer: UnnormalizerProcessorStep | None = None,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
        )
        self.policy = policy
        self.policy.eval()
        self.normalizer = normalizer
        self.unnormalizer = unnormalizer
        self._uint8_image_keys = frozenset(
            key
            for key in self.input_obs_keys
            if is_image_policy_key(_lerobot_observation_key(key))
            and self.observation_space[key].dtype == np.dtype(np.uint8)
        )

    def _to_policy_input(self, key: str, value: th.Tensor) -> th.Tensor:
        if key in self._uint8_image_keys:
            return value.to(th.float32) / 255.0
        return value

    def _normalize(self, batch: dict[str, th.Tensor]) -> dict[str, th.Tensor]:
        if self.normalizer is None:
            return batch
        transition = self.normalizer(create_transition(observation=batch))
        return transition[TransitionKey.OBSERVATION]

    def _unnormalize_action(self, action: th.Tensor) -> th.Tensor:
        if self.unnormalizer is None:
            return action
        transition = self.unnormalizer(create_transition(action=action))
        return transition[TransitionKey.ACTION]

    def forward(self, *args: th.Tensor) -> tuple[th.Tensor, ...]:
        """Normalize observations, predict an ACT action chunk, and restore action units."""
        batch = {
            _lerobot_observation_key(key): self._to_policy_input(key, value)
            for key, value in zip(self.input_obs_keys, args, strict=True)
        }
        chunk = self.policy.predict_action_chunk(self._normalize(batch))
        return (self._unnormalize_action(chunk),)

    def export_onnx_program(self, onnx_opset: int = 21) -> th.onnx.ONNXProgram:
        """
        Export with a fixed batch size of 1.

        ACT's normalization broadcasts specialize a symbolic batch dimension
        during export, so the base class's dynamic-batch export fails.
        """
        self.eval()
        example_inputs = tuple(
            th.as_tensor(self.observation_space[key].sample()).unsqueeze(0)
            for key in self.input_obs_keys
        )
        with th.no_grad():
            onnx_program = th.onnx.export(
                self,
                args=example_inputs,
                input_names=self.input_obs_keys,
                output_names=self.output_action_keys,
                opset_version=onnx_opset,
                dynamo=True,
                report=False,
                optimize=True,
                verbose=False,
            )
        assert onnx_program is not None
        fix_slice_nodes_for_onnx(onnx_program.model)
        ir.passes.common.shape_inference.infer_shapes(onnx_program.model)
        onnx_program.model.metadata_props.update(
            action_chunking_metadata(self.policy.config)
        )
        return onnx_program
