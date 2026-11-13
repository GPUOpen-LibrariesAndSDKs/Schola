# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Schola ONNX wrapper for LeRobot ACT policies."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import onnx_ir as ir
import torch as th
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor import TransitionKey, create_transition
from lerobot.processor.normalize_processor import (
    NormalizerProcessorStep,
    UnnormalizerProcessorStep,
)
from schola.core.model import ScholaModel, fix_slice_nodes_for_onnx

from ..feature_mapping import is_image_policy_key
from .action_processors import make_action_processor


def _lerobot_observation_key(model_input_name: str) -> str:
    """Map a framework-neutral model input name to a LeRobot batch key."""
    if model_input_name.startswith("observation."):
        return model_input_name
    return f"observation.{model_input_name}"


class LeRobotACTScholaModule(ScholaModel):
    """
    Expose an ACT policy through Schola's named ONNX tensor contract.

    The graph is stateless. How the predicted chunks are played is decided here, at export:
    the action processor named by ``action_processor`` (see ``action_processors``) shapes the
    exported outputs and adds a marker output naming itself, and Unreal's ``UNNEPolicy`` creates
    the processor that marker names, with no setting of its own. ``None`` uses the processor
    that is the default for the policy's config.

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
        action_processor: str | None = None,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
        )
        self.policy = policy
        self.policy.eval()
        self.normalizer = normalizer
        self.unnormalizer = unnormalizer
        self.action_processor = make_action_processor(policy.config, action_processor)
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

    @property
    def output_names(self) -> list[str]:
        """Names of the exported outputs, in ``forward`` order."""
        return [
            *self.output_action_keys,
            *self.action_processor.extra_output_names,
            self.action_processor.marker_output_name,
        ]

    def forward(self, *args: th.Tensor) -> tuple[th.Tensor, ...]:
        """Normalize observations, predict an ACT action chunk, and restore action units."""
        batch = {
            _lerobot_observation_key(key): self._to_policy_input(key, value)
            for key, value in zip(self.input_obs_keys, args, strict=True)
        }
        chunk = self._unnormalize_action(
            self.policy.predict_action_chunk(self._normalize(batch))
        )
        outputs = self.action_processor(chunk)
        expected = 1 + len(self.action_processor.extra_output_names)
        if len(outputs) != expected:
            raise ValueError(
                f"{type(self.action_processor).__name__} returned {len(outputs)} outputs, expected "
                f"the action and one per name in extra_output_names, {expected} in total."
            )
        return (*outputs, self.action_processor.marker)

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
                output_names=self.output_names,
                opset_version=onnx_opset,
                dynamo=True,
                report=False,
                optimize=True,
                verbose=False,
            )
        assert onnx_program is not None
        fix_slice_nodes_for_onnx(onnx_program.model)
        ir.passes.common.shape_inference.infer_shapes(onnx_program.model)
        return onnx_program
