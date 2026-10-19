"""Deterministic shared-backbone, multi-head policy reference model."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SharedPolicyReferenceModel(nn.Module):
    """One shared backbone with named combat and support action heads."""

    OBSERVATION_SIZE = 4
    SHARED_FEATURE_SIZE = 3
    ACTION_SIZE = 2

    def __init__(self) -> None:
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(self.OBSERVATION_SIZE, self.SHARED_FEATURE_SIZE),
            nn.ReLU(),
        )
        self.combat_head = nn.Linear(
            self.SHARED_FEATURE_SIZE,
            self.ACTION_SIZE,
        )
        self.support_head = nn.Linear(
            self.SHARED_FEATURE_SIZE,
            self.ACTION_SIZE,
        )

        self._set_fixed_weights()
        self.requires_grad_(False)

    def _set_fixed_weights(self) -> None:
        """Install known parameters so every output is reproducible."""

        backbone_layer = self.backbone[0]
        if not isinstance(backbone_layer, nn.Linear):
            raise TypeError("the first backbone module must be nn.Linear")
        if backbone_layer.bias is None:
            raise TypeError("the backbone linear layer must have a bias")

        with torch.no_grad():
            backbone_layer.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0, 1.0],
                        [1.0, -1.0, 0.0, 0.0],
                    ],
                    dtype=torch.float32,
                )
            )
            backbone_layer.bias.zero_()

            self.combat_head.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 1.0],
                        [0.0, 1.0, -1.0],
                    ],
                    dtype=torch.float32,
                )
            )
            self.combat_head.bias.copy_(
                torch.tensor([0.5, -0.5], dtype=torch.float32)
            )

            self.support_head.weight.copy_(
                torch.tensor(
                    [
                        [0.5, 0.5, 0.0],
                        [-1.0, 0.0, 1.0],
                    ],
                    dtype=torch.float32,
                )
            )
            self.support_head.bias.copy_(
                torch.tensor([1.0, 2.0], dtype=torch.float32)
            )

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        """Return both action heads for a float32 batch shaped [B, 4]."""

        if observations.ndim != 2:
            raise ValueError(
                "observations must have rank 2 with shape [batch, 4]"
            )
        if observations.shape[1] != self.OBSERVATION_SIZE:
            raise ValueError(
                "observations must have shape [batch, 4]; "
                f"received {tuple(observations.shape)}"
            )
        if observations.dtype != torch.float32:
            raise TypeError(
                "observations must use torch.float32; "
                f"received {observations.dtype}"
            )

        shared_features = self.backbone(observations)
        combat_actions = self.combat_head(shared_features)
        support_actions = self.support_head(shared_features)
        return combat_actions, support_actions
