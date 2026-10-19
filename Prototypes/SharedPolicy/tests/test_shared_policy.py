"""Tests for the deterministic PyTorch shared-policy model."""

import torch

from shared_policy import SharedPolicyReferenceModel


def test_single_observation_returns_both_action_heads() -> None:
    model = SharedPolicyReferenceModel()
    observations = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0]],
        dtype=torch.float32,
    )

    combat_actions, support_actions = model(observations)

    assert combat_actions.shape == (1, 2)
    assert support_actions.shape == (1, 2)
    torch.testing.assert_close(
        combat_actions,
        torch.tensor([[4.5, 5.5]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        support_actions,
        torch.tensor([[6.0, -2.0]], dtype=torch.float32),
    )


def test_three_observations_preserve_batch_order() -> None:
    model = SharedPolicyReferenceModel()
    observations = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 1.0, 2.0, 0.0],
            [-1.0, 2.0, 1.0, -3.0],
        ],
        dtype=torch.float32,
    )

    combat_actions, support_actions = model(observations)

    assert combat_actions.shape == (3, 2)
    assert support_actions.shape == (3, 2)
    torch.testing.assert_close(
        combat_actions,
        torch.tensor(
            [
                [4.5, 5.5],
                [9.5, -2.5],
                [0.5, -0.5],
            ],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        support_actions,
        torch.tensor(
            [
                [6.0, -2.0],
                [4.5, -1.0],
                [1.0, 2.0],
            ],
            dtype=torch.float32,
        ),
    )


def test_reference_parameters_are_frozen() -> None:
    model = SharedPolicyReferenceModel()

    assert all(not parameter.requires_grad for parameter in model.parameters())
