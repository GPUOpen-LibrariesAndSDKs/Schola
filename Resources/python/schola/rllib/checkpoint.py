# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Helpers for locating RLlib checkpoint artifacts on disk.
"""

from pathlib import Path
from typing import TYPE_CHECKING, cast

from ray.rllib.core import (
    COMPONENT_LEARNER,
    COMPONENT_LEARNER_GROUP,
    COMPONENT_RL_MODULE,
)

if TYPE_CHECKING:
    from ray.rllib.core.rl_module.rl_module import RLModule


def resolve_checkpoint_dir(checkpoint: Path) -> Path:
    """Normalize a checkpoint file or directory path to the checkpoint directory."""
    checkpoint = Path(checkpoint)
    return checkpoint if checkpoint.is_dir() else checkpoint.parent


def rl_module_dir_from_algorithm_checkpoint(checkpoint: Path) -> Path:
    """Return the on-disk ``MultiRLModule`` root under an Algorithm checkpoint."""
    checkpoint = resolve_checkpoint_dir(checkpoint)
    primary = (
        checkpoint / COMPONENT_LEARNER_GROUP / COMPONENT_LEARNER / COMPONENT_RL_MODULE
    )
    if primary.is_dir():
        return primary
    legacy = checkpoint / "learner" / COMPONENT_RL_MODULE
    if legacy.is_dir():
        return legacy
    raise FileNotFoundError(
        f"No RLModule checkpoint directory found under {checkpoint}. Expected "
        + "`learner_group/learner/rl_module/` (new API stack checkpoint)."
    )


def load_rl_module_from_algorithm_checkpoint(
    checkpoint: Path, module_id: str = "default_policy"
) -> "RLModule":
    """Load one RLModule directly from an Algorithm checkpoint.

    Offline RLlib runs intentionally have no EnvRunner, so restoring an entire
    Algorithm solely to export its policy is not supported. This helper shares
    checkpoint-layout handling with evaluation while avoiding that reconstruction.
    """
    from ray.rllib.core.rl_module.rl_module import RLModule

    module_path = rl_module_dir_from_algorithm_checkpoint(checkpoint) / module_id
    if not module_path.is_dir():
        raise FileNotFoundError(
            f"No RLModule found at {module_path}. Expected a module named "
            + f"{module_id!r} below the RLModule checkpoint directory."
        )
    # ``Checkpointable.from_checkpoint`` is annotated to return "Checkpointable"
    # (its base class) rather than ``Self``/``cls``, even though its docstring
    # guarantees "a new instance of the implementing class". Ray's own example
    # call sites (e.g. rllib/examples/ray_serve/classes/cartpole_deployment.py)
    # rely on that same untyped guarantee without narrowing it; we narrow it
    # here because callers need the RLModule-specific interface.
    return cast(RLModule, RLModule.from_checkpoint(module_path))
