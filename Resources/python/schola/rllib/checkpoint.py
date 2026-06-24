# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Helpers for locating RLlib checkpoint artifacts on disk.
"""

from pathlib import Path


def algorithm_checkpoint_dir(checkpoint: Path) -> Path:
    """Normalize a checkpoint file or directory path to the checkpoint directory."""
    checkpoint = Path(checkpoint)
    return checkpoint if checkpoint.is_dir() else checkpoint.parent


def rl_module_dir_from_algorithm_checkpoint(checkpoint: Path) -> Path:
    """Return the on-disk ``MultiRLModule`` root under an Algorithm checkpoint."""
    from ray.rllib.core import (
        COMPONENT_LEARNER,
        COMPONENT_LEARNER_GROUP,
        COMPONENT_RL_MODULE,
    )

    checkpoint = algorithm_checkpoint_dir(checkpoint)
    primary = (
        checkpoint / COMPONENT_LEARNER_GROUP / COMPONENT_LEARNER / COMPONENT_RL_MODULE
    )
    if primary.is_dir():
        return primary
    legacy = checkpoint / "learner" / COMPONENT_RL_MODULE
    if legacy.is_dir():
        return legacy
    raise FileNotFoundError(
        f"No RLModule checkpoint directory found under {checkpoint}. "
        "Expected `learner_group/learner/rl_module/` (new API stack checkpoint)."
    )
