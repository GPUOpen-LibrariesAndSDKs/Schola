# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Helpers for locating and warm-starting RLlib checkpoint artifacts."""

from __future__ import annotations

import logging
import pickle
from collections.abc import Collection
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from ray.rllib.core import (
    COMPONENT_LEARNER,
    COMPONENT_LEARNER_GROUP,
    COMPONENT_RL_MODULE,
)
from ray.rllib.utils.checkpoints import Checkpointable

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
    from ray.rllib.core.rl_module.rl_module import RLModule

logger = logging.getLogger(__name__)

ResumeMode = Literal["restore", "warm_start"]

RL_MODULE_COMPONENT = (
    f"{COMPONENT_LEARNER_GROUP}/{COMPONENT_LEARNER}/{COMPONENT_RL_MODULE}"
)


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


def load_multi_rl_module_from_algorithm_checkpoint(
    checkpoint: Path,
) -> "MultiRLModule":
    """Load the ``MultiRLModule`` stored under an Algorithm checkpoint.

    This is the shared restore path for evaluation, ONNX export, and resume
    warm-start. Offline RLlib runs have no EnvRunner, so restoring an entire
    Algorithm solely to read module weights is not supported.
    """
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule

    rl_dir = rl_module_dir_from_algorithm_checkpoint(checkpoint)
    # ``Checkpointable.from_checkpoint`` is annotated to return "Checkpointable"
    # (its base class) rather than ``Self``/``cls``, even though its docstring
    # guarantees "a new instance of the implementing class". Ray's own example
    # call sites (e.g. rllib/examples/ray_serve/classes/cartpole_deployment.py)
    # rely on that same untyped guarantee without narrowing it; we narrow it
    # here because callers need the MultiRLModule-specific interface.
    return cast(MultiRLModule, MultiRLModule.from_checkpoint(rl_dir))


def load_rl_module_from_algorithm_checkpoint(
    checkpoint: Path, module_id: str = "default_policy"
) -> "RLModule":
    """Load one RLModule from an Algorithm checkpoint via the shared restore."""
    loaded = load_multi_rl_module_from_algorithm_checkpoint(checkpoint)
    if module_id not in loaded:
        raise FileNotFoundError(
            f"No RLModule named {module_id!r} in checkpoint "
            f"{resolve_checkpoint_dir(checkpoint)}. Found {sorted(loaded.keys())}."
        )
    return loaded[module_id]


def algorithm_class_from_checkpoint(checkpoint: Path) -> type[Any]:
    """Read the saved Algorithm class from RLlib's constructor pickle."""
    checkpoint_dir = resolve_checkpoint_dir(checkpoint)
    ctor_path = checkpoint_dir / Checkpointable.CLASS_AND_CTOR_ARGS_FILE_NAME
    if not ctor_path.is_file():
        raise FileNotFoundError(
            f"No {Checkpointable.CLASS_AND_CTOR_ARGS_FILE_NAME} in {checkpoint_dir}. "
            "Pass an RLlib Algorithm checkpoint directory."
        )
    with ctor_path.open("rb") as ctor_file:
        ctor_info = pickle.load(ctor_file)
    saved_cls = ctor_info.get("class") if isinstance(ctor_info, dict) else None
    if not isinstance(saved_cls, type):
        raise ValueError(
            f"Checkpoint {checkpoint_dir} did not record an Algorithm class."
        )
    return saved_cls


def algorithm_family(cls: type[Any]) -> str:
    """Return the RLlib algorithm name (``PPO``, ``BC``, …), ignoring Schola wrappers."""
    from ray.rllib.algorithms.algorithm import Algorithm

    for candidate in cls.__mro__:
        if candidate is Algorithm or candidate is object:
            continue
        if not isinstance(candidate, type) or not issubclass(candidate, Algorithm):
            continue
        if candidate.__module__.startswith("ray.rllib.algorithms"):
            return candidate.__name__
    return cls.__name__


def resume_mode_for_checkpoint(
    checkpoint: Path, target_algo_class: type[Any]
) -> ResumeMode:
    """Choose Tune restore vs module-only warm start for ``--resume-from``."""
    saved_family = algorithm_family(algorithm_class_from_checkpoint(checkpoint))
    target_family = algorithm_family(target_algo_class)
    if saved_family == target_family:
        return "restore"
    return "warm_start"


def plan_resume_from_checkpoint(
    checkpoint: Path,
    target_algo_class: type[Any],
    target_config: "AlgorithmConfig",
    policy_ids: Collection[str],
) -> tuple[Path | None, Path | None]:
    """Return ``(restore_dir, warm_start_dir)`` for an online ``--resume-from`` path."""
    checkpoint_dir = resolve_checkpoint_dir(checkpoint)
    if resume_mode_for_checkpoint(checkpoint_dir, target_algo_class) == "restore":
        return checkpoint_dir, None
    loaded = load_multi_rl_module_from_algorithm_checkpoint(checkpoint_dir)
    assert_warm_start_compatible(loaded, target_config, policy_ids)
    return None, checkpoint_dir


def _target_module_class(config: "AlgorithmConfig") -> type[Any]:
    spec = config.get_default_rl_module_spec()
    module_class = getattr(spec, "module_class", None)
    if module_class is None:
        raise ValueError("Target RLlib config did not define a default RLModule class.")
    return module_class


def assert_warm_start_compatible(
    loaded_module: "MultiRLModule",
    target_config: "AlgorithmConfig",
    policy_ids: Collection[str],
) -> None:
    """Fail if the checkpoint's modules cannot initialize the target algorithm."""
    target_module_cls = _target_module_class(target_config)
    target_family = algorithm_family(target_config.algo_class)
    unknown_ids = set(loaded_module.keys()) - set(policy_ids)
    if unknown_ids:
        raise ValueError(
            "Warm-start checkpoint has RLModule ids "
            f"{sorted(unknown_ids)} that are not among the live policies "
            f"{sorted(policy_ids)}. Single-agent imitation checkpoints use "
            "'default_policy'."
        )
    for module_id, module in loaded_module.items():
        source_cls = type(module)
        if source_cls is target_module_cls:
            continue
        if _is_bc_to_ppo_style(source_cls, target_module_cls):
            continue
        raise ValueError(
            f"Cannot warm-start {target_family} ({target_module_cls.__name__}) "
            f"from checkpoint module {module_id!r} ({source_cls.__name__}). "
            "BC and MARWIL can continue into PPO or IMPALA when the network "
            "layout matches; SAC and APPO need a same-algorithm checkpoint."
        )


def _is_bc_to_ppo_style(source_cls: type[Any], target_cls: type[Any]) -> bool:
    return (
        source_cls.__name__ == "DefaultBCTorchRLModule"
        and target_cls.__name__ == "DefaultPPOTorchRLModule"
    )


def _copy_bc_weights_into_ppo_module(
    source: "RLModule", target: "RLModule"
) -> None:
    """Copy BC encoder/policy heads into a PPO-style module; leave the critic random."""
    encoder = getattr(source, "_encoder", None)
    pi_head = getattr(source, "_pi_head", None)
    if encoder is None or pi_head is None:
        raise ValueError(
            f"{type(source).__name__} does not expose _encoder/_pi_head for warm start."
        )
    target_encoder = getattr(target, "encoder", None)
    target_pi = getattr(target, "pi", None)
    if target_encoder is None or target_pi is None:
        raise ValueError(
            f"{type(target).__name__} does not expose encoder/pi for warm start."
        )
    try:
        if hasattr(target_encoder, "encoder"):
            target_encoder.encoder.load_state_dict(encoder.state_dict())
        elif hasattr(target_encoder, "actor_encoder"):
            target_encoder.actor_encoder.load_state_dict(encoder.state_dict())
        else:
            raise ValueError(
                f"{type(target_encoder).__name__} has no encoder or actor_encoder."
            )
        target_pi.load_state_dict(pi_head.state_dict())
    except RuntimeError as exc:
        raise ValueError(
            "Warm-start weight copy failed because the BC and online networks "
            "do not match. Use the same --fcnet-hiddens, --activation, and LSTM "
            f"flags as the imitation run. Original error: {exc}"
        ) from exc


def _copy_compatible_weights(source: "RLModule", target: "RLModule") -> None:
    if type(source) is type(target):
        target.set_state(source.get_state())
        return
    if _is_bc_to_ppo_style(type(source), type(target)):
        _copy_bc_weights_into_ppo_module(source, target)
        return
    raise ValueError(
        f"No weight-copy path from {type(source).__name__} to {type(target).__name__}."
    )


def _copy_warm_start_weights_on_learner(
    learner: Any, *, checkpoint_dir: str, **kwargs: Any
) -> None:
    modules = load_multi_rl_module_from_algorithm_checkpoint(Path(checkpoint_dir))
    for module_id in modules.keys():
        _copy_compatible_weights(modules[module_id], learner.module[module_id])


def _sync_learner_weights_to_env_runners(algorithm: "Algorithm") -> None:
    if algorithm.env_runner_group is None or algorithm.learner_group is None:
        return
    algorithm.env_runner_group.sync_weights(
        from_worker_or_learner_group=algorithm.learner_group
    )


def warm_start_algorithm_from_checkpoint(
    algorithm: "Algorithm",
    checkpoint: Path,
    policy_ids: Collection[str],
) -> None:
    """Load checkpoint module weights into a freshly constructed Algorithm."""
    loaded = load_multi_rl_module_from_algorithm_checkpoint(checkpoint)
    assert_warm_start_compatible(loaded, algorithm.config, policy_ids)
    target_module_cls = _target_module_class(algorithm.config)
    same_class = all(
        type(loaded[module_id]) is target_module_cls for module_id in loaded.keys()
    )
    checkpoint_dir = resolve_checkpoint_dir(checkpoint)
    if same_class:
        algorithm.restore_from_path(
            str(checkpoint_dir),
            component=RL_MODULE_COMPONENT,
        )
    else:

        algorithm.learner_group.foreach_learner(
            _copy_warm_start_weights_on_learner,
            checkpoint_dir=str(checkpoint_dir),
        )
    _sync_learner_weights_to_env_runners(algorithm)
    logger.info(
        "Warm-started %s from %s. Optimizer and lifetime step counts start at zero; "
        "the critic is not copied from behaviour cloning.",
        algorithm_family(type(algorithm)),
        resolve_checkpoint_dir(checkpoint),
    )


def make_warm_start_callback(checkpoint: Path, policy_ids: Collection[str]):
    """RLlib ``on_algorithm_init`` hook that loads module weights from *checkpoint*."""
    checkpoint_dir = resolve_checkpoint_dir(checkpoint)
    frozen_ids = tuple(policy_ids)

    def on_algorithm_init(*, algorithm: "Algorithm", **kwargs: Any) -> None:
        warm_start_algorithm_from_checkpoint(algorithm, checkpoint_dir, frozen_ids)

    return on_algorithm_init
