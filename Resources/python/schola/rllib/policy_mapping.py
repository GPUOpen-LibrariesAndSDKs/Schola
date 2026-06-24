# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Agent-to-policy mapping helpers for RLlib train checkpoints and eval.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from schola.rllib.checkpoint import algorithm_checkpoint_dir

logger = logging.getLogger(__name__)

SCHOLA_POLICY_MAPPING_FILENAME = "schola_policy_mapping.json"
_POLICY_MAPPING_VERSION = 1


def build_policy_mapping_record(
    *,
    agent_ids: Iterable[str],
    policy_mapping_fn: Callable[..., str],
    agent_types: Optional[Dict[str, str]] = None,
    module_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Freeze discovery output as JSON-serializable checkpoint metadata."""
    agent_to_policy = {
        str(agent_id): str(policy_mapping_fn(agent_id)) for agent_id in agent_ids
    }
    record: Dict[str, Any] = {
        "version": _POLICY_MAPPING_VERSION,
        "agent_to_policy": agent_to_policy,
    }
    if module_ids is not None:
        record["module_ids"] = sorted(str(module_id) for module_id in module_ids)
    if agent_types is not None:
        record["agent_types"] = {str(k): str(v) for k, v in agent_types.items()}
    return record


def write_policy_mapping_sidecar(checkpoint_dir: Path, record: Dict[str, Any]) -> Path:
    """Write ``schola_policy_mapping.json`` into an Algorithm checkpoint directory."""
    path = Path(checkpoint_dir) / SCHOLA_POLICY_MAPPING_FILENAME
    with path.open("w", encoding="utf-8") as mapping_file:
        json.dump(record, mapping_file, indent=2, sort_keys=True)
        mapping_file.write("\n")
    return path


def load_policy_mapping_sidecar(checkpoint: Path) -> Optional[Dict[str, Any]]:
    """Load ``schola_policy_mapping.json`` from a checkpoint, if present."""
    path = algorithm_checkpoint_dir(checkpoint) / SCHOLA_POLICY_MAPPING_FILENAME
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as mapping_file:
        record = json.load(mapping_file)
    if not isinstance(record, dict):
        raise ValueError(
            f"Expected a JSON object in {path}, got {type(record).__name__}."
        )
    return record


def policy_mapping_fn_from_agent_to_policy(
    agent_to_policy: Dict[str, str],
) -> Callable[..., str]:
    """Build an RLlib ``policy_mapping_fn`` from a static agent-to-policy table."""

    def policy_mapping_fn(agent_id: Any, *args: Any, **kwargs: Any) -> str:
        agent_id = str(agent_id)
        if agent_id in agent_to_policy:
            return agent_to_policy[agent_id]
        return agent_id

    return policy_mapping_fn


def validate_policy_mapping_against_module_ids(
    policy_mapping_fn: Callable[..., str],
    module_ids: Iterable[str],
) -> Callable[..., str]:
    """Wrap ``policy_mapping_fn`` so every routed policy id exists in the checkpoint."""
    allowed = set(module_ids)

    def validated_policy_mapping_fn(agent_id: Any, *args: Any, **kwargs: Any) -> str:
        policy_id = policy_mapping_fn(agent_id, *args, **kwargs)
        if policy_id not in allowed:
            raise KeyError(
                f"Agent {str(agent_id)!r} maps to policy {policy_id!r}, which is not among "
                f"the restored module ids {sorted(allowed)}."
            )
        return policy_id

    return validated_policy_mapping_fn


def log_env_mapping_disagreements(
    *,
    authoritative_agent_to_policy: Dict[str, str],
    agent_ids: Iterable[str],
    env_policy_mapping_fn: Callable[..., str],
    source: str,
) -> None:
    """Log when the live env would route an agent differently than the chosen map."""
    for agent_id in agent_ids:
        agent_id = str(agent_id)
        chosen_policy = authoritative_agent_to_policy.get(agent_id)
        if chosen_policy is None:
            continue
        env_policy = str(env_policy_mapping_fn(agent_id))
        if env_policy != chosen_policy:
            logger.warning(
                "Agent %r: using policy %r from %s, but the live environment would "
                "map it to %r.",
                agent_id,
                chosen_policy,
                source,
                env_policy,
            )


def resolve_policy_mapping_for_eval(
    *,
    cli_agent_to_policy: Optional[Dict[str, str]],
    checkpoint: Path,
    module_ids: Iterable[str],
    agent_ids: Iterable[str],
    env_policy_mapping_fn: Callable[..., str],
) -> Tuple[Callable[..., str], str, Dict[str, str]]:
    """Resolve eval agent-to-policy routing: CLI, checkpoint sidecar, then environment.

    ``agent_ids`` and ``env_policy_mapping_fn`` must come from a prior
    ``discover_env_metadata`` call. When the source is CLI or checkpoint, logs
    warnings if the live env would route an agent differently.

    Returns ``(policy_mapping_fn, source, agent_to_policy_table)``.
    """
    module_id_list = list(module_ids)

    if cli_agent_to_policy:
        chosen_table = {str(k): str(v) for k, v in cli_agent_to_policy.items()}
        source = "CLI"
    else:
        sidecar = load_policy_mapping_sidecar(checkpoint)
        sidecar_table = (
            sidecar.get("agent_to_policy") if isinstance(sidecar, dict) else None
        )
        if sidecar_table:
            chosen_table = {str(k): str(v) for k, v in sidecar_table.items()}
            source = "checkpoint"
        else:
            source = "environment"
            chosen_table = {
                str(agent_id): str(env_policy_mapping_fn(agent_id))
                for agent_id in agent_ids
            }

    if source != "environment":
        log_env_mapping_disagreements(
            authoritative_agent_to_policy=chosen_table,
            agent_ids=agent_ids,
            env_policy_mapping_fn=env_policy_mapping_fn,
            source=source,
        )

    base_fn = (
        env_policy_mapping_fn
        if source == "environment"
        else policy_mapping_fn_from_agent_to_policy(chosen_table)
    )
    policy_mapping_fn = validate_policy_mapping_against_module_ids(
        base_fn, module_id_list
    )
    return policy_mapping_fn, source, chosen_table
