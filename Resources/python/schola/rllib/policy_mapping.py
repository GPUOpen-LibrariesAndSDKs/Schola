# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Agent-to-policy mapping helpers for RLlib train checkpoints and eval.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from schola.rllib.checkpoint import resolve_checkpoint_dir

logger = logging.getLogger(__name__)

SCHOLA_POLICY_MAPPING_FILENAME = "schola_policy_mapping.json"
_POLICY_MAPPING_VERSION = 1


def build_policy_mapping_record(
    *,
    agent_ids: Iterable[str],
    policy_mapping_fn: Callable[..., str],
    policy_mapping_dict: Optional[Dict[str, str]] = None,
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
    if policy_mapping_dict is not None:
        record["policy_mapping_dict"] = {
            str(k): str(v) for k, v in policy_mapping_dict.items()
        }
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
    path = resolve_checkpoint_dir(checkpoint) / SCHOLA_POLICY_MAPPING_FILENAME
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as mapping_file:
        record = json.load(mapping_file)
    if not isinstance(record, dict):
        raise ValueError(
            f"Expected a JSON object in {path}, got {type(record).__name__}."
        )
    return record


def make_policy_mapping_fn_from_dict(
    agent_to_policy: Dict[str, str],
) -> Callable[..., str]:
    """Build an RLlib ``policy_mapping_fn`` from a static agent-to-policy table."""

    def policy_mapping_fn(agent_id: Any, *args: Any, **kwargs: Any) -> str:
        agent_id = str(agent_id)
        if agent_id in agent_to_policy:
            return agent_to_policy[agent_id]
        return agent_id

    return policy_mapping_fn


def validate_agent_to_policy_against_module_ids(
    agent_to_policy: Dict[str, str],
    module_ids: Iterable[str],
) -> None:
    """Validate that every resolved policy id exists in the checkpoint."""
    allowed = set(module_ids)

    for agent_id, policy_id in agent_to_policy.items():
        if policy_id not in allowed:
            raise KeyError(
                f"Agent {agent_id!r} maps to policy {policy_id!r}, which is not among "
                f"the restored module ids {sorted(allowed)}."
            )


def resolve_policy_mapping_for_eval(
    *,
    agent_ids: Iterable[str],
    module_ids: Iterable[str],
    checkpoint: Path,
    env_policy_mapping_fn: Callable[..., str],
    cli_agent_to_policy: Optional[Dict[str, str]] = None,
) -> Tuple[Callable[..., str], Dict[str, str]]:
    """Resolve eval agent-to-policy routing per agent: CLI, checkpoint, then environment.

    For each agent in ``agent_ids``, the first available mapping wins: CLI override,
    checkpoint sidecar, then ``env_policy_mapping_fn``.

    ``agent_ids`` and ``env_policy_mapping_fn`` must come from a prior
    ``discover_env_metadata`` call.

    Returns ``(policy_mapping_fn, agent_to_policy_table)``.
    """
    cli_map = {str(k): str(v) for k, v in (cli_agent_to_policy or {}).items()}

    checkpoint_map: Dict[str, str] = {}
    sidecar = load_policy_mapping_sidecar(checkpoint)
    if sidecar is not None:
        sidecar_table = sidecar.get("agent_to_policy")
        if isinstance(sidecar_table, dict):
            checkpoint_map = {str(k): str(v) for k, v in sidecar_table.items()}

    agent_to_policy: Dict[str, str] = {}
    agent_sources: Dict[str, str] = {}
    for agent_id in agent_ids:
        agent_id = str(agent_id)
        if agent_id in cli_map:
            agent_to_policy[agent_id] = cli_map[agent_id]
            agent_sources[agent_id] = "CLI"
        elif agent_id in checkpoint_map:
            agent_to_policy[agent_id] = checkpoint_map[agent_id]
            agent_sources[agent_id] = "checkpoint"
        else:
            agent_to_policy[agent_id] = str(env_policy_mapping_fn(agent_id))
            agent_sources[agent_id] = "environment"

    logger.info("Resolved agent-to-policy mapping: %s", agent_to_policy)
    logger.debug("Agent-to-policy mapping sources: %s", agent_sources)

    validate_agent_to_policy_against_module_ids(agent_to_policy, module_ids)
    policy_mapping_fn = make_policy_mapping_fn_from_dict(agent_to_policy)
    return policy_mapping_fn, agent_to_policy
