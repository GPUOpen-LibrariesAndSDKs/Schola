# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Agent-to-policy mapping helpers for RLlib train checkpoints and eval.

Multi-agent training resolves each agent to a policy module at startup. That
mapping must be saved with the checkpoint so eval can route agents to the
correct weights. RLlib only persists extra algorithm state through
``Checkpointable`` subcomponents, so :class:`ScholaPolicyMappingCheckpoint`
wraps the frozen mapping table and plugs it into the standard save/restore flow.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.annotations import override
from ray.rllib.utils.checkpoints import Checkpointable
from ray.rllib.utils.typing import StateDict

from schola.rllib.checkpoint import resolve_checkpoint_dir

logger = logging.getLogger(__name__)

# Subcomponent name used when the mapping is attached to an Algorithm checkpoint
# via ``get_checkpointable_components``. RLlib writes the subcomponent's state
# into ``<algorithm_checkpoint>/<SCHOLA_POLICY_MAPPING_COMPONENT>/``.
SCHOLA_POLICY_MAPPING_COMPONENT = "schola_policy_mapping"

# Key under ``AlgorithmConfig.env_config`` where the training script stashes the
# frozen policy-mapping record so the Algorithm can rebuild the checkpointable
# component on every (possibly remote) worker after config serialization.
ENV_CONFIG_POLICY_MAPPING_RECORD_KEY = "schola_policy_mapping_record"

# State-dict key holding the record inside the Checkpointable's state.
_RECORD_STATE_KEY = "record"


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
        "agent_to_policy": agent_to_policy,
    }
    if module_ids is not None:
        record["module_ids"] = sorted(str(module_id) for module_id in module_ids)
    if policy_mapping_dict is not None:
        record["policy_mapping_dict"] = {
            str(k): str(v) for k, v in policy_mapping_dict.items()
        }
    return record


class ScholaPolicyMappingCheckpoint(Checkpointable):
    """RLlib ``Checkpointable`` holding the frozen agent-to-policy mapping record.

    Attaching this as an Algorithm subcomponent (see
    :func:`schola_algorithm_subclass`) makes the mapping be
    saved and restored using RLlib's own checkpoint machinery: its state lands in
    ``<algorithm_checkpoint>/schola_policy_mapping/`` alongside the standard
    ``learner_group/``, ``env_runner/`` subcomponents.
    """

    def __init__(self, record: Optional[Dict[str, Any]] = None) -> None:
        self._record: Dict[str, Any] = copy.deepcopy(record) if record else {}

    @property
    def record(self) -> Dict[str, Any]:
        """The frozen policy-mapping record (a deep copy)."""
        return copy.deepcopy(self._record)

    @property
    def agent_to_policy(self) -> Dict[str, str]:
        """The ``agent_id -> policy_id`` table extracted from the record."""
        table = self._record.get("agent_to_policy") if self._record else None
        if not isinstance(table, dict):
            return {}
        return {str(k): str(v) for k, v in table.items()}

    def get_state(
        self,
        components: Optional[Union[str, Collection[str]]] = None,
        *,
        not_components: Optional[Union[str, Collection[str]]] = None,
        **kwargs: Any,
    ) -> StateDict:
        # The record is plain JSON data, so this state is both pickle- and
        # msgpack-serializable regardless of the Algorithm's checkpoint format.
        return {_RECORD_STATE_KEY: copy.deepcopy(self._record)}

    def set_state(self, state: StateDict) -> None:
        if state and _RECORD_STATE_KEY in state:
            record = state[_RECORD_STATE_KEY]
            self._record = copy.deepcopy(record) if record else {}

    def get_ctor_args_and_kwargs(self) -> Tuple[Tuple, Dict[str, Any]]:
        # State is fully restored via set_state, so no constructor args are needed.
        return ((), {})


def make_policy_mapping_checkpoint_from_config(
    config: Any,
) -> ScholaPolicyMappingCheckpoint:
    """Build a :class:`ScholaPolicyMappingCheckpoint` from an ``AlgorithmConfig``.

    Reads the frozen record the training script stashed under
    ``config.env_config[ENV_CONFIG_POLICY_MAPPING_RECORD_KEY]``. Returns an empty
    checkpoint when no record is present (for example, when resuming from a
    checkpoint trained before this component existed).
    """
    record: Optional[Dict[str, Any]] = None
    env_config = getattr(config, "env_config", None)
    if isinstance(env_config, dict):
        record = env_config.get(ENV_CONFIG_POLICY_MAPPING_RECORD_KEY)
    return ScholaPolicyMappingCheckpoint(record)


class ScholaPolicyMappingMixin(Checkpointable):
    """Adds the Schola policy-mapping record as a native RLlib checkpoint subcomponent.

    Mixed into an ``Algorithm`` (see :func:`schola_algorithm_subclass`), this exposes
    the frozen agent-to-policy record (stashed in ``config.env_config`` by the training
    script) as a :class:`ScholaPolicyMappingCheckpoint` subcomponent, so RLlib's own
    checkpoint machinery saves and restores it under
    ``<algorithm_checkpoint>/schola_policy_mapping/``.

    Notes
    -----
    This is a cooperative mixin: it must be listed *before* the ``Algorithm`` base in
    the MRO so its ``super()`` calls chain into the real algorithm implementation. It
    relies on the host providing ``self.config`` and ``self._check_component`` (both
    from ``Algorithm``/``Checkpointable``), and is not meant to be instantiated alone.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._schola_policy_mapping = make_policy_mapping_checkpoint_from_config(
            self.config
        )

    @override(Checkpointable)
    def get_state(
        self,
        components: Optional[Union[str, Collection[str]]] = None,
        *,
        not_components: Optional[Union[str, Collection[str]]] = None,
        **kwargs: Any,
    ) -> StateDict:
        state = super().get_state(
            components=components,
            not_components=not_components,
            **kwargs,
        )
        # RLlib's ``save_to_path`` pulls each subcomponent's state from the
        # parent via ``get_state(components=<name>)``, so the component must
        # be represented here for it to be written to disk.
        if self._check_component(
            SCHOLA_POLICY_MAPPING_COMPONENT, components, not_components
        ):
            state[SCHOLA_POLICY_MAPPING_COMPONENT] = (
                self._schola_policy_mapping.get_state()
            )
        return state

    @override(Checkpointable)
    def set_state(self, state: StateDict) -> None:
        super().set_state(state)
        if SCHOLA_POLICY_MAPPING_COMPONENT in state:
            self._schola_policy_mapping.set_state(
                state[SCHOLA_POLICY_MAPPING_COMPONENT]
            )

    @override(Checkpointable)
    def get_checkpointable_components(self) -> List[Tuple[str, Any]]:
        components = super().get_checkpointable_components()
        components.append(
            (SCHOLA_POLICY_MAPPING_COMPONENT, self._schola_policy_mapping)
        )
        return components


def schola_algorithm_subclass(base_algo_class: Type[Algorithm]) -> Type[Algorithm]:
    """Return an ``Algorithm`` subclass that checkpoints Schola policy mapping metadata.

    Composes :class:`ScholaPolicyMappingMixin` in front of ``base_algo_class`` (which
    is only known at runtime, e.g. ``PPO``/``SAC``/``IMPALA``/``APPO``), so the frozen
    policy mapping is saved and restored by RLlib's own checkpoint machinery.
    """
    return type(
        f"Schola{base_algo_class.__name__}",
        (ScholaPolicyMappingMixin, base_algo_class),
        {"__module__": __name__},
    )


def load_policy_mapping_record(checkpoint: Path) -> Optional[Dict[str, Any]]:
    """Load the policy-mapping record from an Algorithm checkpoint, if present.

    Restores the ``schola_policy_mapping`` subcomponent written by RLlib's
    ``Checkpointable`` save. Returns ``None`` when the checkpoint was produced
    without the component (for example, an older or non-Schola checkpoint).
    """
    component_dir = resolve_checkpoint_dir(checkpoint) / SCHOLA_POLICY_MAPPING_COMPONENT
    if not component_dir.is_dir():
        logger.info(
            "Checkpoint %s has no %s component; policy mapping will fall back to "
            "environment or CLI overrides.",
            checkpoint,
            SCHOLA_POLICY_MAPPING_COMPONENT,
        )
        return None
    restored = ScholaPolicyMappingCheckpoint.from_checkpoint(component_dir)
    record = restored.record
    return record or None


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
    the checkpoint's ``schola_policy_mapping`` component, then ``env_policy_mapping_fn``.

    ``agent_ids`` and ``env_policy_mapping_fn`` must come from a prior
    ``discover_env_metadata`` call.

    Returns ``(policy_mapping_fn, agent_to_policy_table)``.
    """
    cli_map = {str(k): str(v) for k, v in (cli_agent_to_policy or {}).items()}

    checkpoint_map: Dict[str, str] = {}
    record = load_policy_mapping_record(checkpoint)
    if record is not None:
        checkpoint_table = record.get("agent_to_policy")
        if isinstance(checkpoint_table, dict):
            checkpoint_map = {str(k): str(v) for k, v in checkpoint_table.items()}

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
