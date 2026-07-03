# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Tests for Schola RLlib policy mapping helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

import pytest

from schola.rllib.policy_mapping import (
    ENV_CONFIG_POLICY_MAPPING_RECORD_KEY,
    SCHOLA_POLICY_MAPPING_COMPONENT,
    ScholaPolicyMappingCheckpoint,
    build_policy_mapping_record,
    load_policy_mapping_record,
    make_policy_mapping_checkpoint_from_config,
    resolve_policy_mapping_for_eval,
    validate_agent_to_policy_against_module_ids,
)


@pytest.fixture
def env_policy_fn():
    return lambda agent_id, *args, **kwargs: "EnvPolicy"


@pytest.fixture
def checkpoint_with_mapping(tmp_path: Path) -> Callable[..., Path]:
    """Write the ``schola_policy_mapping`` component under ``tmp_path`` and return it."""

    def _write(
        agent_to_policy: Dict[str, str],
        *,
        module_ids: Optional[List[str]] = None,
    ) -> Path:
        if module_ids is None:
            module_ids = sorted(set(agent_to_policy.values()))
        record = build_policy_mapping_record(
            agent_ids=list(agent_to_policy),
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_to_policy[
                str(agent_id)
            ],
            module_ids=module_ids,
        )
        ScholaPolicyMappingCheckpoint(record).save_to_path(
            tmp_path / SCHOLA_POLICY_MAPPING_COMPONENT
        )
        return tmp_path

    return _write


def test_checkpoint_component_roundtrips_record(tmp_path):
    record = build_policy_mapping_record(
        agent_ids=["agent_0", "agent_1"],
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "Pawn",
        policy_mapping_dict={"agent_0": "Pawn", "agent_1": "Pawn"},
        module_ids=["Pawn"],
    )
    component_dir = tmp_path / SCHOLA_POLICY_MAPPING_COMPONENT
    ScholaPolicyMappingCheckpoint(record).save_to_path(component_dir)

    restored = ScholaPolicyMappingCheckpoint.from_checkpoint(component_dir)
    assert restored.record == record
    assert restored.agent_to_policy == {"agent_0": "Pawn", "agent_1": "Pawn"}


def test_load_policy_mapping_record_returns_none_without_component(tmp_path):
    assert load_policy_mapping_record(tmp_path) is None


def test_load_policy_mapping_record_reads_saved_component(checkpoint_with_mapping):
    checkpoint = checkpoint_with_mapping({"agent_0": "Pawn"})
    record = load_policy_mapping_record(checkpoint)
    assert record is not None
    assert record["agent_to_policy"] == {"agent_0": "Pawn"}


def test_make_policy_mapping_checkpoint_from_config_reads_env_config():
    record = build_policy_mapping_record(
        agent_ids=["agent_0"],
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "Pawn",
    )

    class _Config:
        env_config = {ENV_CONFIG_POLICY_MAPPING_RECORD_KEY: record}

    component = make_policy_mapping_checkpoint_from_config(_Config())
    assert component.record == record


def test_make_policy_mapping_checkpoint_from_config_defaults_to_empty():
    class _Config:
        env_config = {}

    component = make_policy_mapping_checkpoint_from_config(_Config())
    assert component.record == {}
    assert component.agent_to_policy == {}


def test_validate_agent_to_policy_against_module_ids_raises():
    with pytest.raises(KeyError, match="Ghost"):
        validate_agent_to_policy_against_module_ids(
            {"agent_0": "Ghost"},
            ["Pawn"],
        )


def test_resolve_policy_mapping_prefers_cli_over_checkpoint_and_env(
    checkpoint_with_mapping, env_policy_fn
):
    checkpoint = checkpoint_with_mapping({"agent_0": "CheckpointPolicy"})

    fn, table = resolve_policy_mapping_for_eval(
        agent_ids=["agent_0"],
        module_ids=["CliPolicy"],
        checkpoint=checkpoint,
        env_policy_mapping_fn=env_policy_fn,
        cli_agent_to_policy={"agent_0": "CliPolicy"},
    )

    assert table == {"agent_0": "CliPolicy"}
    assert fn("agent_0") == "CliPolicy"


def test_resolve_policy_mapping_uses_checkpoint_when_cli_missing(
    checkpoint_with_mapping, env_policy_fn
):
    checkpoint = checkpoint_with_mapping({"agent_0": "CheckpointPolicy"})

    fn, table = resolve_policy_mapping_for_eval(
        agent_ids=["agent_0"],
        module_ids=["CheckpointPolicy"],
        checkpoint=checkpoint,
        env_policy_mapping_fn=env_policy_fn,
    )

    assert table == {"agent_0": "CheckpointPolicy"}
    assert fn("agent_0") == "CheckpointPolicy"


def test_resolve_policy_mapping_falls_back_to_environment(tmp_path):
    env_fn = lambda agent_id, *args, **kwargs: "Pawn"

    fn, table = resolve_policy_mapping_for_eval(
        agent_ids=["agent_0"],
        module_ids=["Pawn"],
        checkpoint=tmp_path,
        env_policy_mapping_fn=env_fn,
    )

    assert table == {"agent_0": "Pawn"}
    assert fn("agent_0") == "Pawn"


def test_resolve_policy_mapping_merges_per_agent_sources(
    checkpoint_with_mapping, env_policy_fn
):
    checkpoint = checkpoint_with_mapping(
        {"agent_0": "CheckpointPolicy", "agent_1": "CheckpointPolicy"},
        module_ids=["CheckpointPolicy", "CliPolicy", "EnvPolicy"],
    )

    fn, table = resolve_policy_mapping_for_eval(
        agent_ids=["agent_0", "agent_1", "agent_2"],
        module_ids=["CheckpointPolicy", "CliPolicy", "EnvPolicy"],
        checkpoint=checkpoint,
        env_policy_mapping_fn=env_policy_fn,
        cli_agent_to_policy={"agent_0": "CliPolicy"},
    )

    assert table == {
        "agent_0": "CliPolicy",
        "agent_1": "CheckpointPolicy",
        "agent_2": "EnvPolicy",
    }
    assert fn("agent_0") == "CliPolicy"
    assert fn("agent_1") == "CheckpointPolicy"
    assert fn("agent_2") == "EnvPolicy"
