# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Tests for Schola RLlib policy mapping helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

import pytest

from schola.rllib.policy_mapping import (
    ENV_CONFIG_POLICY_MAPPING_RECORD_KEY,
    SCHOLA_POLICY_MAPPING_COMPONENT,
    ScholaPolicyMappingCheckpoint,
    load_agent_to_policy,
    make_policy_mapping_checkpoint_from_config,
    resolve_policy_mapping_for_eval,
    validate_agent_to_policy_against_module_ids,
)


@pytest.fixture
def env_agent_to_policy():
    return {"agent_0": "EnvPolicy", "agent_1": "EnvPolicy", "agent_2": "EnvPolicy"}


@pytest.fixture
def checkpoint_with_mapping(tmp_path: Path) -> Callable[..., Path]:
    """Write the ``schola_policy_mapping`` component under ``tmp_path`` and return it."""

    def _write(agent_to_policy: Dict[str, str]) -> Path:
        ScholaPolicyMappingCheckpoint(agent_to_policy).save_to_path(
            tmp_path / SCHOLA_POLICY_MAPPING_COMPONENT
        )
        return tmp_path

    return _write


def test_checkpoint_component_roundtrips_agent_to_policy(tmp_path):
    agent_to_policy = {"agent_0": "Pawn", "agent_1": "Pawn"}
    component_dir = tmp_path / SCHOLA_POLICY_MAPPING_COMPONENT
    ScholaPolicyMappingCheckpoint(agent_to_policy).save_to_path(component_dir)

    restored = ScholaPolicyMappingCheckpoint.from_checkpoint(component_dir)
    assert restored.agent_to_policy == agent_to_policy


def test_checkpoint_component_reads_legacy_wrapped_format(tmp_path):
    agent_to_policy = {"agent_0": "Pawn"}
    component_dir = tmp_path / SCHOLA_POLICY_MAPPING_COMPONENT
    ScholaPolicyMappingCheckpoint({"agent_to_policy": agent_to_policy}).save_to_path(
        component_dir
    )

    restored = ScholaPolicyMappingCheckpoint.from_checkpoint(component_dir)
    assert restored.agent_to_policy == agent_to_policy


def test_load_agent_to_policy_returns_none_without_component(tmp_path):
    assert load_agent_to_policy(tmp_path) is None


def test_load_agent_to_policy_reads_saved_component(checkpoint_with_mapping):
    checkpoint = checkpoint_with_mapping({"agent_0": "Pawn"})
    assert load_agent_to_policy(checkpoint) == {"agent_0": "Pawn"}


def test_make_policy_mapping_checkpoint_from_config_reads_env_config():
    agent_to_policy = {"agent_0": "Pawn"}

    class _Config:
        env_config = {ENV_CONFIG_POLICY_MAPPING_RECORD_KEY: agent_to_policy}

    component = make_policy_mapping_checkpoint_from_config(_Config())
    assert component.agent_to_policy == agent_to_policy


def test_make_policy_mapping_checkpoint_from_config_defaults_to_empty():
    class _Config:
        env_config = {}

    component = make_policy_mapping_checkpoint_from_config(_Config())
    assert component.agent_to_policy == {}


def test_validate_agent_to_policy_against_module_ids_raises():
    with pytest.raises(KeyError, match="Ghost"):
        validate_agent_to_policy_against_module_ids(
            {"agent_0": "Ghost"},
            ["Pawn"],
        )


def test_resolve_policy_mapping_prefers_cli_over_checkpoint_and_env(
    checkpoint_with_mapping, env_agent_to_policy
):
    checkpoint = checkpoint_with_mapping({"agent_0": "CheckpointPolicy"})

    table = resolve_policy_mapping_for_eval(
        agent_ids=["agent_0"],
        module_ids=["CliPolicy"],
        checkpoint=checkpoint,
        env_agent_to_policy=env_agent_to_policy,
        cli_agent_to_policy={"agent_0": "CliPolicy"},
    )

    assert table == {"agent_0": "CliPolicy"}


def test_resolve_policy_mapping_uses_checkpoint_when_cli_missing(
    checkpoint_with_mapping, env_agent_to_policy
):
    checkpoint = checkpoint_with_mapping({"agent_0": "CheckpointPolicy"})

    table = resolve_policy_mapping_for_eval(
        agent_ids=["agent_0"],
        module_ids=["CheckpointPolicy"],
        checkpoint=checkpoint,
        env_agent_to_policy=env_agent_to_policy,
    )

    assert table == {"agent_0": "CheckpointPolicy"}


def test_resolve_policy_mapping_falls_back_to_environment(
    tmp_path, env_agent_to_policy
):
    table = resolve_policy_mapping_for_eval(
        agent_ids=["agent_0"],
        module_ids=["EnvPolicy"],
        checkpoint=tmp_path,
        env_agent_to_policy=env_agent_to_policy,
    )

    assert table == {"agent_0": "EnvPolicy"}


def test_resolve_policy_mapping_merges_per_agent_sources(
    checkpoint_with_mapping, env_agent_to_policy
):
    checkpoint = checkpoint_with_mapping(
        {"agent_0": "CheckpointPolicy", "agent_1": "CheckpointPolicy"},
    )

    table = resolve_policy_mapping_for_eval(
        agent_ids=["agent_0", "agent_1", "agent_2"],
        module_ids=["CheckpointPolicy", "CliPolicy", "EnvPolicy"],
        checkpoint=checkpoint,
        env_agent_to_policy=env_agent_to_policy,
        cli_agent_to_policy={"agent_0": "CliPolicy"},
    )

    assert table == {
        "agent_0": "CliPolicy",
        "agent_1": "CheckpointPolicy",
        "agent_2": "EnvPolicy",
    }
