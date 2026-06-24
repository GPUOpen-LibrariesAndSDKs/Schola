# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Tests for Schola RLlib policy mapping helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pytest

from schola.rllib.policy_mapping import (
    SCHOLA_POLICY_MAPPING_FILENAME,
    build_policy_mapping_record,
    load_policy_mapping_sidecar,
    log_env_mapping_disagreements,
    policy_mapping_fn_from_agent_to_policy,
    resolve_policy_mapping_for_eval,
    validate_policy_mapping_against_module_ids,
    write_policy_mapping_sidecar,
)


@pytest.fixture
def env_policy_fn():
    return lambda agent_id, *args, **kwargs: "EnvPolicy"


@pytest.fixture
def checkpoint_with_sidecar(tmp_path: Path) -> Callable[..., Path]:
    """Write ``schola_policy_mapping.json`` under ``tmp_path`` and return that dir."""

    def _write(
        agent_to_policy: Dict[str, str],
        *,
        module_ids: Optional[List[str]] = None,
    ) -> Path:
        if module_ids is None:
            module_ids = sorted(set(agent_to_policy.values()))
        write_policy_mapping_sidecar(
            tmp_path,
            build_policy_mapping_record(
                agent_ids=list(agent_to_policy),
                policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_to_policy[
                    str(agent_id)
                ],
                module_ids=module_ids,
            ),
        )
        return tmp_path

    return _write


def test_build_and_load_policy_mapping_sidecar_roundtrip(tmp_path):
    record = build_policy_mapping_record(
        agent_ids=["agent_0", "agent_1"],
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "Pawn",
        agent_types={"agent_0": "Pawn", "agent_1": "Pawn"},
        module_ids=["Pawn"],
    )
    write_policy_mapping_sidecar(tmp_path, record)
    loaded = load_policy_mapping_sidecar(tmp_path)
    assert loaded == record
    assert (tmp_path / SCHOLA_POLICY_MAPPING_FILENAME).is_file()


def test_validate_policy_mapping_against_module_ids_raises():
    fn = validate_policy_mapping_against_module_ids(
        policy_mapping_fn_from_agent_to_policy({"agent_0": "Ghost"}),
        ["Pawn"],
    )
    with pytest.raises(KeyError, match="Ghost"):
        fn("agent_0")


def test_resolve_policy_mapping_prefers_cli_over_checkpoint_and_env(
    checkpoint_with_sidecar, env_policy_fn
):
    checkpoint = checkpoint_with_sidecar({"agent_0": "CheckpointPolicy"})

    fn, source, table = resolve_policy_mapping_for_eval(
        cli_agent_to_policy={"agent_0": "CliPolicy"},
        checkpoint=checkpoint,
        module_ids=["CliPolicy"],
        agent_ids=["agent_0"],
        env_policy_mapping_fn=env_policy_fn,
    )

    assert source == "CLI"
    assert table == {"agent_0": "CliPolicy"}
    assert fn("agent_0") == "CliPolicy"


def test_resolve_policy_mapping_uses_checkpoint_when_cli_missing(
    checkpoint_with_sidecar, env_policy_fn
):
    checkpoint = checkpoint_with_sidecar({"agent_0": "CheckpointPolicy"})

    fn, source, table = resolve_policy_mapping_for_eval(
        cli_agent_to_policy=None,
        checkpoint=checkpoint,
        module_ids=["CheckpointPolicy"],
        agent_ids=["agent_0"],
        env_policy_mapping_fn=env_policy_fn,
    )

    assert source == "checkpoint"
    assert table == {"agent_0": "CheckpointPolicy"}
    assert fn("agent_0") == "CheckpointPolicy"


def test_resolve_policy_mapping_falls_back_to_environment(tmp_path):
    env_fn = lambda agent_id, *args, **kwargs: "Pawn"

    fn, source, table = resolve_policy_mapping_for_eval(
        cli_agent_to_policy=None,
        checkpoint=tmp_path,
        module_ids=["Pawn"],
        agent_ids=["agent_0"],
        env_policy_mapping_fn=env_fn,
    )

    assert source == "environment"
    assert table == {"agent_0": "Pawn"}
    assert fn("agent_0") == "Pawn"


def test_log_env_mapping_disagreements_emits_warning(caplog):
    caplog.set_level(logging.WARNING)
    log_env_mapping_disagreements(
        authoritative_agent_to_policy={"agent_0": "CliPolicy"},
        agent_ids=["agent_0"],
        env_policy_mapping_fn=lambda agent_id, *args, **kwargs: "EnvPolicy",
        source="CLI",
    )
    assert "CliPolicy" in caplog.text
    assert "EnvPolicy" in caplog.text
