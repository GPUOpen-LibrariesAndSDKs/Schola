# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

import ast
import inspect
from pathlib import Path

import pytest
from cyclopts import App
from cyclopts.exceptions import ValidationError

from schola.scripts.common.settings import (
    EnvironmentSettings,
    ExternalSimulatorConfig,
    UnrealProjectSimulatorConfig,
)
from schola.scripts.fabrica.settings import (
    FabricaEditorSnapshotSettings,
    FabricaEnvironmentSettings,
    FabricaLLMSettings,
    FabricaLoopSettings,
    FabricaPathsSettings,
    FabricaScriptSettings,
    _collapse_nested_code_roots,
)

_FABRICA_CLI_SETTING_CLASSES = (
    FabricaLLMSettings,
    FabricaEditorSnapshotSettings,
    FabricaLoopSettings,
    FabricaPathsSettings,
)

def test_policy_feedback_interval_default_is_valid() -> None:
    settings = FabricaLoopSettings()
    assert settings.policy_feedback_interval == 1


@pytest.mark.parametrize("invalid_interval", [0, -1])
def test_policy_feedback_interval_rejects_non_positive(invalid_interval: int) -> None:
    with pytest.raises(ValueError, match="policy_feedback_interval must be greater than 0"):
        FabricaLoopSettings(policy_feedback_interval=invalid_interval)


def test_policy_feedback_interval_cli_rejects_zero() -> None:
    app = App()
    app.default(FabricaLoopSettings)

    with pytest.raises(ValidationError, match="Must be > 0"):
        app.parse_args(["--policy-feedback-interval", "0"], exit_on_error=False)


def test_resolved_code_roots_includes_env_header_and_uproject_parents(tmp_path) -> None:
    header = tmp_path / "Source" / "Game" / "Public" / "Env.h"
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text("", encoding="utf-8")
    uproject = tmp_path / "Game.uproject"
    uproject.write_text("{}", encoding="utf-8")

    settings = FabricaScriptSettings(
        paths_settings=FabricaPathsSettings(env_header=header),
        environment_settings=FabricaEnvironmentSettings(
            simulator_settings=UnrealProjectSimulatorConfig(uproject_path=uproject),
        ),
    )
    roots = settings.resolved_code_roots
    assert roots == [uproject.parent.resolve()]


def test_resolved_code_roots_adds_extra_roots_and_collapses_nested(tmp_path) -> None:
    header = tmp_path / "Source" / "Game" / "Public" / "Env.h"
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text("", encoding="utf-8")
    uproject = tmp_path / "Game.uproject"
    uproject.write_text("{}", encoding="utf-8")
    nested = header.parent / "Nested"
    nested.mkdir()

    settings = FabricaScriptSettings(
        paths_settings=FabricaPathsSettings(
            env_header=header,
            code_roots=[nested, header.parent],
        ),
        environment_settings=FabricaEnvironmentSettings(
            simulator_settings=UnrealProjectSimulatorConfig(uproject_path=uproject),
        ),
    )
    roots = settings.resolved_code_roots
    assert roots == [uproject.parent.resolve()]


def test_resolved_code_roots_keeps_sibling_extra_roots(tmp_path) -> None:
    header = tmp_path / "headers" / "Env.h"
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text("", encoding="utf-8")
    uproject = tmp_path / "gamedir"/ "Game.uproject"
    uproject.parent.mkdir(parents=True, exist_ok=True)
    uproject.write_text("{}", encoding="utf-8")
    extra = tmp_path / "plugins"
    extra.mkdir()

    settings = FabricaScriptSettings(
        paths_settings=FabricaPathsSettings(env_header=header, code_roots=[extra]),
        environment_settings=FabricaEnvironmentSettings(
            simulator_settings=UnrealProjectSimulatorConfig(uproject_path=uproject),
        ),
    )
    roots = settings.resolved_code_roots
    assert set(roots) == set([uproject.parent.resolve(), header.parent.resolve(), extra.resolve()])


def test_collapse_nested_code_roots_keeps_shortest_ancestor(tmp_path) -> None:
    ancestor = tmp_path / "A" / "B"
    descendant = ancestor / "C"
    ancestor.mkdir(parents=True)
    descendant.mkdir(parents=True)

    assert _collapse_nested_code_roots([descendant, ancestor]) == [ancestor.resolve()]
    assert _collapse_nested_code_roots([ancestor, descendant]) == [ancestor.resolve()]


def test_collapse_nested_code_roots_deduplicates_exact_paths(tmp_path) -> None:
    root = tmp_path / "proj"
    root.mkdir()

    assert _collapse_nested_code_roots([root, root]) == [root.resolve()]
