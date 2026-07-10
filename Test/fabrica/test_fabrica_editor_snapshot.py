# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

from pathlib import Path

from schola.scripts.common.settings import (
    EnvironmentSettings,
    UnrealProjectSimulatorConfig,
)
from schola.scripts.fabrica.settings import (
    FabricaEditorSnapshotSettings,
    FabricaScriptSettings,
)


def test_resolved_snapshot_map_from_project_simulator(tmp_path: Path) -> None:
    uproject = tmp_path / "Game.uproject"
    uproject.touch()
    sim = UnrealProjectSimulatorConfig(uproject_path=uproject, map="/Game/FabricaTest")
    run = FabricaScriptSettings(
        editor_snapshot_settings=FabricaEditorSnapshotSettings(enabled=True),
        environment_settings=EnvironmentSettings(simulator_settings=sim),
    )

    assert run.resolved_snapshot_map() == "/Game/FabricaTest"


def test_resolved_uproject_path_from_project_simulator(tmp_path: Path) -> None:
    uproject = tmp_path / "Game.uproject"
    uproject.touch()
    sim = UnrealProjectSimulatorConfig(uproject_path=uproject)
    run = FabricaScriptSettings(
        environment_settings=EnvironmentSettings(simulator_settings=sim),
    )

    assert run.resolved_uproject_path() == uproject.resolve()
