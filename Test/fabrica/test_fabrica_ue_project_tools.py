# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

from pathlib import Path

import pytest

from schola.scripts.fabrica import ue_project_tools


def test_ue_read_file_allowlist(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.txt").write_text("hello", encoding="utf-8")
    data = ue_project_tools.ue_read_file("a.txt", [root], [])
    assert data == "hello"


def test_ue_read_file_strips_redundant_root_prefix(tmp_path: Path) -> None:
    root = tmp_path / "Source"
    root.mkdir()
    nested = root / "Module" / "Public"
    nested.mkdir(parents=True)
    (nested / "Env.h").write_text("class Env {};", encoding="utf-8")
    data = ue_project_tools.ue_read_file("Source/Module/Public/Env.h", [root], [])
    assert data == "class Env {};"


def test_ue_read_file_denied_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    outside = tmp_path / "other"
    outside.mkdir()
    (outside / "secret.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        ue_project_tools.ue_read_file("../other/secret.txt", [root], [])


def test_ue_grep(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "f.h").write_text("int FooGoal = 1;\n", encoding="utf-8")
    out = ue_project_tools.ue_grep("Goal", [root], [], glob="*.h")
    assert "FooGoal" in out


def test_ue_read_file_denied_ignore_glob(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    binaries = root / "Binaries"
    binaries.mkdir(parents=True)
    (binaries / "Game.dll").write_text("x", encoding="utf-8")
    with pytest.raises(PermissionError, match="ignore glob"):
        ue_project_tools.ue_read_file("Binaries/Game.dll", [root], ["Binaries/*"])


def test_format_sandbox_tool_error_permission_vs_not_found() -> None:
    denied = ue_project_tools.format_sandbox_tool_error(
        operation="read",
        target="Binaries/Game.dll",
        exc=PermissionError("Path matches ignore glob: /tmp/Binaries/Game.dll"),
    )
    missing = ue_project_tools.format_sandbox_tool_error(
        operation="read",
        target="Missing.h",
        exc=FileNotFoundError("Missing.h"),
    )
    assert "Access denied" in denied
    assert "ignore glob" in denied
    assert "Failed to read" not in denied
    assert missing == "Read target not found: 'Missing.h'"
