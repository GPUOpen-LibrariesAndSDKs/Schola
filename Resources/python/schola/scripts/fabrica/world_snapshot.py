# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Launch UnrealEditor-Cmd to run the bundled Fabrica world snapshot script."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from schola.core.utils.ubt import resolve_editor_executable
from schola.scripts.fabrica.settings import FabricaEditorSnapshotSettings

logger = logging.getLogger(__name__)

_SNAPSHOT_SCRIPT = (
    Path(__file__).resolve().parent / "unreal_scripts" / "fabrica_world_snapshot.py"
)


def run_world_snapshot(
    uproject: Path,
    output_json: Path,
    settings: FabricaEditorSnapshotSettings,
    *,
    map: str | None = None,
) -> None:
    """Run Editor-Cmd with -ExecutePythonScript on the bundled snapshot script."""
    if settings.editor_path is None:
        editor = resolve_editor_executable(uproject)
    else:
        editor = settings.editor_path.resolve()
    if not _SNAPSHOT_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing snapshot script: {_SNAPSHOT_SCRIPT}")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["FABRICA_SNAPSHOT_OUT"] = str(output_json.resolve())
    env["FABRICA_MAX_ACTORS"] = str(settings.max_actors)
    if settings.class_filter_substrings:
        env["FABRICA_CLASS_FILTER"] = ",".join(settings.class_filter_substrings)

    cmd = [
        str(editor),
        str(uproject.resolve()),
    ]
    if map:
        # Load map before script (Editor expects map path after project)
        cmd.append(map)
    cmd.append(f"-ExecutePythonScript={str(_SNAPSHOT_SCRIPT.resolve())}")
    cmd.append("-unattended")
    cmd.append("-nop4")
    cmd.append("-nosplash")

    logger.info("Running editor snapshot: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=settings.timeout_s,
        env=env,
        check=False,
    )
    logger.info("Editor snapshot stdout:\n%s", proc.stdout)
    logger.info("Editor snapshot stderr:\n%s", proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"UnrealEditor-Cmd snapshot failed with exit {proc.returncode}: {proc.stderr[:500]}"
        )
    if not output_json.is_file():
        raise RuntimeError(
            f"UnrealEditor-Cmd exited 0 but snapshot file was not written: {output_json}"
        )
