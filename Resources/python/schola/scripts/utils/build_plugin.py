# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Utility to package a Schola Unreal plugin using Unreal's RunUAT BuildPlugin.

Usage examples:
  - Default (use default RunUAT path for UE_5.6):
      python -m schola.scripts.utils.build_plugin --plugin-folder Plugins/Schola
  - Specify RunUAT explicitly:
      python -m schola.scripts.utils.build_plugin --plugin-folder Plugins/Schola --uat-path "C:/Program Files/Epic Games/UE_5.6/Engine/Build/BatchFiles/RunUAT.bat"
  - Clean output and target Win64:
      python -m schola.scripts.utils.build_plugin --plugin-folder Plugins/Schola --target-platform Win64 --clean

This script follows the cyclopts-based CLI style used elsewhere in this repository.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import shlex
from pathlib import Path

from cyclopts import App

from schola.scripts.common.console import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

app = App(
    name="build-plugin", help="Package an Unreal plugin using RunUAT BuildPlugin."
)


def _which(cmd: str) -> str | None:
    """Return full path to an executable if found on PATH, otherwise None."""
    return shutil.which(cmd)


def _run(cmd: list[str], cwd: Path | None = None):
    """Run a subprocess command and raise on failure with helpful context."""
    logger.info("Running: %s", " ".join(shlex.quote(p) for p in cmd))
    try:
        subprocess.run(cmd, check=True, cwd=(str(cwd) if cwd else None))
    except subprocess.CalledProcessError as e:
        logger.error("Command failed with exit code %s: %s", e.returncode, e.cmd)
        raise


def find_uplugin(folder: Path) -> Path | None:
    """Return the first .uplugin file found in folder (non-recursive)."""
    if not folder.exists() or not folder.is_dir():
        return None
    for f in folder.iterdir():
        if f.is_file() and f.suffix == ".uplugin":
            return f.resolve()
    return None


@app.default
def main(
    plugin_folder: Path = Path("."),
    uat_path: str | None = None,
    package_dir: Path | None = None,
    target_platform: str | None = None,
    ue_version: str | None = None,
    clean: bool = False,
    extra_args: str = "",
    verbose: bool = False,
):
    """
    Build/package an Unreal plugin using RunUAT BuildPlugin.

    Parameters
    ----------
    plugin_folder : Path
        Path to the root of the plugin (should contain a .uplugin file).
    uat_path : str | None
        Explicit path to RunUAT (e.g. RunUAT.bat). If not provided, a reasonable
        default for UE_5.6 on Windows will be used.
    package_dir : Path | None
        Output directory for the packaged plugin. If not provided, a
        'Packaged/<platform-or-All>' folder under the plugin folder is used.
    target_platform : str | None
        Target platform to pass to BuildPlugin (e.g. 'Win64').
    ue_version : str | None
        Unreal Engine version hint (unused except for diagnostic).
    clean : bool
        If True, remove the package_dir before running BuildPlugin.
    extra_args : str
        Extra arguments forwarded to RunUAT (string will be split with shlex).
    verbose : bool
        Increase logging verbosity.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)

    plugin_folder = Path(plugin_folder).resolve()
    logger.debug("Plugin folder: %s", plugin_folder)

    if not plugin_folder.exists() or not plugin_folder.is_dir():
        raise FileNotFoundError(
            f"Plugin folder does not exist or is not a directory: {plugin_folder}"
        )

    uplugin = find_uplugin(plugin_folder)
    if uplugin is None:
        raise FileNotFoundError(
            f"No .uplugin file found in plugin folder: {plugin_folder}\n"
            "Ensure you're running this from the plugin root or pass --plugin-folder correctly."
        )

    # Resolve RunUAT path. Default to UE_5.6 Windows location if not provided.
    if uat_path:
        uat_path_resolved = Path(uat_path).expanduser().resolve()
    else:
        # Default recommended by the task
        default_win = Path(
            "C:/Program Files/Epic Games/UE_5.6/Engine/Build/BatchFiles/RunUAT.bat"
        )
        if default_win.exists():
            uat_path_resolved = default_win
        else:
            # Try to find any RunUAT on PATH (unlikely), otherwise fall back to the default location
            found = _which("RunUAT.bat") or _which("RunUAT")
            if found:
                uat_path_resolved = Path(found)
            else:
                # Still use the default path for messaging, but error if it doesn't exist.
                uat_path_resolved = default_win

    logger.debug("Resolved RunUAT path: %s", uat_path_resolved)

    if not uat_path_resolved.exists():
        raise FileNotFoundError(
            f"RunUAT tool not found at: {uat_path_resolved}\n"
            "Provide --uat-path to point to your Engine's RunUAT.bat, e.g.:\n"
            "  C:/Program Files/Epic Games/UE_5.6/Engine/Build/BatchFiles/RunUAT.bat"
        )

    # Determine output package directory
    if package_dir:
        out_dir = Path(package_dir).resolve()
    else:
        platform_segment = target_platform if target_platform else "All"
        out_dir = plugin_folder / "Packaged" / platform_segment
    logger.debug("Package output directory: %s", out_dir)

    # Clean if requested
    if clean and out_dir.exists():
        logger.info("Cleaning package output directory: %s", out_dir)
        shutil.rmtree(out_dir)

    # Ensure parent exists so RunUAT can write there
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd: list[str] = [
        str(uat_path_resolved),
        "BuildPlugin",
        f"-Plugin={str(uplugin)}",
        f"-Package={str(out_dir)}",
    ]
    if target_platform:
        # BuildPlugin expects a -TargetPlatforms argument with comma-separated list
        cmd.append(f"-TargetPlatforms={target_platform}")

    if extra_args:
        cmd += shlex.split(extra_args)

    logger.info("Packaging plugin %s -> %s", uplugin.name, out_dir)
    _run(cmd, cwd=plugin_folder)

    logger.info("Plugin packaging complete. Output directory: %s", out_dir)


if __name__ == "__main__":
    app()
