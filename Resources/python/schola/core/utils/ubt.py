# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Utility Functions and classes for working with Unreal Engine's UBT (Unreal Build Tool) system.
"""

import os
import logging
import tempfile
from pathlib import Path
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

logger = logging.getLogger(__name__)


def get_unreal_platform() -> Literal["Win64", "Linux"]:
    """
    Get the current platform for Unreal Engine.

    Returns
    -------
    str
        The platform string for Unreal Engine (e.g., "Win64", "Linux", "Mac").
    """
    if platform.system() == "Windows":
        return "Win64"
    elif platform.system() == "Linux":
        return "Linux"
    else:
        raise ValueError("Unsupported platform: {}".format(platform.system()))


@dataclass
class UBTCommand:
    """
    Dataclass for constructing Unreal Build Tool (UBT) command line arguments.
    """

    ubt_path: Path | str  # Path to the Unreal Build Tool executable
    project_file: Path | str  # Path to the .uproject file
    target_platform: str | None = (
        get_unreal_platform()
    )  # Target platform (e.g., Win64, Linux)
    should_package: bool = True  # Whether to package the build
    should_clean: bool = False  # Whether to clean before building
    should_cook: bool = True  # Whether to cook content
    fast_cook: bool = (
        True  # Use FastCook for quicker iteration, only used if should_cook is True.
    )
    should_build: bool = True  # Whether to build the project
    no_p4: bool = True  # Disable Perforce integration (-NoP4)
    prereqs: bool = True  # Install prerequisites (-prereqs)
    no_compile: bool = True  # Skip compiling C++ code (-nocompile)
    no_compile_uat: bool = True  # Skip compiling Unreal Automation Tool (-nocompileuat)
    configuration: Literal["Development", "Shipping"] = (
        "Development"  # Build configuration
    )
    no_debug_info: bool = True  # Exclude debug info (-nodebuginfo)
    unattended: bool = True  # Run in unattended mode (-unattended)
    staging_dir: Path | str | None = None  # Directory to stage build output
    force_monolithic: bool = (
        False  # Build as a monolithic executable (-ForceMonolithic)
    )
    maps: list[str] = field(default_factory=list)  # List of maps to cook/package
    all_maps: bool = (
        True  # Whether to cook all maps (-AllMaps). If False, maps must be set. If True, maps will be ignored.
    )
    stdout: bool = True  # Whether to route build process output to stdout

    def build_args(self) -> list[str]:
        """
        Build and return the complete list of UBT command line arguments.

        Returns
        -------
        list[str]
            The complete list of UBT command line arguments.
        """
        args: list[str] = [str(self.ubt_path), "BuildCookRun"]

        if self.target_platform:
            args.append(f"-platform={self.target_platform}")

        if self.project_file:
            args.append(f"-project={self.project_file}")

        if self.should_package:
            args.append("-package")

        if self.should_clean:
            args.append("-clean")

        if self.should_cook:
            args.append("-cook")
            if self.fast_cook:
                args.append("-FastCook")

        if self.should_build:
            args.append("-build")

        if self.no_p4:
            args.append("-NoP4")

        if self.prereqs:
            args.append("-prereqs")

        if self.no_compile:
            args.append("-nocompile")

        if self.no_compile_uat:
            args.append("-nocompileuat")

        if self.configuration:
            args.append(f"-configuration={self.configuration}")

        if self.stdout:
            args.append("-stdout")

        if self.no_debug_info:
            args.append("-nodebuginfo")

        if self.unattended:
            args.append("-unattended")

        if self.staging_dir:
            args.append("-stage")
            args.append(f"-stagingdirectory={self.staging_dir}")

        if self.force_monolithic:
            args.append("-ForceMonolithic")

        if self.all_maps:
            args.append("-AllMaps")
        elif len(self.maps) > 0:
            args.append(f"-map={'+'.join(self.maps)}")

        return args

    def run(self) -> subprocess.CompletedProcess[str]:
        """
        Run the UBT command.
        """
        return subprocess.run(self.build_args(), capture_output=True, text=True)


def resolve_uproject_file(uproject_path: Path | str) -> Path:
    """Resolve a ``.uproject`` file or a directory containing one."""
    path = Path(uproject_path)
    if path.is_dir():
        uproject_file = get_project_file(path)
        if uproject_file is None:
            raise FileNotFoundError(f"No .uproject file found in directory: {path}")
        return uproject_file.resolve()
    if path.suffix == ".uproject" and path.exists():
        return path.resolve()
    raise FileNotFoundError(
        f"Not a valid .uproject file or directory containing one: {path}"
    )


def resolve_build_dir(
    uproject_file: Path,
    build_dir: Optional[Path | str] = None,
) -> Path:
    """Return the staging directory for a project build, creating it if needed."""
    if build_dir is not None:
        resolved = Path(build_dir).resolve()
        if resolved.exists() and not resolved.is_dir():
            raise FileExistsError(f"Build directory {resolved} is not a directory")
    else:
        resolved = Path(tempfile.gettempdir()) / f"schola_build_{uproject_file.stem}"
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def expected_executable_path(uproject_file: Path, build_dir: Path) -> Path:
    """Return the staged standalone executable path for a built Unreal project."""
    executable_filename = uproject_file.stem + (
        ".exe" if platform.system() == "Windows" else ""
    )
    return (
        build_dir
        / platform.system()
        / uproject_file.stem
        / "Binaries"
        / get_unreal_platform()
        / executable_filename
    )


def resolve_ubt_path(
    uproject_file: Path,
    ubt_path: Optional[Path | str] = None,
) -> Path:
    """Resolve the RunUAT script for a project, auto-detecting when omitted."""
    if ubt_path is not None:
        return Path(ubt_path).resolve()
    ue_version = get_ue_version(uproject_file)
    if ue_version is None:
        raise ValueError(
            "Could not determine Unreal Engine version from .uproject file"
        )
    resolved = get_ubt_path(uproject_file.parent, ue_version)
    if resolved is None:
        raise FileNotFoundError(
            "Could not find Unreal Build Tool (UBT) path from project folder, "
            "passed ubt_path, or default path."
        )
    return resolved.resolve()


def run_ubt_project_build(
    uproject_file: Path,
    build_dir: Path,
    *,
    ubt_path: Optional[Path | str] = None,
    map: Optional[str] = None,
    extra_ubt_args: Optional[Dict[str, Any]] = None,
) -> subprocess.CompletedProcess[str]:
    """Run UBT for a project and return the completed process."""
    ubt_args: Dict[str, Any] = (
        extra_ubt_args.copy() if extra_ubt_args is not None else {}
    )
    ubt_args.update(
        {
            "ubt_path": resolve_ubt_path(uproject_file, ubt_path),
            "staging_dir": build_dir,
            "project_file": uproject_file,
            "maps": [map] if map is not None else [],
            "all_maps": map is None,
        }
    )
    completed_build = UBTCommand(**ubt_args).run()
    return completed_build


def build_project_executable(
    uproject_path: Path | str,
    *,
    build_dir: Optional[Path | str] = None,
    ubt_path: Optional[Path | str] = None,
    map: Optional[str] = None,
    extra_ubt_args: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Build an Unreal project executable and return its staged path.

    Parameters
    ----------
    uproject_path
        Path to a ``.uproject`` file or a directory containing one.
    build_dir
        Optional staging directory. When omitted, a temp directory is used.
    ubt_path
        Optional RunUAT path. When omitted, it is auto-detected from the project.
    map
        Optional map to cook/package. When omitted, all maps are cooked.
    build_log_path
        Optional file to write captured UBT stdout/stderr to.
    extra_ubt_args
        Extra keyword arguments forwarded to :class:`UBTCommand`.
    """
    uproject_file = resolve_uproject_file(uproject_path)
    staging_dir = resolve_build_dir(uproject_file, build_dir)
    completed_build = run_ubt_project_build(
        uproject_file,
        staging_dir,
        ubt_path=ubt_path,
        map=map,
        extra_ubt_args=extra_ubt_args,
    )
    executable_path = expected_executable_path(uproject_file, staging_dir)
    if completed_build.returncode != 0:
        exception_message = f"Unreal Build failed with return code {completed_build.returncode} and the following output:\n"
        if completed_build.stderr:
            exception_message += f"stderr:\n {completed_build.stderr}\n"
        if completed_build.stdout:
            exception_message += f"stdout:\n {completed_build.stdout}\n"
        raise Exception(exception_message)
    if not executable_path.exists():
        raise FileNotFoundError(f"Built executable not found at {executable_path}")
    logger.info("Built executable to: %s", executable_path)
    return executable_path


def get_ue_version(project_file: Path) -> str | None:
    """
    Extract the Unreal Engine version from a .uproject file.

    Parameters
    ----------
    project_file : Path
        Path to the .uproject file.

    Returns
    -------
    str
        The Unreal Engine version string (e.g., "5.5").
    """
    with open(project_file, "r") as f:
        # read the file as JSON
        import json

        loaded = json.load(f)
    if not isinstance(loaded, dict):
        return None
    engine_association = loaded.get("EngineAssociation")
    if isinstance(engine_association, str):
        return engine_association
    return None


def get_project_file(project_folder: Path) -> Path | None:
    """
    Find the .uproject file in a project folder.

    Parameters
    ----------
    project_folder : str or Path
        Path to the project folder to search.

    Returns
    -------
    Path or None
        Path to the .uproject file if found, None otherwise.
    """
    for file in os.listdir(project_folder):
        if file.endswith(".uproject"):
            return Path(os.path.join(project_folder, file))
    return None


def get_engine_path_from_sln(sln_file: Path) -> Path | None:
    """
    Extract the Unreal Engine path from a Visual Studio solution file.

    Parameters
    ----------
    sln_file : Path
        Path to the .sln file.

    Returns
    -------
    Path
        Path to the Unreal Engine installation directory.
    """
    with open(sln_file, "r") as f:
        for line in f:
            if '"UnrealBuildTool"' in line:
                # extract the path to the UnrealBuildTool
                parts = line.split(",")
                # Get the path to the engine folder using the UBT entry in the solution file
                return sln_file.parent / Path(parts[1].split("Engine")[0].strip(' "'))
    return None


def get_sln_file_from_project(project_folder: Path) -> Path | None:
    """
    Find the Visual Studio solution file in a project folder.

    Parameters
    ----------
    project_folder : Path
        Path to the project folder to search.

    Returns
    -------
    Optional[Path]
        Path to the .sln file if found, None otherwise.
    """
    for file in os.listdir(project_folder):
        if file.endswith(".sln"):
            return project_folder / file
    return None


def get_ubt_path(sln_or_project_folder: Path, ue_version: str = "5.5") -> Path | None:
    """
    Get the path to the Unreal Build Tool (UBT) RunUAT script.

    Parameters
    ----------
    project_folder : Path
        Path to the project folder.
    ue_version : str, default="5.5"
        The Unreal Engine version to use for fallback default path.

    Returns
    -------
    Optional[Path]
        Path to the RunUAT batch/shell script if found, None otherwise.
    """
    sln_file = None
    if sln_or_project_folder.is_file() and sln_or_project_folder.suffix == ".sln":
        sln_file = sln_or_project_folder
    elif sln_or_project_folder.is_dir():
        sln_file = get_sln_file_from_project(sln_or_project_folder)

    if sln_file is not None:
        engine_path = get_engine_path_from_sln(sln_file)
        if engine_path is not None:
            return (
                engine_path
                / "Engine"
                / "Build"
                / "BatchFiles"
                / f"RunUAT.{('bat' if platform.system() == 'Windows' else 'sh')}"
            )
    else:
        # if we can't find it in the sln file, try and use a default path
        if platform.system() == "Windows":
            return Path(
                "C:/Program Files/Epic Games/UE_"
                + ue_version
                + "/Engine/Build/BatchFiles/RunUAT.bat"
            )
    return None


def get_editor_executable_path(engine_path: Path) -> Path:
    """
    Get the path to the Unreal Editor command-line executable.

    Parameters
    ----------
    engine_path : Path
        Path to the Unreal Engine installation directory.

    Returns
    -------
    Path
        Path to the UnrealEditor-Cmd executable.
    """
    editor_tool = (
        "UnrealEditor-Cmd.exe" if platform.system() == "Windows" else "UnrealEditor-Cmd"
    )
    bin_dir = "Win64" if platform.system() == "Windows" else "Linux"
    return engine_path / "Engine" / "Binaries" / bin_dir / editor_tool


def resolve_editor_executable(uproject: Path) -> Path:
    """
    Resolve the Unreal Editor command-line executable from a .uproject file.

    Parameters
    ----------
    uproject : Path
        Path to the .uproject file.

    Returns
    -------
    Path
        Path to the UnrealEditor-Cmd executable.

    Raises
    ------
    FileNotFoundError
        If no .sln file is found or the Engine path cannot be resolved.
    """
    sln = get_sln_file_from_project(uproject.parent)
    if sln is None:
        raise FileNotFoundError(
            f"Could not locate a .sln next to {uproject.parent} to resolve the Engine install."
        )
    engine = get_engine_path_from_sln(sln)
    if engine is None:
        raise FileNotFoundError(f"Could not resolve Engine path from {sln}")
    editor = get_editor_executable_path(engine)
    if not editor.exists():
        raise FileNotFoundError(f"Could not resolve UnrealEditor-Cmd executable from {engine}")
    return editor


def build_executable(
    project_file: Path | str,
    build_dir: Path | str,
    ubt_path: Path | str,
    **kwargs,
) -> subprocess.CompletedProcess[bytes]:
    """
    Build an Unreal Engine project executable using the Unreal Build Tool.

    Parameters
    ----------
    project_file : str
        Path to the .uproject file.
    build_dir : str
        Directory where the built executable will be staged.
    ubt_path : str
        Path to the Unreal Build Tool (RunUAT) script.
    **kwargs
        Additional arguments to pass to the UBTCommand constructor.

    Returns
    -------
    subprocess.CompletedProcess
        The result of the build process.
    """
    args = UBTCommand(
        ubt_path=ubt_path, project_file=project_file, staging_dir=build_dir, **kwargs
    ).build_args()
    comp_process = subprocess.run(args, capture_output=True)
    return comp_process


def quick_build_unreal_project(
    project_folder: Path | str,
    build_dir: Path | str,
    ubt_path: Path | str | None = None,
) -> Path:
    """
    Build function with reasonable defaults to build an Unreal Engine project and return the path to the executable.

    This function automates the process of building an Unreal project by:
    1. Finding the .uproject file
    2. Determining the UE version
    3. Locating the build tool
    4. Building the executable
    5. Returning the path to the built executable

    Parameters
    ----------
    project_folder : str
        Path to the project folder containing the .uproject file.
    build_dir : str
        Directory where the built executable will be staged.
    ubt_path : Optional[str], optional
        Path to the Unreal Build Tool. If None, it will be auto-detected.

    Returns
    -------
    Path
        Path to the built executable file.

    Raises
    ------
    FileNotFoundError
        If the .uproject file is not found, or the UBT path cannot be determined from the project folder.
    ValueError
        If the UE version cannot be determined from the .uproject file.
    """
    project_folder = Path(project_folder).resolve()
    build_dir = Path(build_dir).resolve()
    uproject_file = get_project_file(project_folder)
    if uproject_file is None:
        raise FileNotFoundError("No .uproject file found in the project folder")
    ue_version = get_ue_version(uproject_file)
    if ue_version is None:
        raise ValueError(
            "Could not determine Unreal Engine version from .uproject file"
        )
    ubt_path = (
        get_ubt_path(project_folder, ue_version) if ubt_path is None else Path(ubt_path)
    )
    if ubt_path is None:
        raise FileNotFoundError(
            "Could not find Unreal Build Tool (UBT) path from project folder."
        )

    completed_build_process = build_executable(
        project_file=uproject_file,
        build_dir=build_dir,
        ubt_path=ubt_path,
    )

    if completed_build_process.returncode != 0:
        exception_message = f"Unreal build failed with return code {completed_build_process.returncode} and the following output:\n"
        if completed_build_process.stderr:
            exception_message += f"stderr:\n {completed_build_process.stderr.decode('utf-8', errors='ignore')}\n"
        if completed_build_process.stdout:
            exception_message += f"stdout:\n {completed_build_process.stdout.decode('utf-8', errors='ignore')}\n"
        raise Exception(exception_message)

    executable_filename = Path(uproject_file).name.split(".")[0] + (
        ".exe" if platform.system() == "Windows" else ""
    )
    built_path = (
        Path(build_dir)
        / platform.system()
        / "ScholaExamples"
        / "Binaries"
        / get_unreal_platform()
        / executable_filename
    )
    return built_path
