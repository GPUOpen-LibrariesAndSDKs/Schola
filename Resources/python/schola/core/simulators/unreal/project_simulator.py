# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""
A connection builds an Unreal Project if Necessary and then launches a standalone Executable Connector.
"""

import logging
from pathlib import Path
import platform
import tempfile
from typing import Any
from schola.core.simulators.unreal.executable_simulator import UnrealExecutable

logger = logging.getLogger(__name__)
from schola.core.utils.ubt import (
    UBTCommand,
    get_project_file,
    get_ubt_path,
    get_ue_version,
    get_unreal_platform,
)

logger = logging.getLogger(__name__)


def get_executable_path(build_dir: Path, project_name: str) -> Path:
    return (
        build_dir
        / platform.system()
        / project_name
        / "Binaries"
        / get_unreal_platform()
        / (project_name + (".exe" if platform.system() == "Windows" else ""))
    )


def resolve_build_arguments(
    uproject_path: Path,
    build_dir: Path | str | None = None,
    ubt_path: Path | str | None = None,
) -> tuple[Path, Path, Path, Path]:
    uproject_file: Path
    if uproject_path.is_dir():
        discovered_uproject = get_project_file(uproject_path)
        if discovered_uproject is None:
            raise FileNotFoundError(
                f"No .uproject file found in directory: {uproject_path}"
            )
        uproject_file = discovered_uproject
    elif uproject_path.exists() and uproject_path.suffix == ".uproject":
        uproject_file = uproject_path
    else:
        raise FileNotFoundError(
            f"Not a valid .uproject file or directory containing a .uproject file: {uproject_path}"
        )
    uproject_file = uproject_file.resolve()
    project_name = uproject_file.stem

    if build_dir is not None:
        build_dir = Path(build_dir).resolve()
    else:
        # Generate a cache folder with the build in temp directory
        build_dir = Path(tempfile.gettempdir()) / f"schola_build_{project_name}"
        logger.info(
            "No Build Directory Provided, Using temporary build directory: %s",
            build_dir,
        )
    executable_path = get_executable_path(build_dir, project_name)

    ubt_path = Path(ubt_path).resolve() if ubt_path is not None else None

    if ubt_path is None:
        ue_version = get_ue_version(uproject_file)
        if ue_version is None:
            raise ValueError(
                "Could not determine Unreal Engine version from .uproject file"
            )
        project_folder = uproject_file.parent
        ubt_path = get_ubt_path(project_folder, ue_version)

    if ubt_path is None:
        raise FileNotFoundError(
            "Could not find Unreal Build Tool (UBT) path from project folder, passed ubt_path or default path."
        )

    return uproject_file, build_dir, ubt_path, executable_path


def build_project(
    uproject_path: Path,
    build_dir: Path,
    ubt_path: Path,
    map_path: str | None = None,
    extra_ubt_args: dict[str, Any] | None = None,
) -> None:

    project_name = uproject_path.stem
    # Warn the user if map doesn't start with /Game/
    if map_path is not None:
        map_path = try_and_resolve_map(map_path)

    ubt_args = extra_ubt_args.copy() if extra_ubt_args is not None else {}
    ubt_args.update(
        {
            "ubt_path": ubt_path,
            "staging_dir": build_dir,
            "project_file": uproject_path,
            "maps": [map_path] if map_path is not None else [],
            "all_maps": False if map_path is not None else True,
        }
    )
    ubt_command = UBTCommand(**ubt_args)
    completed_build_process = ubt_command.run()

    if completed_build_process.returncode != 0:
        exception_message = f"Unreal build failed with return code {completed_build_process.returncode} and the following output:\n"
        if completed_build_process.stderr:
            exception_message += f"stderr:\n {completed_build_process.stderr}\n"
        if completed_build_process.stdout:
            exception_message += f"stdout:\n {completed_build_process.stdout}\n"
        raise Exception(exception_message)


def is_valid_map_path(map_path: str) -> bool:
    """
    Check if a map path is valid.

    Parameters
    ----------
    map_path : str
        The map path to check.

    Returns
    -------
    bool
        True if the map path is valid, False otherwise.
    """
    return (
        map_path.startswith("/Game/")
        and not map_path.endswith("/")
        and not map_path.endswith(".umap")
    )


def try_and_resolve_map(map_path: str) -> str:
    """
    Normalize a user-supplied map path to Unreal ``/Game/...`` form and validate it.

    Parameters
    ----------
    map_path : str
        The map path to normalize.

    Returns
    -------
    str
        The normalized map path.

    Raises
    ------
    FileNotFoundError
        If the map path is not a valid map path.
    """
    if is_valid_map_path(map_path):
        return map_path

    if "/Content/" in map_path:
        map_path = "/Game/" + map_path.split("/Content/")[1]
        logger.warning(
            f"Converted Map path to {map_path} by removing everything before '/Content/' and converting to '/Game/'."
        )

    if map_path.startswith("Game/"):
        map_path = "/" + map_path
        logger.warning(
            f"Map path {map_path} starts with 'Game/'. Converted Map path to {map_path} by adding '/' to the start."
        )

    if "/Game/" in map_path and not map_path.startswith("/Game/"):
        map_path = "/Game/" + map_path.split("/Game/")[1]
        logger.warning(
            f" /Game/ prefix found in map path, but not at the start. Converted Map path to {map_path} by removing everything before '/Game/'."
        )

    # ignore the .umap extension if it exists in the supplied map name
    if map_path.endswith(".umap"):
        logger.warning(
            f"Map path {map_path} seems to end with .umap. Ignoring the extension."
        )
        map_path = map_path[:-5]

    if not is_valid_map_path(map_path):
        raise FileNotFoundError(
            f"Map '{map_path}' is not a valid map path. Unreal expects all map paths to start with '/Game/' and be a map file without umap extension."
        )

    return map_path


class UnrealProject(UnrealExecutable):
    """
    A simulator that builds an Unreal Project if necessary and then launches a standalone Executable.

    This class extends UnrealExecutable by adding build functionality. It will automatically
    build the Unreal project before running it if needed.

    Parameters
    ----------
    uproject_path : Path | str
        Path to the .uproject file or directory containing it
    build_dir : Path | str | None, default=None
        Directory where the built executable will be staged. If None, uses a temporary directory.
    ubt_path : Path | str | None, default=None
        Path to the Unreal Build Tool (RunUAT) script. If None, auto-detects from project.
    use_cached_build : bool, default=True
        Whether to use a cached build if it exists. If True, will only build if the executable does not exist.
    headless_mode : bool, default=False
        Whether to run in headless mode
    map : str, optional
        The map to run. If blank, cooks/packages all maps and runs the default map on start.
    display_logs : bool, default=True
        Whether to display logs
    set_fps : int, optional
        Use a fixed fps while running
    disable_script : bool, default=True
        Whether to disable the autolaunch script setting in the Unreal Engine Schola Plugin
    extra_executable_args : Optional[List[str]]
        Additional arguments to pass to the command line when launching the executable.

    Attributes
    ----------
    uproject_path : Path
        Path to the .uproject file
    uproject_file : Path
        Resolved path to the ``.uproject`` file
    build_dir : Path
        Directory where the built executable is staged
    ubt_path : Optional[Path]
        Path to the Unreal Build Tool
    use_cached_build : bool
        Whether to use a cached build if it exists. If True, will only build if the executable does not exist.
    """

    uproject_file: Path
    build_dir: Path
    ubt_path: Path | None

    def __init__(
        self,
        uproject_path: Path | str,
        build_dir: Path | str | None = None,
        ubt_path: Path | str | None = None,
        use_cached_build: bool = False,
        headless_mode: bool = False,
        map: str | None = None,
        display_logs: bool = True,
        set_fps: int | None = None,
        disable_script: bool = True,
        extra_executable_args: list[str] | None = None,
        extra_ubt_args: dict[str, Any] | None = None,
    ):
        self.use_cached_build = use_cached_build
        self.extra_executable_args = extra_executable_args
        self.extra_ubt_args = extra_ubt_args
        # Convert to Path and resolve
        if isinstance(uproject_path, str):
            uproject_path = Path(uproject_path)

        self.uproject_file, self.build_dir, self.ubt_path, self.executable_path = (
            resolve_build_arguments(uproject_path, build_dir, ubt_path)
        )
        executable_path = get_executable_path(self.build_dir, self.uproject_file.stem)
        # Determine the expected executable path
        # Warn the user if map doesn't start with /Game/
        if map is not None:
            map = try_and_resolve_map(map)

        if not use_cached_build or not executable_path.exists():
            # Either we are forcing a rebuild or the executable doesn't exist
            build_project(
                uproject_path=self.uproject_file,
                build_dir=self.build_dir,
                ubt_path=self.ubt_path,
                map_path=map,
                extra_ubt_args=extra_ubt_args,
            )

        super().__init__(
            executable_path=executable_path,
            headless_mode=headless_mode,
            map=map,  # this sets self.map
            display_logs=display_logs,
            set_fps=set_fps,
            disable_script=disable_script,
            extra_args=extra_executable_args,
        )

    @property
    def project_name(self) -> str:
        """
        Project name derived from the ``.uproject`` filename (stem).

        Returns
        -------
        str
            Name used for build outputs and executable discovery.
        """
        return self.uproject_file.stem

    def get_simulator_args(self) -> dict[str, Any]:
        return {
            "uproject_path": self.uproject_file,
            "build_dir": self.build_dir,
            "ubt_path": self.ubt_path,
            "use_cached_build": self.use_cached_build,
            "headless_mode": self.headless_mode,
            "map": self.map,
            "display_logs": self.display_logs,
            "set_fps": self.set_fps,
            "disable_script": self.disable_script,
            "extra_executable_args": self.extra_executable_args,
            "extra_ubt_args": self.extra_ubt_args,
        }
