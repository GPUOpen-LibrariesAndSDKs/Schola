# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""
A connection builds an Unreal Project if Necessary and then launches a standalone Executable Connector.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from schola.core.simulators.unreal.executable_simulator import UnrealExecutable

logger = logging.getLogger(__name__)
import platform
import tempfile
from schola.core.utils.ubt import (
    UBTCommand,
    get_project_file,
    get_ue_version,
    get_ubt_path,
    get_unreal_platform,
    build_executable,
)


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
    build_dir : Optional[Path | str], default=None
        Directory where the built executable will be staged. If None, uses a temporary directory.
    ubt_path : Optional[Path | str], default=None
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

    def __init__(
        self,
        uproject_path: Path | str,
        build_dir: Optional[Path | str] = None,
        ubt_path: Optional[Path | str] = None,
        use_cached_build: bool = False,
        headless_mode: bool = False,
        map: Optional[str] = None,
        display_logs: bool = True,
        set_fps: Optional[int] = None,
        disable_script: bool = True,
        extra_executable_args: Optional[List[str]] = None,
        extra_ubt_args: Optional[Dict[str, Any]] = None,
    ):
        # Convert to Path and resolve
        if isinstance(uproject_path, str):
            uproject_path = Path(uproject_path)

        # If uproject_path is a directory, find the .uproject file in it
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
        self.uproject_file = uproject_file.resolve()

        # Set up build directory
        if build_dir is not None:
            self.build_dir = Path(build_dir).resolve()
        else:
            # Generate a cache folder with the build in temp directory
            self.build_dir = (
                Path(tempfile.gettempdir()) / f"schola_build_{self.project_name}"
            )
            logger.info(
                "No Build Directory Provided, Using temporary build directory: %s",
                self.build_dir,
            )

        self.build_dir.mkdir(parents=True, exist_ok=True)

        self.ubt_path = Path(ubt_path).resolve() if ubt_path is not None else None

        # Determine the expected executable path
        executable_filename = self.project_name + (
            ".exe" if platform.system() == "Windows" else ""
        )
        executable_path = (
            self.build_dir
            / platform.system()
            / self.project_name
            / "Binaries"
            / get_unreal_platform()
            / executable_filename
        )
        # Warn the user if map doesn't start with /Game/
        if map is not None:
            map = try_and_resolve_map(map)

        # Build if necessary
        if not use_cached_build or not executable_path.exists():
            # Either we are forcing a rebuild or the executable doesn't exist
            self._build(
                uproject_path=self.uproject_file,
                build_dir=self.build_dir,
                ubt_path=self.ubt_path,
                _map=map,
                extra_ubt_args=extra_ubt_args,
            )
            logger.info(f"Built executable to: {executable_path}")

        # Initialize parent UnrealExecutable with the built executable path
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

    def _build(
        self,
        uproject_path: Path,
        build_dir: Path,
        ubt_path: Optional[Path] = None,
        _map: Optional[str] = None,
        extra_ubt_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Build the Unreal project executable using the Unreal Build Tool.

        This is an internal method called during initialization if the executable
        doesn't exist or if use_cached_build is False.

        Parameters
        ----------
        uproject_path : Path
            Path to the .uproject file
        build_dir : Path
            Directory where the built executable will be staged
        ubt_path : Optional[Path], default=None
            Path to the Unreal Build Tool. If None, auto-detects from project.
        map: Optional[str], default=None
            The map to cook/package. If blank, cooks/packages all maps.
        Raises
        ------
        FileNotFoundError
            If the .uproject file or UBT path cannot be found
        ValueError
            If the UE version cannot be determined
        Exception
            If the build process fails
        """

        if ubt_path is None:
            ue_version = get_ue_version(uproject_path)
            if ue_version is None:
                raise ValueError(
                    "Could not determine Unreal Engine version from .uproject file"
                )
            project_folder = uproject_path.parent
            ubt_path = get_ubt_path(project_folder, ue_version)

        if ubt_path is None:
            raise FileNotFoundError(
                "Could not find Unreal Build Tool (UBT) path from project folder, passed ubt_path or default path."
            )
        ubt_args = extra_ubt_args.copy() if extra_ubt_args is not None else {}
        # override any weird duplicates in the extra_ubt_args
        ubt_args.update(
            {
                "ubt_path": ubt_path,
                "staging_dir": build_dir,
                "project_file": uproject_path,
                "maps": [_map] if _map is not None else [],
                "all_maps": False if _map is not None else True,
            }
        )
        ubt_command = UBTCommand(**ubt_args)
        # Build the executable
        completed_build_process = ubt_command.run()

        if completed_build_process.returncode != 0:
            exception_message = f"Unreal build failed with return code {completed_build_process.returncode} and the following output:\n"
            if completed_build_process.stderr:
                exception_message += f"stderr:\n {completed_build_process.stderr}\n"
            if completed_build_process.stdout:
                exception_message += f"stdout:\n {completed_build_process.stdout}\n"
            raise Exception(exception_message)
