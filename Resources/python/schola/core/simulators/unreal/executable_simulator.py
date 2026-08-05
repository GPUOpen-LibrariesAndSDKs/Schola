# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""
A connection that launches an Environment in a new Process.
"""

import logging
from pathlib import Path
import subprocess
import sys
from typing import Any


from schola.core.simulators.unreal.base_simulator import BaseUnrealSimulator

logger = logging.getLogger(__name__)

PROCESS_KILL_TIMEOUT = 1


class UnrealExecutable(BaseUnrealSimulator):
    """
    A standalone client that launches an Unreal Engine instance when started.

    Parameters
    ----------
    url : str
        The URL to connect to
    ue_path : str
        The path to the Unreal Engine executable
    headless_mode : bool
        Whether to run in headless mode
    map : str, optional
        The map to load. Defaults to the default map in the Unreal Engine project
    display_logs : bool, default=True
        Whether to display logs
    set_fps : int, optional
        Use a fixed fps while running, if None, no fixed timestep is used
    disable_script : bool, default=False
        Whether to disable the autolaunch script setting in the Unreal Engine Schola Plugin

    Attributes
    ----------
    executable_path : str
        The path to the Unreal Engine executable
    headless_mode : bool, default=False
        Whether to run in headless mode
    display_logs : bool
        Whether to display logs
    set_fps : int, optional
        Use a fixed fps while running, if None, no fixed timestep is used
    env_process : subprocess.Popen, optional
        The process running the Unreal Engine. None if the process is not running
    disable_script : bool
        Whether to disable the autolaunch script setting in the Unreal Engine Schola Plugin
    map : str, optional
        The map to load.  Defaults to the default map in the Unreal Engine project
    """

    executable_path: Path
    headless_mode: bool
    display_logs: bool
    set_fps: int | None
    env_process: subprocess.Popen[bytes] | None
    disable_script: bool
    map: str | None
    extra_args: list[str]

    def __init__(
        self,
        executable_path: str | Path,
        headless_mode: bool = False,
        map: str | None = None,
        display_logs: bool = True,
        set_fps: int | None = None,
        disable_script: bool = True,
        extra_args: list[str] | None = None,
        validate_path: bool = True,
    ):
        super().__init__()
        if isinstance(executable_path, str):
            executable_path = Path(executable_path)

        if validate_path and (
            not executable_path.exists() or not executable_path.is_file()
        ):
            raise FileNotFoundError(
                f"executable_path {executable_path} does not exist or is not a file"
            )
        self.executable_path = executable_path
        self.headless_mode = headless_mode
        self.display_logs = display_logs
        self.set_fps = set_fps
        self.env_process = None
        self.disable_script = disable_script
        # Note any maps we want to use here need to be added to the build via Project Settings>Packaging>Advanced> List of Maps...
        # or on the command line with the -Map flag for UnrealAutomationTool
        self.map = map
        self.extra_args = extra_args if extra_args is not None else []
        self.validate_path = validate_path

    def spawn(self, count: int = 1) -> list["UnrealExecutable"]:
        """
        Return new UnrealExecutable instances with the same launch settings as this instance.

        Use this to create additional simulator instances from the same executable
        without changing any launch options. None of the returned instances are started.

        Parameters
        ----------
        count : int, default=1
            Number of instances to spawn.

        Returns
        -------
        list[UnrealExecutable]
            New simulator instances with the same executable path and launch options.
        """
        return [UnrealExecutable(**self.get_spawn_args()) for _ in range(count)]

    def _get_executable_arg_dict(self, validate_path: bool = False) -> dict[str, Any]:
        return {
            "executable_path": self.executable_path,
            "headless_mode": self.headless_mode,
            "map": self.map,
            "display_logs": self.display_logs,
            "set_fps": self.set_fps,
            "disable_script": self.disable_script,
            "extra_args": self.extra_args.copy() if self.extra_args else None,
            "validate_path": validate_path,
        }

    def get_simulator_args(self) -> dict[str, Any]:
        """Return a dictionary of arguments used to create a new instance of this simulator via constructor."""
        return self._get_executable_arg_dict(validate_path=self.validate_path)

    def get_spawn_args(self) -> dict[str, Any]:
        """Return a dictionary of arguments used when spawning a new instance of this simulator via the spawn() method."""
        return self._get_executable_arg_dict(validate_path=False)

    def make_args(self) -> list[str]:
        """
        Make the arguments supplied to the Unreal Engine Executable.

        Returns
        -------
        List[str]
            The arguments to be supplied to the Unreal Engine Executable
        """

        args = [str(self.executable_path)]

        args.append("-UNATTENDED")
        if self.headless_mode:
            args += ["-nullRHI"]
        else:
            args += ["-WINDOWED"]

        if self.map:
            args += [self.map]
        if self.display_logs:
            args += ["-LOG"]
        if self.set_fps is not None:
            args += ["-BENCHMARK"]
            args += ["-FPS=" + str(self.set_fps)]
        if self.disable_script:
            args += ["-ScholaDisableScript"]
        args += self.extra_args
        return args

    def _resolve_executable_path(self) -> Path:
        """
        Validate and return the configured executable path.

        Returns
        -------
        Path
            The validated executable path.

        Raises
        ------
        FileNotFoundError
            If the path does not exist or is not a file.
        """
        if not self.executable_path.exists() or not self.executable_path.is_file():
            raise FileNotFoundError(
                f"executable_path {self.executable_path} does not exist or is not a file"
            )
        return self.executable_path

    def start(self, protocol_properties: dict[str, Any]) -> None:
        """
        Start the Unreal Engine process.

        Raises
        ------
        Exception
            If the subprocess is already running
        FileNotFoundError
            If the executable path cannot be resolved at launch time
        """
        if self.env_process is not None:
            raise RuntimeError("Subprocess already running")

        self.executable_path = self._resolve_executable_path()
        args = self.make_args()

        for key, value in protocol_properties.items():
            args += [f"-Schola{key}={value}"]

        self.env_process = subprocess.Popen(args)
        logger.info("Executable launched with PID: %s", self.env_process.pid)

    def stop(self) -> None:
        """
        Close the connection to the Unreal Engine. Kills the Unreal Engine process if it is running.
        """
        super().stop()

        if self.env_process is not None:
            logger.debug("Killing subprocess")
            self.env_process.kill()
            try:
                self.env_process.wait(timeout=PROCESS_KILL_TIMEOUT)
            except subprocess.TimeoutExpired:
                if self.env_process.poll() is None:
                    logger.warning(
                        "Subprocess.kill() failed, forcibly killing subprocess"
                    )
                    if sys.platform.startswith("win"):
                        subprocess.run(
                            f"TASKKILL /F /PID {self.env_process.pid} /T", check=False
                        )
                    else:
                        subprocess.run(
                            ["kill", "-9", str(self.env_process.pid)], check=False
                        )

    def __bool__(self) -> bool:
        # We have a process, and it hasn't completed yet
        return (self.env_process is not None) and (self.env_process.poll() is None)
