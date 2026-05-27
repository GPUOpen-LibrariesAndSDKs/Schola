# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Common utility functions and classes for use in Schola scripts.
"""

from enum import Enum
from typing import Annotated, Dict, Literal, Optional, Tuple, List, Type, Union

from dataclasses import dataclass, field
from cyclopts import App, Parameter, validators, group_extractors, Group, types
from pathlib import Path

from rich.console import Console

console = Console()


class ActivationFunctionEnum(str, Enum):
    """
    Activation functions for neural networks.
    """

    ReLU = "relu"  #: Rectified Linear Unit activation function.
    Sigmoid = "sigmoid"  #: Sigmoid activation function.
    TanH = "tanh"  #: Hyperbolic Tangent activation function.


def get_activation_function(activation: ActivationFunctionEnum) -> Type["torch.nn.Module"]:  # type: ignore
    """
    Get the PyTorch activation function class for the specified activation type.

    Parameters
    ----------
    activation : ActivationFunctionEnum
        The activation function type to retrieve.

    Returns
    -------
    Type[torch.nn.Module]
        The PyTorch activation function class (not an instance).

    Raises
    ------
    ValueError
        If the activation function type is not supported.

    Notes
    -----
    PyTorch is imported lazily within this function to avoid import overhead
    when not needed in cli scripts.
    """
    # we don't use a value on the ActivationFunctionEnum in order to lazily import torch only when needed
    from torch import nn as nn

    if activation == ActivationFunctionEnum.ReLU:
        return nn.ReLU6
    elif activation == ActivationFunctionEnum.Sigmoid:
        return nn.Sigmoid
    elif activation == ActivationFunctionEnum.TanH:
        return nn.Tanh
    else:
        raise ValueError(f"Unsupported activation function: {activation}")


@dataclass
class UnrealExecutableSimulatorConfig:
    """
    Arguments for the Unreal Engine executable simulator in Schola.

    This dataclass is used when you want to create a standalone Unreal Engine environment controlled
    by the Schola Python API.
    """

    executable_path: Annotated[
        Path,
        Parameter(
            validator=validators.Path(exists=True, file_okay=True, dir_okay=False)
        ),
    ]
    "Path to the standalone executable, when launching a standalone Environment must exist and be a file"

    disable_script: bool = True
    "Flag indicating if the autolaunch script setting in the Unreal Engine Schola Plugin should be disabled. Useful for testing."

    headless: Annotated[bool, Parameter(alias="-h")] = False
    "Flag indicating if the standalone Unreal Engine process should run in headless mode"

    map: Optional[str] = None
    "Map to load when launching a standalone Unreal Engine process"

    fps: Optional[int] = None
    "Fixed FPS to use when running standalone, if None no fixed timestep is used"

    display_logs: bool = True
    "Whether to render logs in a standalone window."

    num_simulators: Annotated[
        int, Parameter(validator=validators.Number(gte=1), alias="-n")
    ] = 1
    "Number of parallel simulator processes. Headless mode is recommended when N > 1. With a fixed port P, instances use P, P+1, ... P+N-1."

    def make(self):
        """
        Create an UnrealExecutable simulator instance with the specified settings.

        Returns
        -------
        UnrealExecutable
            A configured UnrealExecutable simulator instance.
        """
        from schola.core.simulators.unreal.executable_simulator import UnrealExecutable

        return UnrealExecutable(
            self.executable_path,
            self.headless,
            self.map,
            self.display_logs,
            self.fps,
            self.disable_script,
        )


@dataclass
class UnrealProjectSimulatorConfig:
    """
    Arguments for the Unreal Engine project simulator in Schola.
    """

    uproject_path: Annotated[
        Path,
        Parameter(
            validator=validators.Path(exists=True, file_okay=True, dir_okay=False)
        ),
    ]
    "Path to the .uproject file"

    build_dir: Optional[Path] = None
    "Directory to build the Unreal Engine project to, if None builds to Build/Staging in the project directory."

    ubt_path: Optional[Path] = None
    "Path to the Unreal Build Tool, if None will be automatically detected from the project directory."

    disable_script: bool = True
    "Flag indicating if the autolaunch script setting in the Unreal Engine Schola Plugin should be disabled. Useful for testing."

    headless: Annotated[bool, Parameter(alias="-h")] = False
    "Flag indicating if the standalone Unreal Engine process should run in headless mode"

    map: Optional[str] = None
    "Map to load when launching a standalone Unreal Engine process"

    fps: Optional[int] = None
    "Fixed FPS to use when running standalone, if None no fixed timestep is used"

    display_logs: bool = True
    "Whether to render logs in a standalone window."

    num_simulators: Annotated[
        int, Parameter(validator=validators.Number(gte=1), alias="-n")
    ] = 1
    "Number of parallel simulator processes. One project build is used; headless recommended when N > 1. With a fixed port P, instances use P, P+1, ... P+N-1."

    def make(self):
        """
        Create a UnrealProject simulator instance with the specified settings.

        Returns
        -------
        UnrealProject
            A configured UnrealProject simulator instance.
        """
        from schola.core.simulators.unreal.project_simulator import UnrealProject

        return UnrealProject(
            self.uproject_path,
            self.build_dir,
            self.ubt_path,
            use_cached_build=False,
            headless_mode=self.headless,
            map=self.map,
            display_logs=self.display_logs,
            set_fps=self.fps,
            disable_script=self.disable_script,
        )


@dataclass
class ExternalSimulatorConfig:
    """
    Arguments for an externally managed process.

    Use this when the game server is started outside of Python — for
    example as a Kubernetes pod/sidecar, a systemd service, or an existing Unreal Editor session.
    The simulator performs no process lifecycle management.
    """

    num_simulators: Annotated[
        int, Parameter(validator=validators.Number(gte=1), alias="-n")
    ] = 1
    "Number of externally managed simulator instances. Each instance is expected to be reachable at the protocol address with port offsets 0..N-1 (or a fixed port when ``port_offset_mode`` is ``fixed``)."

    readiness_timeout: Optional[int] = None
    "Seconds to wait for the external process to become reachable (reserved for future use)."

    def make(self):
        """
        Create an ExternalSimulator instance.

        Returns
        -------
        ExternalSimulator
            A configured ExternalSimulator instance.
        """
        from schola.core.simulators.external_simulator import ExternalSimulator

        return ExternalSimulator(readiness_timeout=self.readiness_timeout)


def protocol_port_for_index(base_port: Optional[int], index: int) -> Optional[int]:
    """
    Return the port for simulator index i when using a base port.

    If base_port is set, returns base_port + index; otherwise None (auto port).
    """
    if base_port is None:
        return None
    return base_port + index


class PortOffsetMode(str, Enum):
    """
    How gRPC ports are assigned to parallel simulator instances.
    """

    PER_WORKER = "per_worker"  #: Each worker adds its index to the base port (default, for single-host).
    FIXED = "fixed"  #: Every worker uses the same port (for K8s pods with isolated networks).


class CredentialMode(str, Enum):
    """
    gRPC channel credential strategy.
    """

    LOCAL = "local"  #: Same-machine credential check (default, uses ``grpc.local_channel_credentials``).
    INSECURE = (
        "insecure"  #: No authentication (suitable for trusted in-cluster connections).
    )


@dataclass
class GrpcProtocolConfig:
    """
    Settings for the gRPC protocol in Schola.
    """

    port: Annotated[Optional[int], Parameter(alias="-p")] = None
    "Port to connect to the Unreal Engine process, if None an open port will be automatically selected when running standalone. Port is required if connecting to an existing Unreal Engine process."

    url: Annotated[str, Parameter(alias="-u")] = "localhost"
    "URL to connect to the Unreal Engine process."

    environment_start_timeout: Optional[int] = 45
    "Timeout for waiting to see if the environment is ready before assuming it crashed, in seconds."

    port_offset_mode: PortOffsetMode = PortOffsetMode.PER_WORKER
    "How ports are assigned across parallel simulators. ``per_worker`` adds ``worker_index`` to the base port (single host). ``fixed`` uses the same port on every worker (Kubernetes pods with isolated networks)."

    credential_mode: CredentialMode = CredentialMode.LOCAL
    "gRPC channel credential strategy. ``local`` requires same-machine connections. ``insecure`` skips authentication (for trusted in-cluster or cross-container connections)."

    grpc_close_timeout: float = 5.0
    "Seconds to wait for the graceful shutdown of the protocol. Increase if shutdown is slow; decrease to fail faster when the peer is wedged (e.g. during error cleanup)."

    def make(self):
        """
        Create a GrpcProtocol instance with the specified settings.

        Returns
        -------
        GrpcProtocol
            A configured GrpcProtocol instance for communication with Unreal Engine.
        """
        from schola.core.protocols.protobuf.grpc_protocol import GrpcProtocol

        return GrpcProtocol(
            self.url,
            self.port,
            self.environment_start_timeout,
            credential_mode=self.credential_mode.value,
            grpc_close_timeout=self.grpc_close_timeout,
        )

    def make_n_async(self, n: int):
        """
        Create N AsyncGrpcProtocol instances with the specified settings.

        Returns
        -------
        List[AsyncGrpcProtocol]
            A list of configured async gRPC protocol instances.
        """
        from schola.core.protocols.protobuf.async_grpc_protocol import AsyncGrpcProtocol

        return [
            AsyncGrpcProtocol(
                self.url,
                protocol_port_for_index(self.port, i),
                self.environment_start_timeout,
                credential_mode=self.credential_mode.value,
            )
            for i in range(n)
        ]


IgnoreParameter = Parameter(show=False, parse=False)


@dataclass
class CheckpointSettings:
    """
    Settings for checkpoints in Schola.
    """

    enable_checkpoints: Annotated[bool, Parameter(alias="-c")] = False
    "Enable saving checkpoints"

    checkpoint_dir: types.Directory = Path("./ckpt")
    "Directory to save checkpoints to."

    save_freq: Annotated[int, Parameter(validator=validators.Number(gte=0))] = 100000
    "Frequency with which to save checkpoints."

    name_prefix_override: Optional[str] = None
    "Override the name prefix for the checkpoint files (e.g. SAC, PPO, etc.)"

    export_onnx: bool = False
    "Whether to export the model to ONNX format instead of just saving a checkpoint."

    save_final_policy: bool = False
    "Whether to save the final policy after training is complete."

    def __post_init__(self):
        if (
            self.enable_checkpoints or self.save_final_policy
        ) and not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class EnvironmentSettings:
    """
    Settings for the environment in Schola.
    """

    simulator_settings: Annotated[
        Union[
            UnrealExecutableSimulatorConfig,
            UnrealProjectSimulatorConfig,
            ExternalSimulatorConfig,
        ],
        IgnoreParameter,
    ] = field(default_factory=ExternalSimulatorConfig)
    "Settings for the simulator to use during training"

    protocol_settings: Annotated[
        GrpcProtocolConfig, Parameter(group="Protocol Arguments", name="*")
    ] = field(default_factory=GrpcProtocolConfig)
    "Settings for the protocol to use for communicating with the external simulator"

    env_options: Annotated[Dict[str, str], Parameter(group="Environment Arguments")] = (
        field(default_factory=dict)
    )
    "Key=value reset options forwarded to the simulator on the first env.reset(). Repeat the flag to set multiple keys, e.g. --env-options.level=1 --env-options.curriculum=easy."


@dataclass
class Sb3LauncherExtension:
    """
    Extension hooks for Stable-Baselines3 training CLIs.

    Subclasses override the hook methods to attach extra ``KVWriter`` instances
    or SB3 callbacks without modifying the core train script.

    Notes
    -----
    Default implementations return empty lists (no-op).
    """

    def get_extra_KVWriters(self) -> List["stable_baselines3.common.logger.KVWriter"]:  # type: ignore
        """
        Returns a list of additional KVWriter to add to the training loop.

        Returns
        -------
        List[KVWriter]
            A list of additional KVWriters to add to the training loop.
        """
        return []

    def get_extra_callbacks(self) -> List["stable_baselines3.common.callbacks.BaseCallback"]:  # type: ignore
        """
        Returns a list of additional callbacks to add to the training loop.

        Returns
        -------
        List[BaseCallback]
            A list of additional callbacks to add to the training loop.
        """
        return []


@dataclass
class RllibLauncherExtension:
    """
    Extension hooks for Ray RLlib training CLIs.

    Subclasses override ``get_extra_callbacks`` to register additional Tune or
    RLlib callbacks.

    Notes
    -----
    Default implementation returns an empty list (no-op).
    """

    def get_extra_callbacks(self) -> List["ray.tune.callback"]:
        """
        Returns a list of additional callbacks to add to the training loop.

        Returns
        -------
        List[Callback]
            A list of additional callbacks to add to the training loop.
        """
        return []
