# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Base Class for Unreal Connections
"""

from typing_extensions import override

from schola.core.protocols.async_base_protocol import AsyncBaseRLProtocol
from schola.core.protocols.base_protocol import BaseProtocol
from schola.core.protocols.protobuf.async_grpc_protocol import AsyncGrpcProtocol
from schola.core.protocols.protobuf.grpc_protocol import GrpcProtocol
from schola.core.simulators.base_simulator import BaseSimulator


class BaseUnrealSimulator(BaseSimulator):
    """
    Abstract Base Class for a gRPC based connection to Unreal Engine.

    Parameters
    ----------
    url : str
        The url to connect to

    port : int
        The port on that URL to connect to

    Attributes
    ----------
    url: str
        The URL to connect to
    port: int
        The port on the URL to connect to
    channel: grpc.Channel
        The channel connecting to Unreal Engine on the chosen address
    """

    def __init__(self): ...

    @property
    @override
    def supported_protocols(self) -> tuple[type[BaseProtocol], ...]:
        """
        Synchronous gRPC protocol types supported by Unreal simulators.

        Returns
        -------
        tuple of type
            Tuple containing :class:`~schola.core.protocols.protobuf.grpc_protocol.GrpcProtocol`.
        """
        return (GrpcProtocol,)

    @property
    @override
    def supported_async_protocols(self) -> tuple[type[AsyncBaseRLProtocol], ...]:
        """
        Asynchronous gRPC protocol types supported by Unreal simulators.

        Returns
        -------
        tuple of type
            Tuple containing :class:`~schola.core.protocols.protobuf.async_grpc_protocol.AsyncGrpcProtocol`.
        """
        return (AsyncGrpcProtocol,)
