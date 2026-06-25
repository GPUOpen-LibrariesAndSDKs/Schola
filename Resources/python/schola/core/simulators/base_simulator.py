# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Abstract simulator base types shared by simulator backends.
"""

from schola.core.protocols.async_base_protocol import AsyncBaseRLProtocol
from schola.core.protocols.base_protocol import BaseProtocol


class UnsupportedProtocolException(Exception):
    """
    Exception raised when a protocol is not supported by a simulator.

    Raised when the protocol is not listed in ``supported_protocols`` (sync) or
    ``supported_async_protocols`` (async).
    """

    pass


class BaseSimulator:
    """
    Base class for all simulators.

    This abstract class defines the interface for simulator implementations
    that manage simulation instances (e.g. Unreal Editor, standalone executable, etc.).
    """

    def start(self, _protocol_properties: dict[str, object]) -> None:
        """
        Start the Simulator.

        Parameters
        ----------
        _protocol_properties : dict[str, object]
            Protocol-specific properties to pass to the simulator at startup. Simulator is responsible for passing these. (e.g. Port)
        """
        ...

    def stop(self) -> None:
        """
        Stop the simulator.

        This method should safely shut down the simulator and clean up resources.
        """
        ...

    @property
    def supported_protocols(self) -> tuple[type[BaseProtocol], ...]:
        """
        Get the protocols supported by this simulator.

        Returns
        -------
        tuple[type[BaseProtocol], ...]
            A tuple of protocol classes that this simulator supports.
        """
        return tuple()

    @property
    def supported_async_protocols(self) -> tuple[type[AsyncBaseRLProtocol], ...]:
        """
        Async RL protocol classes this simulator supports (see ``AsyncBaseRLProtocol``).

        Returns
        -------
        tuple[type, ...]
            Tuple of concrete async protocol types; empty if the simulator has no async support.
        """
        return tuple()

    def __bool__(self) -> bool:
        """
        Check if the simulator is currently running.

        Returns
        -------
        bool
            True if the simulator is running, False otherwise.
        """
        ...
