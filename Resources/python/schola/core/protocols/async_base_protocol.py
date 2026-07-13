# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Asyncio versions of the base protocol classes for Unreal Connections.
"""

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym

from .base_protocol import AutoResetType, DEFAULT_AUTO_RESET_TYPE


class AsyncBaseProtocol(ABC):
    """
    Async base class for all communication protocols with Schola.

    This abstract class defines the async interface for communication protocols
    used to connect Python environments with simulations.
    """

    @abstractmethod
    async def close(self) -> None:
        """
        Close the protocol connection asynchronously.

        Notes
        -----
        This method should be safe to call multiple times.
        """
        ...

    @abstractmethod
    async def start(self) -> None:
        """
        Start the protocol connection asynchronously.

        Initializes and opens the connection to the Unreal Engine.
        """
        ...

    @abstractmethod
    def __bool__(self) -> bool:
        """
        Returns whether the connection is active or not.

        Returns
        -------
        bool
            True iff the connection is active
        """
        ...

    @abstractmethod
    async def send_startup_msg(
        self,
        auto_reset_type: AutoResetType | None = None,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Send the initial startup message to Unreal Engine.

        Parameters
        ----------
        auto_reset_type : AutoResetType, optional
            Auto-reset behavior when supported by the concrete protocol.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        ...

    @abstractmethod
    async def get_definition(self, *args: Any, **kwargs: Any) -> Any:
        """
        Get the environment definition from Unreal Engine.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        Any
            The environment definition containing information about agents,
            observation spaces, and action spaces.
        """
        ...

    @property
    def properties(self) -> dict[str, Any]:
        """
        Get protocol-specific properties.

        Returns
        -------
        Dict[str, Any]
            A dictionary of protocol properties that can be passed to simulators.
        """
        return dict()


class AsyncBaseProtocolMixin:
    """
    Mixin class for async protocol implementations.

    This class provides additional async hooks that can be mixed into
    async protocol implementations via multiple inheritance.
    """

    async def on_close(self) -> None:
        """
        Hook called when the protocol is being closed.

        Override this method to perform cleanup specific to the mixin.
        """
        ...

    async def on_start(self) -> None:
        """
        Hook called when the protocol is starting.

        Override this method to perform initialization specific to the mixin.
        """
        ...

    @property
    def mixin_properties(self) -> dict[str, Any]:
        """
        Get mixin-specific properties.

        Returns
        -------
        Dict[str, Any]
            A dictionary of properties provided by this mixin.
        """
        return dict()


class AsyncBaseRLProtocol(AsyncBaseProtocol, ABC):
    """
    Async base class for reinforcement learning protocols.

    This class extends AsyncBaseProtocol with async methods specific to RL
    environments, including reset, step, and action messaging.
    """

    @abstractmethod
    async def send_startup_msg(
        self,
        auto_reset_type: AutoResetType = DEFAULT_AUTO_RESET_TYPE,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Send the startup message with auto-reset configuration.

        Parameters
        ----------
        auto_reset_type : AutoResetType, default=AutoResetType.SAME_STEP
            The type of auto-reset behavior to use when episodes end.
        """
        ...

    @abstractmethod
    async def get_definition(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[
        list[list[str]],
        dict[int, dict[str, str]],
        dict[int, dict[str, gym.Space[Any]]],
        dict[int, dict[str, gym.Space[Any]]],
    ]:
        """
        Get the environment definition from Unreal Engine.

        Returns
        -------
        Tuple[List[List[str]], Dict[int, Dict[str, str]], Dict[int, Dict[str, gym.Space]], Dict[int, Dict[str, gym.Space]]]
            A tuple containing:
            - List of agent IDs per environment
            - Agent types indexed by environment and agent
            - Observation spaces for each environment and agent
            - Action spaces for each environment and agent
        """
        ...

    @abstractmethod
    async def send_reset_msg(
        self,
        seeds: list[Any] | None = None,
        options: list[Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        """
        Send a reset message to restart the environment.

        Parameters
        ----------
        seeds : List, optional
            List of random seeds for each environment.
        options : List, optional
            List of reset options for each environment.

        Returns
        -------
        Tuple[List[Dict[str, Any]], List[Dict[str, str]]]
            A tuple containing:
            - List of initial observations for each environment
            - List of initial info dicts for each environment
        """
        ...

    @abstractmethod
    async def send_action_msg(
        self,
        actions: dict[int, dict[str, Any]],
        action_space: dict[str, gym.Space[Any]],
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, float]],
        list[dict[str, bool]],
        list[dict[str, bool]],
        list[dict[str, dict[str, str]]],
        dict[int, dict[str, Any]],
        dict[int, dict[str, str]],
    ]:
        """
        Send actions to the environment and receive the next state.

        Parameters
        ----------
        actions : Dict[int, Dict[str, Any]]
            Actions to take, indexed by environment ID and agent ID.
        action_space : Dict[str, gym.Space]
            The action spaces used to serialize the actions.

        Returns
        -------
        Tuple[List[Dict[str, Any]], List[Dict[str, float]], List[Dict[str, bool]], List[Dict[str, bool]], List[Dict[str, Dict[str, str]]], Dict[int, Dict[str, Any]], Dict[int, Dict[str, str]]]
            A tuple containing:
            - Observations for each environment
            - Rewards for each environment
            - Termination flags for each environment
            - Truncation flags for each environment
            - Info dicts for each environment
            - Initial observations if auto-reset occurred
            - Initial info dicts if auto-reset occurred
        """
        ...
