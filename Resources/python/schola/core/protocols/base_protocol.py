# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Base Class for Unreal Connections
"""

from __future__ import annotations

from collections.abc import Mapping
import sys
from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym

from schola.core.utils.dict_helpers import NestedDict

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum   # pyright: ignore[reportMissingImports]


class AutoResetType(StrEnum):
    """
    Enum for Auto Reset Types.
    """

    DISABLED = "Disabled"
    SAME_STEP = "SameStep"
    NEXT_STEP = "NextStep"


DEFAULT_AUTO_RESET_TYPE = AutoResetType("SameStep")


# Type Defs


class BaseProtocol(ABC):
    """
    Base class for all communication protocols with Schola.

    This abstract class defines the basic interface for communication protocols
    used to connect Python environments with simulations.
    """

    @abstractmethod
    def close(self) -> None:
        """
        Close the protocol connection.

        Notes
        -----
        This method should be safe to call multiple times.
        """
        ...

    @abstractmethod
    def start(self) -> None:
        """
        Start the protocol connection.

        Initializes and opens the connection to the Unreal Engine.
        """
        ...

    @abstractmethod
    def __bool__(self) -> bool:
        """
        Returns whether the connection is active or not

        Returns
        -------
        bool
            True iff the connection is active
        """
        ...

    @abstractmethod
    def send_startup_msg(self, *args: Any, **kwargs: Any) -> Any:
        """
        Send the initial startup message to Unreal Engine.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        ...

    @abstractmethod
    def get_definition(self, *args: Any, **kwargs: Any) -> Any:
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
        dict[str, Any]
            A dictionary of protocol properties that can be passed to simulators.
        """
        return dict()


class BaseProtocolMixin:
    """
    Mixin class for protocol implementations.

    This class provides additional functionality that can be mixed into
    protocol implementations via multiple inheritance.
    """

    def on_close(self) -> None:
        """
        Hook called when the protocol is being closed.

        Override this method to perform cleanup specific to the mixin.
        """
        ...

    def on_start(self) -> None:
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
        dict[str, Any]
            A dictionary of properties provided by this mixin.
        """
        return dict()


class BaseRLProtocol(BaseProtocol, ABC):
    """
    Base class for reinforcement learning protocols.

    This class extends BaseProtocol with methods specific to RL environments,
    including reset, step, and action messaging.
    """

    @abstractmethod
    def send_startup_msg(
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
    def get_definition(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[
        list[list[str]],
        list[dict[str, str]],
        dict[int, dict[str, gym.Space[Any]]],
        dict[int, dict[str, gym.Space[Any]]],
    ]:
        """
        Get the environment definition from Unreal Engine.

        Returns
        -------
        tuple[list[list[str]], list[dict[str, str]], dict[int, dict[str, gym.Space]], dict[int, dict[str, gym.Space]]]
            A tuple containing:
            - List of agent IDs per environment
            - Agent types indexed by environment and agent
            - Observation spaces for each environment and agent
            - Action spaces for each environment and agent
        """
        ...

    @abstractmethod
    def send_reset_msg(
        self, seeds: list[Any] | None = None, options: list[Any] | None = None
    ) -> tuple[list[dict[str, dict[str, Any]]], list[dict[str, dict[str, str]]]]:
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
        tuple[list[dict[str, dict[str, Any]]], list[dict[str, dict[str, str]]]]
            A tuple containing:
            - List of initial observations for each environment and agent
            - List of initial info dicts for each environment and agent
        """
        ...

    @abstractmethod
    def send_action_msg(
        self,
        actions: Mapping[int, NestedDict[str, Any]],
        action_space: dict[str, gym.Space[Any]],
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, float]],
        list[dict[str, bool]],
        list[dict[str, bool]],
        list[dict[str, dict[str, str]]],
        dict[int, dict[str, dict[str, Any]]],
        dict[int, dict[str, dict[str, str]]],
    ]:
        """
        Send actions to the environment and receive the next state.

        Parameters
        ----------
        actions : dict[int, dict[str, Any]]
            Actions to take, indexed by environment ID and agent ID.
        action_space : dict[str, gym.Space]
            The action spaces used to serialize the actions.

        Returns
        -------
        tuple[list[dict[str,Any]], list[dict[str,float]], list[dict[str,bool]], list[dict[str,bool]], list[dict[str,dict[str,str]]], dict[int,dict[str, Any]], dict[int,dict[str, str]]]
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


class BaseImitationProtocol(BaseProtocol, ABC):
    """
    Base class for imitation learning protocols.

    This class extends BaseProtocol with methods specific to collecting
    demonstration data for imitation learning.

    Call GetDefinition to get the environment definition before calling any other methods.
    Call SendStartupMsg to start collecting data.
    """

    @abstractmethod
    def send_startup_msg(
        self, seeds: list[int] | None = None, options: list[Any] | None = None
    ) -> Any:
        """
        Send the startup message for imitation learning data collection.

        Parameters
        ----------
        seeds : List, optional
            List of random seeds for each environment.
        options : List, optional
            List of startup options for each environment.

        Returns
        -------
        Any
            Defined by concrete protocol implementations (often ``None``).
        """
        ...

    @abstractmethod
    def get_definition(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[
        list[list[str]],
        list[dict[str, str]],
        dict[int, dict[str, gym.Space[Any]]],
        dict[int, dict[str, gym.Space[Any]]],
    ]:
        """
        Get the environment definition for imitation learning.

        Returns
        -------
        Tuple[List[List[str]], Dict[int, Dict[str, str]], Dict[int,Dict[str,gym.Space]], Dict[int,Dict[str,gym.Space]]]
            A tuple containing:
            - List of agent IDs per environment
            - Agent types indexed by environment and agent
            - Observation spaces indexed by environment and agent
            - Action spaces indexed by environment and agent
        """
        ...

    @abstractmethod
    def get_data(
        self,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, float]],
        list[dict[str, bool]],
        list[dict[str, bool]],
        list[dict[str, dict[str, str]]],
        dict[int, dict[str, dict[str, Any]]],
        dict[int, dict[str, dict[str, str]]],
        list[dict[str, Any]],
    ]:
        """
        Get demonstration data from the environment.

        Returns
        -------
        Tuple[List[Dict[str,Any]], List[Dict[str,float]], List[Dict[str,bool]], List[Dict[str,bool]], List[Dict[str,Dict[str,str]]], Dict[int,Dict[str, Any]], Dict[int,Dict[str, str]], List[Dict[str,Any]]]
            A tuple containing:
            - Observations for each timestep
            - Rewards for each timestep
            - Termination flags
            - Truncation flags
            - Info dicts
            - Initial agent observations
            - Initial agent info dicts
            - Demonstration actions
        """
        ...
