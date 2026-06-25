# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Base Class for Unreal Connections
"""

import sys
from typing import Any, Dict, List, Optional, Tuple, TypedDict
import grpc
import socket
import gymnasium as gym

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum


class AutoResetType(StrEnum):
    """
    Enum for Auto Reset Types.
    """

    DISABLED = "Disabled"
    SAME_STEP = "SameStep"
    NEXT_STEP = "NextStep"


# Type Defs


class BaseProtocol:
    """
    Base class for all communication protocols with Schola.

    This abstract class defines the basic interface for communication protocols
    used to connect Python environments with simulations.
    """

    def close(self) -> None:
        """
        Close the protocol connection.

        Notes
        -----
        This method should be safe to call multiple times.
        """
        ...

    def start(self) -> None:
        """
        Start the protocol connection.

        Initializes and opens the connection to the Unreal Engine.
        """
        ...

    def __bool__(self) -> bool:
        """
        Returns whether the connection is active or not

        Returns
        -------
        bool
            True iff the connection is active
        """
        ...

    def send_startup_msg(self, *args, **kwargs):
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
    def properties(self) -> Dict[str, Any]:
        """
        Get protocol-specific properties.

        Returns
        -------
        Dict[str, Any]
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
    def mixin_properties(self) -> Dict[str, Any]:
        """
        Get mixin-specific properties.

        Returns
        -------
        Dict[str, Any]
            A dictionary of properties provided by this mixin.
        """
        return dict()


class BaseRLProtocol(BaseProtocol):
    """
    Base class for reinforcement learning protocols.

    This class extends BaseProtocol with methods specific to RL environments,
    including reset, step, and action messaging.
    """

    def send_startup_msg(
        self, auto_reset_type: AutoResetType = AutoResetType.SAME_STEP
    ):
        """
        Send the startup message with auto-reset configuration.

        Parameters
        ----------
        auto_reset_type : AutoResetType, default=AutoResetType.SAME_STEP
            The type of auto-reset behavior to use when episodes end.
        """
        ...

    def get_definition(
        self,
    ) -> Tuple[
        List[List[str]],
        List[Dict[str, str]],
        Dict[int, Dict[str, gym.Space[Any]]],
        Dict[int, Dict[str, gym.Space[Any]]],
    ]:
        """
        Get the environment definition from Unreal Engine.

        Returns
        -------
        Tuple[List[List[str]], List[Dict[str, str]], Dict[int, Dict[str, gym.Space]], Dict[int, Dict[str, gym.Space]]]
            A tuple containing:
            - List of agent IDs per environment
            - List of agent groups per environment (used for grouping agents)
            - Observation spaces for each environment and agent
            - Action spaces for each environment and agent
        """
        ...

    def send_reset_msg(
        self, seeds: Optional[List[Any]] = None, options: Optional[List[Any]] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Dict[str, str]]]]:
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
        Tuple[List[Dict[str, Any]], List[Dict[str, Dict[str,str]]]]
            A tuple containing:
            - List of initial observations for each environment
            - List of initial info dicts for each environment
        """
        ...

    def send_action_msg(
        self, actions: Dict[int, Dict[str, Any]], action_space: Dict[str, gym.Space[Any]]
    ) -> Tuple[
        List[Dict[str, Any]],
        List[Dict[str, float]],
        List[Dict[str, bool]],
        List[Dict[str, bool]],
        List[Dict[str, str]],
        Dict[int, Dict[str, Any]],
        Dict[int, Dict[str, str]],
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
        Tuple[List[Dict[str,Any]], List[Dict[str,float]], List[Dict[str,bool]], List[Dict[str,bool]], List[Dict[str,str]], Dict[int,Dict[str, Any]], Dict[int,Dict[str, str]]]
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


class BaseImitationProtocol(BaseProtocol):
    """
    Base class for imitation learning protocols.

    This class extends BaseProtocol with methods specific to collecting
    demonstration data for imitation learning.

    Call GetDefinition to get the environment definition before calling any other methods.
    Call SendStartupMsg to start collecting data.
    """

    def send_startup_msg(
        self, seeds: Optional[List[Any]] = None, options: Optional[List[Any]] = None
    ):
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

    def get_definition(
        self,
    ) -> Tuple[
        List[List[str]],
        Dict[int, Dict[str, str]],
        Dict[int, Dict[str, gym.Space[Any]]],
        Dict[int, Dict[str, gym.Space[Any]]],
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

    def get_data(
        self,
    ) -> Tuple[
        List[Dict[str, Any]],
        List[float],
        List[Dict[str, bool]],
        List[Dict[str, bool]],
        List[Dict[str, str]],
        Dict[int, Dict[str, Any]],
        Dict[int, Dict[str, str]],
        Dict[int, Dict[str, Any]],
    ]:
        """
        Get demonstration data from the environment.

        Returns
        -------
        Tuple[List[Dict[str,Any]], List[float], List[Dict[str,bool]], List[Dict[str,bool]], List[Dict[str,str]], Dict[int,Dict[str, Any]], Dict[int,Dict[str, str]], Dict[int, Dict[str,Any]]]
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
