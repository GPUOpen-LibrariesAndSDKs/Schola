# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Base class for connections that use the gRPC server.
"""

from typing import Any, Literal

import grpc

from schola.core.protocols.base_protocol import (
    AutoResetType,
    DEFAULT_AUTO_RESET_TYPE,
    BaseRLProtocol,
)
from schola.core.protocols.protobuf.deserialize import from_proto
from schola.core.protocols.protobuf.serialize import to_proto, fill_generic
import schola.generated.GymConnector_pb2_grpc as gym_grpc
import schola.generated.GymConnector_pb2 as util_messages
import schola.generated.Definitions_pb2 as env_definitions
import schola.generated.State_pb2 as state
import schola.generated.StateUpdates_pb2 as state_updates
from schola.core.protocols.socket_protocol import SocketProtocolMixin
import gymnasium as gym
import logging

logger = logging.getLogger(__name__)


class BaseGrpcProtocol(SocketProtocolMixin):
    """
    Base class for gym gRPC wiring shared by sync and async clients.

    Notes
    -----
    Subclasses implement ``channel_connected`` and own the concrete ``grpc`` channel type.
    """

    VALID_CREDENTIAL_MODES = ("local", "insecure")

    def __init__(
        self,
        url: str,
        port: int | None = None,
        environment_start_timeout: int | None = 45,
        credential_mode: Literal["local", "insecure"] = "local",
    ):
        super().__init__(url, port, client_only=(credential_mode == "insecure"))
        if credential_mode not in self.VALID_CREDENTIAL_MODES:
            raise ValueError(
                f"credential_mode must be one of {self.VALID_CREDENTIAL_MODES}, "
                f"got {credential_mode!r}"
            )
        self._gym_stub: gym_grpc.GymServiceStub | None = None
        self.environment_start_timeout = environment_start_timeout
        self.credential_mode = credential_mode

    @property
    def gym_stub(self) -> gym_grpc.GymServiceStub:
        assert self._gym_stub is not None, "gRPC stub is not initialized"
        return self._gym_stub

    def prepare_start_msg(
        self,
        auto_reset_type: AutoResetType,
    ) -> util_messages.GymConnectorStartRequest:
        start_msg = util_messages.GymConnectorStartRequest()

        if auto_reset_type == AutoResetType.DISABLED:
            start_msg.autoreset_type = util_messages.AutoResetType.DISABLED
        elif auto_reset_type == AutoResetType.SAME_STEP:
            start_msg.autoreset_type = util_messages.AutoResetType.SAME_STEP
        elif auto_reset_type == AutoResetType.NEXT_STEP:
            start_msg.autoreset_type = util_messages.AutoResetType.NEXT_STEP

        return start_msg

    def prepare_reset_msg(
        self,
        seeds: list[Any] | None = None,
        options: list[Any] | None = None,
    ) -> state_updates.StateUpdate:
        state_update = state_updates.StateUpdate(reset=state_updates.Reset())
        reset_msg: state_updates.Reset = state_update.reset

        if seeds is not None:
            for env_id, seed in enumerate(seeds):
                if seed is not None:
                    reset_msg.environments[env_id].seed = seed

        if options is not None:
            for env_id, option_dict in enumerate(options):
                for key, option in option_dict.items():
                    reset_msg.environments[env_id].options[key] = str(option)
        return state_update

    def prepare_action_msg(
        self,
        actions: dict[int, dict[str, Any]],
        action_space: dict[str, gym.Space[Any]],
    ) -> state_updates.StateUpdate:
        state_update = state_updates.StateUpdate(step=state_updates.Step())
        state_update.status = state_updates.CommunicatorStatus.GOOD

        for env_id in actions:
            env_update = state_update.step.environments.add()
            for agent_id, action in actions[env_id].items():
                fill_generic(
                    to_proto(action_space[agent_id], action),
                    env_update.updates[agent_id],
                )
        return state_update

    @property
    def channel_connected(self) -> bool:
        raise NotImplementedError(
            "channel_connected is not implemented for BaseGrpcProtocol"
        )

    def __bool__(self) -> bool:
        """
        Returns whether the connection is active or not

        Returns
        -------
        bool
            True iff the connection is active
        """
        return (self.has_socket or self.is_started) and self.channel_connected

    @property
    def properties(self) -> dict[str, Any]:
        return self.mixin_properties


class GrpcProtocol(BaseGrpcProtocol, BaseRLProtocol):
    """
    Synchronous ``BaseRLProtocol`` over gRPC.

    See Also
    --------
    BaseGrpcProtocol
    BaseRLProtocol
    """

    def __init__(
        self,
        url: str,
        port: int | None = None,
        environment_start_timeout: int | None = 45,
        credential_mode: Literal["local", "insecure"] = "local",
        grpc_close_timeout: float = 5.0,
    ):
        super().__init__(url, port, environment_start_timeout, credential_mode)
        self.grpc_close_timeout = grpc_close_timeout
        self.channel: grpc.Channel | None = None

    def close(self) -> None:
        """
        Close the Unreal Connection. Method must be safe to call multiple times.
        """
        logger.debug("Close invoked")
        SocketProtocolMixin.on_close(self)

        if self.channel is not None:
            try:
                state_update = state_updates.StateUpdate(
                    status=state_updates.CommunicatorStatus.CLOSED
                )
                self.gym_stub.UpdateState(state_update, timeout=self.grpc_close_timeout)
            except grpc.RpcError as e:
                logger.debug(
                    "gRPC close message failed (connection may already be closed): %s",
                    e,
                )
            finally:
                self.channel.close()
                self.channel = None
                self._gym_stub = None
        else:
            logger.debug("gRPC channel already closed")

    def start(self) -> None:
        """
        Open the Connection to Unreal Engine.
        """
        SocketProtocolMixin.on_start(self)

        options = [
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ]

        if self.credential_mode == "insecure":
            self.channel = grpc.insecure_channel(
                self.address, options=options
            ).__enter__()
        else:
            self.channel = grpc.secure_channel(
                self.address, grpc.local_channel_credentials(), options=options
            ).__enter__()
        self._gym_stub = gym_grpc.GymServiceStub(self.channel)

    def send_startup_msg(
        self,
        auto_reset_type: AutoResetType = DEFAULT_AUTO_RESET_TYPE,
    ) -> None:
        start_msg = self.prepare_start_msg(auto_reset_type)
        self.gym_stub.StartGymConnector(
            start_msg, timeout=self.environment_start_timeout, wait_for_ready=True
        )

    def get_definition(
        self,
    ) -> tuple[
        list[list[str]],
        dict[int, dict[str, str]],
        dict[int, dict[str, gym.Space[Any]]],
        dict[int, dict[str, gym.Space[Any]]],
    ]:
        training_defn: env_definitions.TrainingDefinition = (
            self.gym_stub.RequestTrainingDefinition(
                util_messages.TrainingDefinitionRequest()
            )
        )

        uids, agent_types, obs_spaces, act_spaces = from_proto(training_defn)

        return uids, agent_types, obs_spaces, act_spaces

    def send_reset_msg(
        self,
        seeds: list[Any] | None = None,
        options: list[Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        # abort any inprogress stuff
        state_update = self.prepare_reset_msg(seeds, options)

        response: state.State = self.gym_stub.UpdateState(state_update)
        obs, info = from_proto(response.initial_state)
        # Convert from Dict[Dict[envID, Dict[agentID, Any]]] to list[Dict[agentID, Any]]
        observations = [obs[env_id] for env_id in range(len(obs))]
        infos = [info[env_id] for env_id in range(len(info))]
        return observations, infos

    def send_action_msg(
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
        state_update = self.prepare_action_msg(actions, action_space)

        training_state: state.State = self.gym_stub.UpdateState(state_update)
        observations, rewards, terminateds, truncateds, infos = from_proto(
            training_state.training_state
        )

        if training_state.HasField("initial_state"):
            initial_obs, initial_info = from_proto(training_state.initial_state)
        else:
            initial_obs, initial_info = {}, {}

        return (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
            initial_obs,
            initial_info,
        )

    @property
    def channel_connected(self) -> bool:
        """
        Returns whether the connection is active or not

        Returns
        -------
        bool
            Whether the connection is active or not
        """
        return self.channel is not None
