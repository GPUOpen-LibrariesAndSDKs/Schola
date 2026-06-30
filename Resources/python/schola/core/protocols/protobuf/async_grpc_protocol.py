# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Async gRPC protocol for non-blocking connections to the gRPC server.
"""

from typing import Any, Literal

import grpc
import grpc.aio
import gymnasium as gym
from gymnasium.vector.vector_env import AutoresetMode

from schola.core.protocols.base_protocol import AutoResetType, DEFAULT_AUTO_RESET_TYPE
from schola.core.protocols.async_base_protocol import AsyncBaseRLProtocol
import schola.generated.GymConnector_pb2 as util_messages
import schola.generated.GymConnector_pb2_grpc as gym_grpc
import schola.generated.State_pb2 as state
import schola.generated.StateUpdates_pb2 as state_updates
import logging
from schola.core.protocols.protobuf.deserialize import from_proto
from schola.core.protocols.protobuf.grpc_protocol import BaseGrpcProtocol
from schola.core.protocols.socket_protocol import SocketProtocolMixin

logger = logging.getLogger(__name__)


class AsyncGrpcProtocol(AsyncBaseRLProtocol, BaseGrpcProtocol):
    """
    Non-blocking gRPC client for the Schola gym connector.

    Notes
    -----
    Uses ``grpc.aio`` channels and stubs together with ``AsyncBaseRLProtocol``.
    """

    def __init__(
        self,
        url: str,
        port: int | None = None,
        environment_start_timeout: int | None = 45,
        credential_mode: Literal["local", "insecure"] = "local",
    ):
        super().__init__(url, port, environment_start_timeout, credential_mode)
        self.channel: grpc.aio.Channel | None = None

    async def close(self) -> None:
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
                await self.gym_stub.UpdateState(state_update)
            except grpc.RpcError as e:
                # Unreal may have already closed the connection, which is fine
                logger.debug(
                    "gRPC close message failed (connection may already be closed): %s",
                    e,
                )
            finally:
                await self.channel.close()
                self.channel = None
                self._gym_stub = None
        else:
            logger.debug("gRPC channel already closed")

    async def start(self) -> None:
        """
        Open the Connection to Unreal Engine.
        """
        SocketProtocolMixin.on_start(self)

        options = [
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ]

        if self.credential_mode == "insecure":
            self.channel = await grpc.aio.insecure_channel(
                self.address, options=options
            ).__aenter__()
        else:
            self.channel = await grpc.aio.secure_channel(
                self.address, grpc.local_channel_credentials(), options=options
            ).__aenter__()
        self._gym_stub = gym_grpc.GymServiceStub(self.channel)

    async def send_startup_msg(
        self,
        auto_reset_type: AutoResetType | AutoresetMode | int = DEFAULT_AUTO_RESET_TYPE,
    ) -> None:
        start_msg = self.prepare_start_msg(auto_reset_type)

        await self.gym_stub.StartGymConnector(
            start_msg, timeout=self.environment_start_timeout, wait_for_ready=True
        )

    async def get_definition(
        self,
    ) -> tuple[
        list[list[str]],
        dict[int, dict[str, str]],
        dict[int, dict[str, gym.Space[Any]]],
        dict[int, dict[str, gym.Space[Any]]],
    ]:
        training_defn = await self.gym_stub.RequestTrainingDefinition(
            util_messages.TrainingDefinitionRequest()
        )

        uids, agent_types, obs_spaces, act_spaces = from_proto(training_defn)

        return uids, agent_types, obs_spaces, act_spaces

    async def send_reset_msg(
        self,
        seeds: list[Any] | None = None,
        options: list[Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        # abort any inprogress stuff
        state_update = self.prepare_reset_msg(seeds, options)
        response: state.State = await self.gym_stub.UpdateState(state_update)
        obs, info = from_proto(response.initial_state)
        # Convert from Dict[Dict[envID, Dict[agentID, Any]]] to list[Dict[agentID, Any]]
        observations = [obs[env_id] for env_id in range(len(obs))]
        infos = [info[env_id] for env_id in range(len(info))]
        return observations, infos

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
        state_update = self.prepare_action_msg(actions, action_space)

        training_state: state.State = await self.gym_stub.UpdateState(state_update)
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
