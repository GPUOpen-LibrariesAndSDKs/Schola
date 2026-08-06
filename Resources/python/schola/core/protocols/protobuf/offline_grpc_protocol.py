# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Base class for connections that use the gRPC server (imitation / offline).
"""

from typing import Any

import grpc

from schola.generated.Definitions_pb2 import TrainingDefinition
from schola.core.protocols.base_protocol import BaseImitationProtocol
from schola.core.protocols.protobuf.deserialize import from_proto
import schola.generated.ImitationConnector_pb2_grpc as imitation_grpc
import schola.generated.ImitationConnector_pb2 as imitation_messages
import schola.generated.ImitationState_pb2 as imitation_state_messages
from schola.core.protocols.socket_protocol import SocketProtocolMixin
import gymnasium as gym
import logging

logger = logging.getLogger(__name__)


class GrpcImitationProtocol(BaseImitationProtocol, SocketProtocolMixin):
    """
    gRPC client for imitation / offline data exchange with Unreal.

    See Also
    --------
    BaseImitationProtocol
    """

    def __init__(
        self, url: str, port: int | None = None, protocol_start_timeout: int = 45
    ):
        super().__init__(url, port)
        self.channel: grpc.Channel | None = None
        self._stub: imitation_grpc.ImitationConnectorServiceStub | None = None
        self.protocol_start_timeout = protocol_start_timeout

    @property
    def stub(self) -> imitation_grpc.ImitationConnectorServiceStub:
        assert self._stub is not None, "gRPC stub is not initialized"
        return self._stub

    def close(self) -> None:
        """
        Close the Unreal Connection. Method must be safe to call multiple times.
        """
        logger.info("... Close invoked")
        SocketProtocolMixin.on_close(self)

        if self.channel is not None:
            self.channel.close()
            self.channel = None
            self._stub = None
        else:
            logger.info("... gRPC channel already closed?")

    def start(self) -> None:
        """
        Open the Connection to Unreal Engine.
        """
        SocketProtocolMixin.on_start(self)

        self.channel = grpc.secure_channel(
            self.address, grpc.local_channel_credentials()
        ).__enter__()
        self._stub = imitation_grpc.ImitationConnectorServiceStub(self.channel)

    def send_startup_msg(
        self,
        seeds: list[int | None] | None = None,
        options: list[dict[str, Any]] | None = None,
    ) -> None:
        start_msg = imitation_messages.ImitationConnectorStartRequest()

        if seeds is not None or options is not None:
            resolved_seeds: list[int | None]
            resolved_options: list[dict[str, Any]]
            if seeds is None:
                resolved_seeds = [None] * len(options or [])
            else:
                resolved_seeds = seeds
            if options is None:
                resolved_options = [{} for _ in range(len(resolved_seeds))]
            else:
                resolved_options = options

            # environments is a map, so we populate it like a dictionary
            for env_id, (seed, option_dict) in enumerate(
                zip(resolved_seeds, resolved_options)
            ):
                env_settings = start_msg.environments[env_id]
                if seed is not None:
                    env_settings.seed = seed
                if option_dict:
                    for key, value in option_dict.items():
                        env_settings.options[key] = str(value)

        self.stub.StartImitationConnector(
            start_msg, timeout=self.protocol_start_timeout, wait_for_ready=True
        )

    def get_definition(
        self,
    ) -> tuple[
        list[list[str]],
        list[dict[str, str]],
        dict[int, dict[str, gym.Space[Any]]],
        dict[int, dict[str, gym.Space[Any]]],
    ]:
        definition: TrainingDefinition = self.stub.RequestTrainingDefinition(
            imitation_messages.ImitationDefinitionRequest()
        )
        return from_proto(definition)

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
        data_request = imitation_messages.ImitationStateRequest()
        data: imitation_state_messages.ImitationState = self.stub.RequestState(
            data_request
        )
        return from_proto(data)

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
        return self.has_socket and self.channel_connected

    @property
    def properties(self) -> dict[str, Any]:
        return self.mixin_properties
