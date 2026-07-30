# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""In-process Gymnasium simulator backed by a local gRPC server."""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Optional, Type

import grpc
import gymnasium as gym

from schola.core.protocols.async_base_protocol import AsyncBaseRLProtocol
from schola.core.protocols.base_protocol import BaseProtocol
from schola.core.simulators.base_simulator import BaseSimulator
from schola.core.simulators.gym.servicer import (
    GymToGymServiceServicer,
    VecGymToGymServiceServicer,
)
from schola.core.utils.shared_thread_pool_executor import SharedThreadPoolExecutor
import schola.generated.GymConnector_pb2_grpc as gym_grpc

logger = logging.getLogger(__name__)

_GRPC_OPTIONS = [
    ("grpc.max_send_message_length", 100 * 1024 * 1024),
    ("grpc.max_receive_message_length", 100 * 1024 * 1024),
]


class GymSimulator(BaseSimulator):
    """
    Launch a Gymnasium environment as an in-process gRPC server.

    Parameters
    ----------
    env_id : str
        Gymnasium environment identifier passed to ``gymnasium.make``.
    num_envs : int, default=1
        Number of parallel Gymnasium instances served by one gRPC connector.
        Values greater than one use the vector servicer implementation.
    wrappers : list[type[gym.Wrapper]], optional
        Gymnasium wrapper classes applied to each created environment.
    thread_pool : SharedThreadPool, optional
        Reference-counted executor passed to the gRPC server. When omitted, a
        dedicated pool is created in :meth:`start`. Use :meth:`spawn`
        to allocate a shared pool across multiple simulator instances.
    """

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        wrappers: list[type[gym.Wrapper]] | None = None,
        thread_pool: SharedThreadPoolExecutor | None = None,
    ):
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self.num_envs = num_envs
        self._wrappers: list[type[gym.Wrapper]] = wrappers if wrappers else []
        self._thread_pool = thread_pool
        self._server: grpc.Server | None = None
        self._servicer: GymToGymServiceServicer | VecGymToGymServiceServicer | None = (
            None
        )
        self.env_id = env_id

    def get_simulator_args(self) -> dict[str, Any]:
        """
        Return kwargs that reproduce this instance (e.g. for Ray ``env_config``).

        Returns
        -------
        dict[str, Any]
            Mapping suitable for ``GymSimulator(**args)``.
        """
        args: dict[str, Any] = {
            "env_id": self.env_id,
            "num_envs": self.num_envs,
        }
        args["wrappers"] = self._wrappers.copy()
        return args

    def spawn(self, count: int = 1) -> list[GymSimulator]:
        """
        Return additional GymSimulator instances that share launch settings.

        If this instance has no thread pool yet, a shared
        :class:`~schola.core.utils.shared_thread_pool_executor.SharedThreadPool` is
        created and assigned to this instance and every spawned clone. If a pool
        is already set, a new pool is created and distributed across the new instances.

        Parameters
        ----------
        count : int
            Number of additional simulator instances to create.

        Returns
        -------
        list[GymSimulator]
            New simulator instances (none started), excluding this instance.
        """
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        if count == 0:
            return []

        if self._thread_pool is None:
            total_workers = self.num_envs * (count + 1)
        else:
            total_workers = self.num_envs * (count)
        thread_pool: SharedThreadPoolExecutor = SharedThreadPoolExecutor(max_workers=total_workers)

        if self._thread_pool is None:
            self._thread_pool = thread_pool.share()
        return [
            GymSimulator(**self.get_simulator_args(), thread_pool=thread_pool.share())
            for _ in range(count)
        ]

    def get_spawn_args(self) -> dict[str, Any]:
        """Return a dictionary of arguments used to create a new instance of this simulator."""
        return self.get_simulator_args()

    def start(self, protocol_properties: dict[str, object]) -> None:
        """
        Start the in-process gRPC server on the port supplied by the protocol.

        Parameters
        ----------
        protocol_properties : dict[str, object]
            Must include ``Port`` with the listening TCP port.
        """
        if self._server is not None:
            raise RuntimeError("GymSimulator gRPC server is already running")

        port = int(protocol_properties["Port"])
        env_factory = functools.partial(gym.make, self.env_id)

        if self.num_envs == 1:
            servicer = GymToGymServiceServicer(env_factory, self._wrappers)
        else:
            env_factories = [env_factory for _ in range(self.num_envs)]
            servicer = VecGymToGymServiceServicer(env_factories, self._wrappers)

        if self._thread_pool is None:
            self._thread_pool = SharedThreadPoolExecutor(max_workers=self.num_envs).share()

        server = grpc.server(self._thread_pool, options=_GRPC_OPTIONS)  # type: ignore[arg-type]
        gym_grpc.add_GymServiceServicer_to_server(servicer, server)
        server.add_insecure_port(f"[::]:{port}")
        server.start()

        self._server = server
        self._servicer = servicer
        logger.info(
            "GymSimulator started gRPC server for %s (%d env(s)) on port %d",
            self.env_id,
            self.num_envs,
            port,
        )

    def stop(self) -> None:
        """Stop the gRPC server and release this instance's thread-pool reference."""
        if self._server is not None:
            logger.debug("Stopping GymSimulator gRPC server")
            self._server.stop(grace=2)
            if self._server.wait_for_termination(timeout=5):
                logger.warning("GymSimulator gRPC server did not terminate gracefully")
            self._server = None
            self._servicer = None

        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=False)
            self._thread_pool = None

    def __bool__(self) -> bool:
        return self._server is not None

    @property
    def supported_protocols(self) -> tuple[type[BaseProtocol], ...]:
        from schola.core.protocols.protobuf.grpc_protocol import GrpcProtocol

        return (GrpcProtocol,)

    @property
    def supported_async_protocols(self) -> tuple[type[AsyncBaseRLProtocol], ...]:
        from schola.core.protocols.protobuf.async_grpc_protocol import AsyncGrpcProtocol

        return (AsyncGrpcProtocol,)
