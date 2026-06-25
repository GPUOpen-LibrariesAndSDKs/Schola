# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Simulator for externally managed Unreal Engine processes.

Use this when the UE game server is managed outside of Python — for example,
as a Kubernetes sidecar container, a separate Kubernetes service, or a
pre-started process on a remote host.  The simulator performs no process
lifecycle management: ``start`` and ``stop`` are no-ops.
"""

import logging

from typing_extensions import override

from schola.core.protocols.async_base_protocol import AsyncBaseRLProtocol
from schola.core.protocols.base_protocol import BaseProtocol
from schola.core.simulators.base_simulator import BaseSimulator

logger = logging.getLogger(__name__)


class ExternalSimulator(BaseSimulator):
    """
    A simulator stub for Unreal Engine instances managed by an external
    orchestrator (Kubernetes, systemd, manual launch, etc.).

    This class intentionally does **not** launch or stop any process.
    It exists so that the Schola protocol/simulator pairing works when the
    game server is already running and reachable over the network.

    Parameters
    ----------
    readiness_timeout : int, optional
        Reserved for future use.  Seconds to wait for the external process
        to become reachable before giving up.  Currently unused — readiness
        is assumed to be handled by the orchestrator (e.g. K8s readiness
        probes).
    """

    readiness_timeout: int | None

    def __init__(self, readiness_timeout: int | None = None):
        self.readiness_timeout = readiness_timeout

    def get_simulator_args(self) -> dict[str, int]:
        """
        Return kwargs that reproduce this instance (e.g. for Ray ``env_config``).

        Returns
        -------
        dict[str, int]
            Mapping suitable for ``ExternalSimulator(**args)``; includes
            ``readiness_timeout`` only when it is not ``None``.

        Notes
        -----
        Mirrors the executable simulator's argument helper so ``simulator_args``
        round-trips when the config is shipped to Ray env runners.
        """
        args: dict[str, int] = {}
        if self.readiness_timeout is not None:
            args["readiness_timeout"] = self.readiness_timeout
        return args

    @override
    def start(self, _protocol_properties: dict[str, object]) -> None:
        """No-op — the external orchestrator manages the process."""
        logger.debug(
            "ExternalSimulator.start() called (no-op); expecting UE to be reachable at the configured protocol address."
        )

    @override
    def stop(self) -> None:
        """No-op — we don't own the process, so we don't kill it."""
        logger.debug("ExternalSimulator.stop() called (no-op).")

    @override
    def __bool__(self) -> bool:
        """Assume the externally managed process is always running."""
        return True

    @property
    @override
    def supported_protocols(self) -> tuple[type[BaseProtocol], ...]:
        """
        Synchronous protocol implementations compatible with this simulator.

        Returns
        -------
        tuple of type
            Tuple containing :class:`~schola.core.protocols.protobuf.grpc_protocol.GrpcProtocol`.
        """
        from schola.core.protocols.protobuf.grpc_protocol import GrpcProtocol

        return (GrpcProtocol,)

    @property
    @override
    def supported_async_protocols(self) -> tuple[type[AsyncBaseRLProtocol], ...]:
        """
        Asynchronous protocol implementations compatible with this simulator.

        Returns
        -------
        tuple of type
            Tuple containing :class:`~schola.core.protocols.protobuf.async_grpc_protocol.AsyncGrpcProtocol`.
        """
        from schola.core.protocols.protobuf.async_grpc_protocol import AsyncGrpcProtocol

        return (AsyncGrpcProtocol,)
