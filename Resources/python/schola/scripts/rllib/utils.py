# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Shared helper functions for the Schola RLlib scripts (train, eval, export)
"""

from __future__ import annotations

import signal
from typing import TYPE_CHECKING, Any, TypeVar

from schola.core.simulators import SupportsSpawn
from schola.core.simulators.base_simulator import BaseSimulator
from schola.scripts.common.settings import EnvironmentSettings
from schola.scripts.common.settings import BaseSimulatorConfig
from schola.scripts.rllib.settings import RllibEnvironmentSettings


def discover_env_metadata(
    environment_settings: RllibEnvironmentSettings,
    *,
    schola_verbosity: int = 0,
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    """Discover policy metadata by briefly standing up a temporary environment.

    Returns ``(agent_ids, agent_to_policy, env_config)``. On failure
    (including ``KeyboardInterrupt``) the protocol and simulator are released before
    the exception is re-raised so a launched Unreal process is never leaked.
    """
    from schola.rllib.env import RayVecEnv

    sim_args = environment_settings.simulator_settings
    protocol_args = environment_settings.protocol_settings

    primary_sim = sim_args.make()
    discovery_protocol = protocol_args.make()
    try:
        tmp_env = RayVecEnv(
            discovery_protocol,
            primary_sim,
            verbosity=schola_verbosity,
        )
        agent_ids = sorted(tmp_env.possible_agents)
        agent_to_policy = tmp_env.make_agent_to_policy()
        env_config = build_env_config(environment_settings, primary_sim)
    except (Exception, KeyboardInterrupt) as exc:
        if isinstance(exc, KeyboardInterrupt):
            import logging

            logging.getLogger(__name__).info(
                "Ctrl-C received. Shutting down gracefully;"
            )
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        discovery_protocol.close()
        primary_sim.stop()
        raise

    tmp_env.close()
    return agent_ids, agent_to_policy, env_config


def build_env_config(
    environment_settings: RllibEnvironmentSettings,
    simulator: BaseSimulator | None = None,
) -> dict[str, Any]:
    """
    Build the RLlib ``env_config`` dict consumed by ``ScholaEnvRunner.make_env``.

    Shared by the RLlib train and eval scripts so the protocol/simulator
    serialization (including the external-vs-executable simulator branch) lives
    in exactly one place.

    Parameters
    ----------
    environment_settings : EnvironmentSettings
        The CLI/script environment settings describing the protocol, simulator,
        and reset options to serialize.
    simulator : BaseSimulator, optional
        An already-constructed simulator to serialize. Callers that have built a
        simulator (e.g. training's space-discovery step) pass it so this helper
        stays side-effect-free and never builds a second time. When ``None`` a
        simulator is constructed from ``environment_settings.simulator_settings``.

    Returns
    -------
    dict[str, Any]
        The ``env_config`` consumed by ``ScholaEnvRunner.make_env``.
    """
    from schola.core.protocols.protobuf.grpc_protocol import GrpcProtocol
    from schola.core.simulators.external_simulator import ExternalSimulator
    from schola.core.simulators.unreal.executable_simulator import UnrealExecutable

    protocol_settings = environment_settings.protocol_settings
    primary_sim = (
        simulator
        if simulator is not None
        else environment_settings.simulator_settings.make()
    )

    return {
        "protocol": GrpcProtocol,
        "protocol_args": {
            "url": protocol_settings.url,
            "port": protocol_settings.port,
            "credential_mode": protocol_settings.credential_mode.value,
            "environment_start_timeout": protocol_settings.environment_start_timeout,
            "grpc_close_timeout": protocol_settings.grpc_close_timeout,
        },
        "port_offset_mode": protocol_settings.port_offset_mode.value,
        "simulator": environment_settings.simulator_settings.get_sim_cls(),
        "simulator_args": (
            primary_sim.get_spawn_args()
            if isinstance(primary_sim, SupportsSpawn)
            else primary_sim.get_simulator_args()
        ),
        "options": dict(environment_settings.env_options),
    }
