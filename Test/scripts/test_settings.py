# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for ``schola.scripts.common.settings``."""

from pathlib import Path

from schola.scripts.common.settings import (
    EnvironmentSettings,
    GrpcProtocolConfig,
    UnrealExecutableSimulatorConfig,
)


# ---- EnvironmentSettings.make_env_config tests -----------------------------


def test_make_env_config_defaults_to_external_simulator():
    """Default simulator settings yield an ExternalSimulator gRPC config that
    carries the protocol address and env options."""
    from schola.core.protocols.protobuf.grpc_protocol import GrpcProtocol
    from schola.core.simulators.external_simulator import ExternalSimulator

    env = EnvironmentSettings(
        protocol_settings=GrpcProtocolConfig(url="localhost", port=1),
        env_options={"k": "v"},
    )

    cfg = env.make_env_config()

    assert cfg["protocol"] is GrpcProtocol
    assert cfg["simulator"] is ExternalSimulator
    assert cfg["protocol_args"]["url"] == "localhost"
    assert cfg["protocol_args"]["port"] == 1
    assert cfg["options"] == {"k": "v"}


def test_make_env_config_uses_executable_simulator(tmp_path: Path):
    """An executable simulator config serializes via ``get_executable_args``
    (renamed kwargs + ``validate_path=False`` for remote reconstruction)."""
    from schola.core.simulators.unreal.executable_simulator import UnrealExecutable

    exe = tmp_path / "game.exe"
    exe.write_text("")
    env = EnvironmentSettings(
        simulator_settings=UnrealExecutableSimulatorConfig(executable_path=exe),
    )

    cfg = env.make_env_config()

    assert cfg["simulator"] is UnrealExecutable
    assert cfg["simulator_args"]["executable_path"] == exe
    assert cfg["simulator_args"]["validate_path"] is False


def test_make_env_config_reuses_passed_simulator(mocker):
    """A caller-supplied simulator is serialized directly without constructing a
    second one. Guards the train double-build regression: training builds its
    simulator once for space discovery and hands it to ``make_env_config``."""
    env = EnvironmentSettings(
        protocol_settings=GrpcProtocolConfig(url="localhost", port=1),
    )
    prebuilt = env.simulator_settings.make()
    spy = mocker.spy(env.simulator_settings, "make")

    cfg = env.make_env_config(prebuilt)

    spy.assert_not_called()
    assert cfg["simulator_args"] == prebuilt.get_simulator_args()
