# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for the ``schola env inspect`` CLI."""

import logging

import pytest
from cyclopts import App

from schola.scripts.common.settings import (
    EnvironmentSettings,
    ExternalSimulatorConfig,
    GrpcProtocolConfig,
)
from schola.scripts.env.inspect.inspect import EnvInspectCommand, main
from schola.scripts.env.settings import EnvInspectScriptSettings, EnvToolsEnvironmentSettings


@pytest.fixture
def mock_main(mocker):
    return mocker.patch("schola.scripts.env.inspect.inspect.main")


@pytest.fixture
def mock_inspect_app(mock_main):
    base = App(
        name="inspect",
        help="Start a Schola environment and report agent definitions plus one reset.",
    )
    logger = logging.getLogger(__name__)

    class TestEnvInspectCommand(EnvInspectCommand):
        @property
        def main_func(self):
            return mock_main

    return TestEnvInspectCommand(base, logger).make()


def test_inspect_cli_default_external(mock_inspect_app, mock_main):
    mock_inspect_app(
        ["external"],
        result_action="return_value",
        exit_on_error=False,
    )
    mock_main.assert_called_once()
    args = mock_main.call_args[0][0]
    assert isinstance(args, EnvInspectScriptSettings)
    assert isinstance(args.environment_settings.simulator_settings, ExternalSimulatorConfig)
    assert args.environment_settings.simulator_settings.num_simulators == 1
    assert args.logging_settings.schola_verbosity == 0


def test_inspect_cli_env_options_and_seed(mock_inspect_app, mock_main):
    mock_inspect_app(
        [
            "external",
            "--seed",
            "42",
            "--env-options.level",
            "1",
            "--env-options.curriculum",
            "easy",
        ],
        result_action="return_value",
        exit_on_error=False,
    )
    args = mock_main.call_args[0][0]
    assert args.environment_settings.seed == 42
    assert args.environment_settings.env_options == {
        "level": "1",
        "curriculum": "easy",
    }


def test_inspect_cli_protocol_port(mock_inspect_app, mock_main):
    mock_inspect_app(
        ["external", "-p", "7777"],
        result_action="return_value",
        exit_on_error=False,
    )
    args = mock_main.call_args[0][0]
    assert args.environment_settings.protocol_settings.port == 7777


def test_inspect_cli_accepts_n_flag(mock_inspect_app, mock_main):
    mock_inspect_app(
        ["external", "-n", "3"],
        result_action="return_value",
        exit_on_error=False,
    )
    args = mock_main.call_args[0][0]
    assert args.environment_settings.simulator_settings.num_simulators == 3


def test_inspect_main_warns_on_multiple_simulators(
    make_vec_env_server, caplog, mocker
):
    import gymnasium as gym

    mocker.patch(
        "schola.gym.env.GymVectorEnv",
        return_value=mocker.MagicMock(id_manager=mocker.MagicMock()),
    )
    mocker.patch("schola.scripts.env.inspect.inspect.inspect_agents")

    port = make_vec_env_server([gym.make("CartPole-v1")])
    args = EnvInspectScriptSettings(
        environment_settings=EnvToolsEnvironmentSettings(
            simulator_settings=ExternalSimulatorConfig(num_simulators=3),
            protocol_settings=GrpcProtocolConfig(port=port, url="localhost"),
        )
    )

    with caplog.at_level(logging.WARNING):
        main(args)

    assert "not supported for inspect" in caplog.text


def test_inspect_main_on_real_env(make_vec_env_server, caplog):
    import gymnasium as gym

    port = make_vec_env_server([gym.make("CartPole-v1")])
    args = EnvInspectScriptSettings(
        environment_settings=EnvToolsEnvironmentSettings(
            simulator_settings=ExternalSimulatorConfig(),
            protocol_settings=GrpcProtocolConfig(port=port, url="localhost"),
        )
    )

    with caplog.at_level(logging.INFO):
        main(args)

    messages = "\n".join(record.message for record in caplog.records)
    assert "Sub-environments: 1" in messages
    assert "Agent definitions:" in messages
    assert "Initial Obs in Space: True" in messages
