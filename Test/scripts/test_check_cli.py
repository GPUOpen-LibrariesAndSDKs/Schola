# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for the ``schola env check`` CLI."""

import logging

import pytest
from cyclopts import App

from schola.scripts.common.settings import (
    EnvironmentSettings,
    ExternalSimulatorConfig,
    GrpcProtocolConfig,
    GymSimulatorConfig,
)
from schola.scripts.env.check.check import EnvCheckCommand, main
from schola.scripts.env.settings import EnvCheckScriptSettings, EnvToolsEnvironmentSettings


@pytest.fixture
def mock_main(mocker):
    return mocker.patch("schola.scripts.env.check.check.main")


@pytest.fixture
def mock_check_app(mock_main):
    base = App(
        name="check",
        help="Start a Schola environment and run Gymnasium's environment checker on it.",
    )
    logger = logging.getLogger(__name__)

    class TestEnvCheckCommand(EnvCheckCommand):
        @property
        def main_func(self):
            return mock_main

    return TestEnvCheckCommand(base, logger).make()


def test_check_cli_default_external(mock_check_app, mock_main):
    mock_check_app(
        ["external"],
        result_action="return_value",
        exit_on_error=False,
    )
    mock_main.assert_called_once()
    args = mock_main.call_args[0][0]
    assert isinstance(args, EnvCheckScriptSettings)
    assert isinstance(args.environment_settings.simulator_settings, ExternalSimulatorConfig)
    assert args.environment_settings.simulator_settings.num_simulators == 1
    assert args.logging_settings.schola_verbosity == 0


def test_check_cli_env_options_and_seed(mock_check_app, mock_main):
    mock_check_app(
        [
            "external",
            "--seed",
            "42",
            "--env-options.level",
            "1",
        ],
        result_action="return_value",
        exit_on_error=False,
    )
    args = mock_main.call_args[0][0]
    assert args.environment_settings.seed == 42
    assert args.environment_settings.env_options == {"level": "1"}


def test_check_main_warns_on_multiple_simulators(
    make_vec_env_server, caplog, mocker
):
    import gymnasium as gym

    mocker.patch(
        "schola.gym.env.GymEnv",
        return_value=mocker.MagicMock(
            action_space=mocker.MagicMock(),
            observation_space=mocker.MagicMock(),
        ),
    )
    mocker.patch("schola.scripts.env.check.check.run_gym_env_checker")

    port = make_vec_env_server([gym.make("CartPole-v1")])
    args = EnvCheckScriptSettings(
        environment_settings=EnvToolsEnvironmentSettings(
            simulator_settings=ExternalSimulatorConfig(num_simulators=3),
            protocol_settings=GrpcProtocolConfig(port=port, url="localhost"),
        )
    )

    with caplog.at_level(logging.WARNING):
        main(args)

    assert "not supported for check" in caplog.text


def test_check_main_on_real_env(caplog, mocker):
    mock_check_env = mocker.patch("schola.scripts.env.utils.check_env")

    args = EnvCheckScriptSettings(
        environment_settings=EnvToolsEnvironmentSettings(
            simulator_settings=GymSimulatorConfig(env_id="CartPole-v1", num_environments=1),
        )
    )

    with caplog.at_level(logging.INFO):
        main(args)

    mock_check_env.assert_called_once()
    assert mock_check_env.call_args.kwargs["skip_render_check"] is True
    messages = "\n".join(record.message for record in caplog.records)
    assert "Running Gymnasium environment checker" in messages
    assert "Environment checker passed." in messages


def test_check_main_gym_forces_single_simulator(mocker, caplog):
    mock_gym_env = mocker.MagicMock(
        action_space=mocker.MagicMock(),
        observation_space=mocker.MagicMock(),
    )
    mock_gym_env_cls = mocker.patch(
        "schola.gym.env.GymEnv", return_value=mock_gym_env
    )
    mock_gym_simulator = mocker.patch(
        "schola.core.simulators.gym.simulator.GymSimulator"
    )
    mocker.patch("schola.scripts.env.check.check.run_gym_env_checker")

    args = EnvCheckScriptSettings(
        environment_settings=EnvToolsEnvironmentSettings(
            simulator_settings=GymSimulatorConfig(
                env_id="CartPole-v1",
                num_simulators=2,
                num_environments=4,
            ),
        )
    )

    with caplog.at_level(logging.WARNING):
        main(args)

    mock_gym_simulator.assert_called_once_with("CartPole-v1", num_envs=4)
    mock_gym_env_cls.assert_called_once()
    assert "not supported for check" in caplog.text
