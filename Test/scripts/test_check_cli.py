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
    SingularExternalSimulatorConfig,
)
from schola.scripts.env.check.check import EnvCheckCommand, main
from schola.scripts.env.check.settings import (
    EnvCheckEnvironmentSettings,
    EnvCheckScriptSettings,
)


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
    assert isinstance(args.environment_settings.simulator_settings, SingularExternalSimulatorConfig)
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


def test_check_main_on_real_env(capsys, mocker):
    mock_check_env = mocker.patch("schola.scripts.env.utils.check_env")

    args = EnvCheckScriptSettings(
        environment_settings=EnvCheckEnvironmentSettings(
            simulator_settings=GymSimulatorConfig(env_id="CartPole-v1", num_environments=1),
        )
    )

    main(args)

    mock_check_env.assert_called_once()
    assert mock_check_env.call_args.kwargs["skip_render_check"] is True
    err = capsys.readouterr().err
    assert "Running Gymnasium environment checker" in err
    assert "Environment checker passed." in err
