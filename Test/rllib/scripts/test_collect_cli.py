# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for the RLlib collect CLI argument parsing."""

import logging

import pytest
from cyclopts import App

from schola.scripts.common.settings import (
    ExternalSimulatorConfig,
    UnrealExecutableSimulatorConfig,
)
from schola.scripts.rllib.collect.collect import CollectRllibCommand
from schola.scripts.rllib.collect.settings import (
    RllibCollectLoggingSettings,
    RllibCollectScriptSettings,
    RllibCollectionSettings,
)


@pytest.fixture
def mock_main(mocker):
    return mocker.patch("schola.scripts.rllib.collect.collect.main")


@pytest.fixture
def mock_app(mock_main):
    app = App(
        name="collect",
        help="Collect imitation learning datasets in RLlib's offline Parquet format",
    )
    logger = logging.getLogger(__name__)

    class MetaCollectRllibCommand(CollectRllibCommand):
        @property
        def main_func(self):
            return mock_main

    return MetaCollectRllibCommand(app, logger).make()


def test_collect_cli_requires_output(mock_app, mock_main):
    with pytest.raises(Exception):
        mock_app(
            ["external"],
            result_action="return_value",
            exit_on_error=False,
        )
    mock_main.assert_not_called()


def test_collect_cli_default_simulator_is_external(mock_app, mock_main, tmp_path):
    output = tmp_path / "demos"
    mock_app(
        ["--output", str(output)],
        result_action="return_value",
        exit_on_error=False,
    )
    args: RllibCollectScriptSettings = mock_main.call_args[0][0]
    assert args.collection_settings.output == output
    assert args.collection_settings.max_steps is None
    assert isinstance(
        args.environment_settings.simulator_settings, ExternalSimulatorConfig
    )


def test_collect_cli_executable_binds_simulator_config(
    mock_app, mock_main, tmp_path
):
    output = tmp_path / "demos"
    executable_path = tmp_path / "UnrealGame.exe"
    executable_path.touch()
    mock_app(
        [
            "executable",
            "--executable-path",
            str(executable_path),
            "--output",
            str(output),
            "--max-steps",
            "50",
            "--seed",
            "7",
        ],
        result_action="return_value",
        exit_on_error=False,
    )
    args: RllibCollectScriptSettings = mock_main.call_args[0][0]
    assert args.collection_settings.max_steps == 50
    assert args.collection_settings.seed == 7
    assert isinstance(
        args.environment_settings.simulator_settings, UnrealExecutableSimulatorConfig
    )
    assert (
        args.environment_settings.simulator_settings.executable_path == executable_path
    )


def test_collect_script_settings_defaults():
    args = RllibCollectScriptSettings()
    assert isinstance(args.collection_settings, RllibCollectionSettings)
    assert isinstance(args.logging_settings, RllibCollectLoggingSettings)
    assert args.collection_settings.output is None
    assert args.collection_settings.max_steps is None
    assert args.logging_settings.schola_verbosity == 0
