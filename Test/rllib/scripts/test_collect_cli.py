# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for the RLlib demonstration collection command."""

import logging

import pytest
from cyclopts import App

from schola.scripts.common.settings import (
    ExternalSimulatorConfig,
    UnrealExecutableSimulatorConfig,
)
from schola.scripts.rllib.collect.collect import CollectRllibCommand
from schola.scripts.rllib.collect.settings import RllibCollectScriptSettings


@pytest.fixture
def mock_collect_main(mocker):
    return mocker.patch("schola.scripts.rllib.collect.collect.main")


@pytest.fixture
def mock_app(mock_collect_main):
    class MockCollectRllibCommand(CollectRllibCommand):
        @property
        def main_func(self):
            return mock_collect_main

    app = App(name="collect", help="Collect RLlib demonstrations")
    return MockCollectRllibCommand(app, logging.getLogger(__name__)).make()


def test_collect_defaults_to_external_simulator(mock_app, mock_collect_main, tmp_path):
    output = tmp_path / "demos"

    mock_app.meta(
        ["--output", str(output)],
        result_action="return_value",
        exit_on_error=False,
    )

    args: RllibCollectScriptSettings = mock_collect_main.call_args[0][0]
    assert args.collection_settings.output == output
    assert args.collection_settings.num_steps == 1000
    assert args.collection_settings.episodes_per_shard == 64
    assert isinstance(
        args.environment_settings.simulator_settings, ExternalSimulatorConfig
    )


def test_collect_parses_fixed_step_options(mock_app, mock_collect_main, tmp_path):
    output = tmp_path / "demos"

    mock_app.meta(
        [
            "external",
            "--output",
            str(output),
            "--num-steps",
            "250",
            "--episodes-per-shard",
            "8",
            "--seed",
            "42",
        ],
        result_action="return_value",
        exit_on_error=False,
    )

    args: RllibCollectScriptSettings = mock_collect_main.call_args[0][0]
    assert args.collection_settings.num_steps == 250
    assert args.collection_settings.episodes_per_shard == 8
    assert args.environment_settings.seed == 42


def test_collect_binds_executable_simulator(mock_app, mock_collect_main, tmp_path):
    output = tmp_path / "demos"
    executable = tmp_path / "Game.exe"
    executable.touch()

    mock_app.meta(
        [
            "executable",
            "--executable-path",
            str(executable),
            "--output",
            str(output),
        ],
        result_action="return_value",
        exit_on_error=False,
    )

    args: RllibCollectScriptSettings = mock_collect_main.call_args[0][0]
    simulator = args.environment_settings.simulator_settings
    assert isinstance(simulator, UnrealExecutableSimulatorConfig)
    assert simulator.executable_path == executable


def test_collect_config_file(mock_app, mock_collect_main, tmp_path):
    output = tmp_path / "demos"
    config_file = tmp_path / "collect.yaml"
    config_file.write_text(
        f"""
output: {output.as_posix()}
num_steps: 50
episodes_per_shard: 4
""".strip(),
        encoding="utf-8",
    )

    mock_app.meta(
        ["--config-file", str(config_file)],
        result_action="return_value",
        exit_on_error=False,
    )

    args: RllibCollectScriptSettings = mock_collect_main.call_args[0][0]
    assert args.collection_settings.output == output
    assert args.collection_settings.num_steps == 50
    assert args.collection_settings.episodes_per_shard == 4
