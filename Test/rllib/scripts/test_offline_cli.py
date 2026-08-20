# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for the RLlib BC and MARWIL CLI commands."""

import logging

import pytest
from cyclopts import App

from schola.scripts.common.command_template import ScholaCommandTemplate
from schola.scripts.rllib.offline_train import BCScriptSettings, MARWILScriptSettings
from schola.scripts.rllib.settings import BCSettings, MARWILSettings
from schola.scripts.rllib.train.train import ResourcePlan


@pytest.fixture
def mock_offline_main(mocker):
    return mocker.patch("schola.scripts.rllib.offline_train.main_offline")


@pytest.fixture
def mock_bc_app(mock_offline_main):
    logger = logging.getLogger(__name__)
    app = App(name="bc", help="Train BC")

    class MockBcCommand(ScholaCommandTemplate[BCScriptSettings]):
        @property
        def algorithm_table(self):
            return {}

        @property
        def simulator_table(self):
            return {}

        @property
        def script_args_type(self):
            return BCScriptSettings

        @property
        def main_func(self):
            return mock_offline_main

    return MockBcCommand(app, logger).make()


@pytest.fixture
def mock_marwil_app(mock_offline_main):
    logger = logging.getLogger(__name__)
    app = App(name="marwil", help="Train MARWIL")

    class MockMarwilCommand(ScholaCommandTemplate[MARWILScriptSettings]):
        @property
        def algorithm_table(self):
            return {}

        @property
        def simulator_table(self):
            return {}

        @property
        def script_args_type(self):
            return MARWILScriptSettings

        @property
        def main_func(self):
            return mock_offline_main

    return MockMarwilCommand(app, logger).make()


def test_bc_requires_input(mock_bc_app, mock_offline_main):
    with pytest.raises((Exception, SystemExit)):
        mock_bc_app.meta([], result_action="return_value", exit_on_error=False)
    mock_offline_main.assert_not_called()


def test_bc_input_argument(mock_bc_app, mock_offline_main, tmp_path):
    dataset_dir = tmp_path / "demos"
    dataset_dir.mkdir()
    mock_bc_app.meta(
        ["--input", str(dataset_dir)],
        result_action="return_value",
        exit_on_error=False,
    )

    args: BCScriptSettings = mock_offline_main.call_args[0][0]
    assert isinstance(args.algorithm_settings, BCSettings)
    assert args.algorithm_settings.input_path == dataset_dir


def test_bc_config_file_reads_input(mock_bc_app, mock_offline_main, tmp_path):
    dataset_dir = tmp_path / "demos"
    dataset_dir.mkdir()
    config_file = tmp_path / "offline.yaml"
    config_file.write_text(
        f"""
input: {dataset_dir.as_posix()}
offline_data_workers: 3
training_settings:
  timesteps: 64
""".strip(),
        encoding="utf-8",
    )

    mock_bc_app.meta(
        ["--config-file", str(config_file)],
        result_action="return_value",
        exit_on_error=False,
    )

    args: BCScriptSettings = mock_offline_main.call_args[0][0]
    assert args.algorithm_settings.input_path == dataset_dir
    assert args.algorithm_settings.offline_data_workers == 3
    assert args.training_settings.timesteps == 64


def test_marwil_beta_argument(mock_marwil_app, mock_offline_main, tmp_path):
    dataset_dir = tmp_path / "demos"
    dataset_dir.mkdir()
    mock_marwil_app.meta(
        ["--input", str(dataset_dir), "--beta", "0.0"],
        result_action="return_value",
        exit_on_error=False,
    )

    args: MARWILScriptSettings = mock_offline_main.call_args[0][0]
    assert isinstance(args.algorithm_settings, MARWILSettings)
    assert args.algorithm_settings.beta == 0.0
    assert args.algorithm_settings.get_settings_dict()["beta"] == 0.0


def test_offline_resource_plan_accounts_for_reader_and_workers(tmp_path):
    dataset_dir = tmp_path / "demos"
    dataset_dir.mkdir()
    args = BCScriptSettings()
    args.algorithm_settings = BCSettings(input_path=dataset_dir)

    plan = ResourcePlan.offline(args, args.algorithm_settings)

    assert plan.minimum_cpus == 4
    assert plan.ray_cpus == 4
    assert "2 pre-learner CPUs" in plan.description


def test_bc_rejects_simulator_subcommand(mock_bc_app, mock_offline_main, tmp_path):
    dataset_dir = tmp_path / "demos"
    dataset_dir.mkdir()
    with pytest.raises(Exception):
        mock_bc_app.meta(
            ["--input", str(dataset_dir), "executable", "--executable-path", "game.exe"],
            result_action="return_value",
            exit_on_error=False,
        )
