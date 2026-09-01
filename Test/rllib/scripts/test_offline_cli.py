# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for the dedicated RLlib offline training command."""

import logging

import pytest
from cyclopts import App

from schola.scripts.rllib.offline_train.offline_train import OfflineTrainCommand
from schola.scripts.rllib.offline_train.settings import OfflineRllibScriptSettings
from schola.scripts.rllib.settings import BCSettings, MARWILSettings
from schola.scripts.rllib.training import ResourcePlan


@pytest.fixture
def mock_offline_main(mocker):
    return mocker.patch("schola.scripts.rllib.offline_train.offline_train.main")


@pytest.fixture
def mock_app(mock_offline_main):
    class MockOfflineTrainCommand(OfflineTrainCommand):
        @property
        def main_func(self):
            return mock_offline_main

    app = App(name="offline-train", help="Train from an RLlib dataset")
    return MockOfflineTrainCommand(app, logging.getLogger(__name__)).make()


def test_bc_binds_existing_input(mock_app, mock_offline_main, tmp_path):
    dataset_dir = tmp_path / "demos"
    dataset_dir.mkdir()

    mock_app.meta(
        ["bc", "--input", str(dataset_dir)],
        result_action="return_value",
        exit_on_error=False,
    )

    args: OfflineRllibScriptSettings = mock_offline_main.call_args[0][0]
    assert isinstance(args.algorithm_settings, BCSettings)
    assert args.algorithm_settings.input_path == dataset_dir


def test_marwil_binds_algorithm_arguments(mock_app, mock_offline_main, tmp_path):
    dataset_dir = tmp_path / "demos"
    dataset_dir.mkdir()

    mock_app.meta(
        ["marwil", "--input", str(dataset_dir), "--beta", "0.0"],
        result_action="return_value",
        exit_on_error=False,
    )

    args: OfflineRllibScriptSettings = mock_offline_main.call_args[0][0]
    assert isinstance(args.algorithm_settings, MARWILSettings)
    assert args.algorithm_settings.input_path == dataset_dir
    assert args.algorithm_settings.beta == 0.0


def test_offline_train_accepts_seed(mock_app, mock_offline_main, tmp_path):
    dataset_dir = tmp_path / "demos"
    dataset_dir.mkdir()

    mock_app.meta(
        ["bc", "--input", str(dataset_dir), "--seed", "42"],
        result_action="return_value",
        exit_on_error=False,
    )

    args: OfflineRllibScriptSettings = mock_offline_main.call_args[0][0]
    assert args.seed == 42


def test_config_file_reads_input(mock_app, mock_offline_main, tmp_path):
    dataset_dir = tmp_path / "demos"
    dataset_dir.mkdir()
    config_file = tmp_path / "offline.yaml"
    config_file.write_text(
        f"""
algorithm:
  bc:
    input: {dataset_dir.as_posix()}
    offline_data_workers: 3
training_settings:
  timesteps: 64
""".strip(),
        encoding="utf-8",
    )

    mock_app.meta(
        ["bc", "--config-file", str(config_file)],
        result_action="return_value",
        exit_on_error=False,
    )

    args: OfflineRllibScriptSettings = mock_offline_main.call_args[0][0]
    assert args.algorithm_settings.input_path == dataset_dir
    assert args.algorithm_settings.offline_data_workers == 3
    assert args.training_settings.timesteps == 64


def test_offline_resource_plan_accounts_for_reader_and_workers(tmp_path):
    dataset_dir = tmp_path / "demos"
    dataset_dir.mkdir()
    args = OfflineRllibScriptSettings()
    args.algorithm_settings = BCSettings(input_path=dataset_dir)

    plan = ResourcePlan.offline(args, args.algorithm_settings)

    assert plan.minimum_cpus == 4
    assert plan.ray_cpus == 4
    assert "2 pre-learner CPUs" in plan.description


def test_offline_train_has_no_simulator_commands(mock_app):
    assert "external" not in mock_app
    assert "executable" not in mock_app
    assert "project" not in mock_app
    assert "gym" not in mock_app
