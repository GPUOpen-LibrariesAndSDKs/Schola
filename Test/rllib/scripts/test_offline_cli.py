# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for the RLlib BC and MARWIL CLI commands."""

import pytest

from schola.scripts.common.settings import (
    ExternalSimulatorConfig,
    UnrealExecutableSimulatorConfig,
)
from schola.scripts.rllib.offline_train import (
    BCScriptSettings,
    MARWILScriptSettings,
    _make_offline_command,
    _resolve_dataset_path,
)
from schola.scripts.rllib.settings import BCSettings, MARWILSettings
from schola.scripts.rllib.train.train import ResourcePlan


@pytest.fixture
def mock_offline_main(mocker):
    return mocker.patch("schola.scripts.rllib.offline_train.main_offline")


@pytest.fixture
def mock_bc_app(mock_offline_main):
    return _make_offline_command("bc", "Train BC", BCScriptSettings)


@pytest.fixture
def mock_marwil_app(mock_offline_main):
    return _make_offline_command("marwil", "Train MARWIL", MARWILScriptSettings)


def test_bc_train_only_binds_input_without_simulator(
    mock_bc_app, mock_offline_main, tmp_path
):
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
    assert args.environment_settings.simulator_settings is None
    assert args.collection_settings.output is None


def test_bc_does_not_default_to_external_simulator(
    mock_bc_app, mock_offline_main, tmp_path
):
    dataset_dir = tmp_path / "demos"
    dataset_dir.mkdir()
    mock_bc_app.meta(
        ["--input", str(dataset_dir)],
        result_action="return_value",
        exit_on_error=False,
    )
    args: BCScriptSettings = mock_offline_main.call_args[0][0]
    assert args.environment_settings.simulator_settings is None


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
    assert args.environment_settings.simulator_settings is None


def test_bc_executable_collect_then_train_binds_output(
    mock_bc_app, mock_offline_main, tmp_path
):
    output = tmp_path / "demos"
    executable_path = tmp_path / "UnrealGame.exe"
    executable_path.touch()
    mock_bc_app.meta(
        [
            "executable",
            "--executable-path",
            str(executable_path),
            "--output",
            str(output),
            "--max-steps",
            "50",
        ],
        result_action="return_value",
        exit_on_error=False,
    )
    args: BCScriptSettings = mock_offline_main.call_args[0][0]
    assert args.collection_settings.output == output
    assert args.collection_settings.max_steps == 50
    assert args.algorithm_settings.input_path is None
    assert isinstance(
        args.environment_settings.simulator_settings, UnrealExecutableSimulatorConfig
    )
    assert (
        args.environment_settings.simulator_settings.executable_path == executable_path
    )


def test_explicit_external_simulator_is_bound(mock_bc_app, mock_offline_main, tmp_path):
    output = tmp_path / "demos"
    mock_bc_app.meta(
        ["external", "--output", str(output)],
        result_action="return_value",
        exit_on_error=False,
    )
    args: BCScriptSettings = mock_offline_main.call_args[0][0]
    assert isinstance(
        args.environment_settings.simulator_settings, ExternalSimulatorConfig
    )
    assert args.collection_settings.output == output


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


def test_resolve_dataset_path_requires_input_without_simulator():
    with pytest.raises(ValueError, match="--input is required"):
        _resolve_dataset_path(BCScriptSettings())


def test_resolve_dataset_path_rejects_output_without_simulator(tmp_path):
    args = BCScriptSettings()
    args.collection_settings.output = tmp_path / "demos"
    with pytest.raises(ValueError, match="--output requires a simulator"):
        _resolve_dataset_path(args)


def test_resolve_dataset_path_requires_output_with_simulator():
    args = BCScriptSettings()
    args.environment_settings.simulator_settings = ExternalSimulatorConfig()
    with pytest.raises(ValueError, match="--output is required"):
        _resolve_dataset_path(args)


def test_resolve_dataset_path_rejects_input_with_simulator(tmp_path):
    dataset_dir = tmp_path / "demos"
    dataset_dir.mkdir()
    args = BCScriptSettings()
    args.environment_settings.simulator_settings = ExternalSimulatorConfig()
    args.collection_settings.output = tmp_path / "out"
    args.algorithm_settings.input_path = dataset_dir
    with pytest.raises(ValueError, match="--input cannot be combined"):
        _resolve_dataset_path(args)


def test_resolve_dataset_path_train_only_returns_input(tmp_path):
    dataset_dir = tmp_path / "demos"
    dataset_dir.mkdir()
    args = BCScriptSettings()
    args.algorithm_settings.input_path = dataset_dir
    assert _resolve_dataset_path(args) == dataset_dir
