# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for the Minari CLI argument parsing and command structure."""

import logging
import os
import minari
from minari.utils import MinariStorage, get_dataset_path
import pytest
import gymnasium as gym
from cyclopts import App
from schola.scripts.minari.collect import collect_app, main
from schola.scripts.minari.settings import (
    MinariScriptSettings,
    MinariCollectionSettings,
    MinariLoggingSettings,
)
from schola.scripts.common.settings import EnvironmentSettings, ExternalSimulatorConfig
from schola.scripts.common.command_template import ScholaCommandTemplate

# CLI Mocking Tests - verify CLI argument parsing creates correct settings classes

 

@pytest.fixture
def mock_main(mocker):
    """Mock the main function."""
    return mocker.patch("schola.scripts.minari.collect.main")


@pytest.fixture
def mock_app(mock_main):
    """Build a fresh app with mocked main (no global injection)."""
    app = App(name="collect", help="Collect imitation learning datasets using Minari")
    logger = logging.getLogger(__name__)
    class MetaCollectMinariCommand(ScholaCommandTemplate[MinariScriptSettings]):
        
        @property
        def algorithm_table(self):
            return {}

        @property
        def script_args_type(self):
            return MinariScriptSettings
        
        @property
        def main_func(self):
            return mock_main

    app = MetaCollectMinariCommand(app, logger).make()
    return app.meta


def test_collect_cli_default_args(mock_app, mock_main):
    """Test collect command with default arguments creates correct settings."""
    mock_app(
        ["external", "--dataset-id", "test-dataset-v0"],
        result_action="return_value",
        exit_on_error=False,
    )
    assert mock_main.call_count == 1

    # Get the MinariScriptSettings that was passed to main
    args = mock_main.call_args[0][0]

    # Verify it's the correct type
    assert isinstance(args, MinariScriptSettings)

    # Verify default collection parameters
    assert args.collection_settings.dataset_id == "test-dataset-v0"
    assert args.collection_settings.num_steps == 1000
    assert args.collection_settings.seed is None
    assert args.collection_settings.author is None
    assert args.collection_settings.author_email is None
    assert args.collection_settings.code_permalink is None
    assert args.collection_settings.algorithm_name is None
    assert args.collection_settings.description is None
    assert args.collection_settings.record_infos is False


def test_collect_complex_scenario(mock_app, mock_main):
    """Test collect command with a complex combination of arguments."""
    command, bound, _ = mock_app.parse_args(
        [
            "external",
            "--dataset-id",
            "complex-dataset-v0",
            "--num-steps",
            "10000",
            "--seed",
            "123",
            "--author",
            "Jane Smith",
            "--author-email",
            "jane@example.com",
            "--code-permalink",
            "https://github.com/jane/repo",
            "--algorithm-name",
            "ppo",
            "--description",
            "Complex test scenario with all parameters",
            "--record-infos",
            "--schola-verbosity",
            "1",
        ],
        exit_on_error=False,
    )
    command(*bound.args, **bound.kwargs)

    args = mock_main.call_args[0][0]

    # Verify all settings are correctly combined
    assert args.collection_settings.dataset_id == "complex-dataset-v0"
    assert args.collection_settings.num_steps == 10000
    assert args.collection_settings.seed == 123
    assert args.collection_settings.author == "Jane Smith"
    assert args.collection_settings.author_email == "jane@example.com"
    assert args.collection_settings.code_permalink == "https://github.com/jane/repo"
    assert args.collection_settings.algorithm_name == "ppo"
    assert (
        args.collection_settings.description
        == "Complex test scenario with all parameters"
    )
    assert args.collection_settings.record_infos is True
    assert args.logging_settings.schola_verbosity == 1
    assert isinstance(
        args.environment_settings.simulator_settings, ExternalSimulatorConfig
    )


def test_minari_script_args_structure():
    """Test that MinariScriptSettings properly composes settings."""
    args = MinariScriptSettings()

    # Should have collection_settings
    assert hasattr(args, "collection_settings")
    assert isinstance(args.collection_settings, MinariCollectionSettings)

    # Should have logging_settings
    assert hasattr(args, "logging_settings")
    assert isinstance(args.logging_settings, MinariLoggingSettings)

    # Should have environment_settings
    assert hasattr(args, "environment_settings")
    assert isinstance(args.environment_settings, EnvironmentSettings)


def test_minari_collection_args_defaults():
    """Test that MinariCollectionSettings has correct default values."""
    args = MinariCollectionSettings(dataset_id="test-dataset-v0")

    assert args.dataset_id == "test-dataset-v0"
    assert args.num_steps == 1000
    assert args.seed is None
    assert args.author is None
    assert args.author_email is None
    assert args.code_permalink is None
    assert args.algorithm_name is None
    assert args.description is None
    assert args.record_infos is False


def test_minari_logging_args_defaults():
    """Test MinariLoggingSettings default values."""
    args = MinariLoggingSettings()
    assert args.schola_verbosity == 0


def test_minari_logging_args_custom():
    """Test MinariLoggingSettings with custom values."""
    args = MinariLoggingSettings(schola_verbosity=2)
    assert args.schola_verbosity == 2


def test_minari_collection_script(
    tmp_path, make_imitation_server, simple_policy, imitation_id_and_wrappers
):
    """Test collect command via CLI, including properties of output."""
    # Create a temp directory for output
    data_path = tmp_path / "minari_datasets"
    dataset_id = "test-script-dataset-v0"
    env_id, wrappers = imitation_id_and_wrappers
    port = make_imitation_server(env_id, simple_policy, wrappers)

    initial_minari_datasets_path = os.environ.get("MINARI_DATASETS_PATH", None)

    collect_app.meta(
        [
            "external",
            "--dataset-id",
            dataset_id,
            "--num-steps",
            "22",
            "--author",
            "testuser",
            "--author-email",
            "test@example.com",
            "--code-permalink",
            "https://github.com/test/repo",
            "--data-path",
            str(data_path.resolve()),
            "--port",
            f"{port}",
            "--description",
            "test description",
            "--algorithm-name",
            "test algorithm",
        ],
        result_action="return_value",
        exit_on_error=False,
    )

    storage = MinariStorage.read(data_path / dataset_id / "data")
    baseline_env = gym.make(env_id)
    assert (
        storage.action_space == baseline_env.action_space
    ), f"Action space mismatch: {storage.action_space} != {baseline_env.action_space}"
    assert (
        storage.observation_space == baseline_env.observation_space
    ), f"Observation space mismatch: {storage.observation_space} != {baseline_env.observation_space}"
    baseline_env.close()

    assert (
        storage.total_steps == 22
    ), f"Total steps mismatch: {storage.total_steps} != 22"
    assert storage.metadata.get("author") == {
        "testuser"
    }, f"Author mismatch: {storage.metadata.get('author')} != {'testuser'}"
    assert storage.metadata.get("author_email") == {
        "test@example.com"
    }, f"Author email mismatch: {storage.metadata.get('author_email')} != {'test@example.com'}"
    assert (
        storage.metadata.get("code_permalink") == "https://github.com/test/repo"
    ), f"Code permalink mismatch: {storage.metadata.get('code_permalink')} != https://github.com/test/repo"
    assert (
        storage.metadata.get("algorithm_name") == "test algorithm"
    ), f"Algorithm name mismatch: {storage.metadata.get('algorithm_name')} != test algorithm"
    assert (
        storage.metadata.get("description") == "test description"
    ), f"Description mismatch: {storage.metadata.get('description')} != test description"
    assert (
        os.environ.get("MINARI_DATASETS_PATH") == initial_minari_datasets_path
    ), f"MINARI_DATASETS_PATH was not restored to the initial value {initial_minari_datasets_path}"
