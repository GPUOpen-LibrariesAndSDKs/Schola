# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for the SB3 CLI"""

import logging
from cyclopts import App
import pytest
import gymnasium as gym
from unittest.mock import Mock, patch, MagicMock
from dataclasses import replace
from Test.sb3.envs import DictActionBoxEnv, make_dict_action_env
from schola.scripts.sb3.train.train import MetaTrainSB3Command, app, main
from schola.scripts.sb3.train.settings import (
    Sb3TrainScriptSettings,
    PPOTrainSettings,
    SACTrainSettings,
)
from schola.scripts.common.settings import (
    ExternalSimulatorConfig,
    GrpcProtocolConfig,
    UnrealExecutableSimulatorConfig,
    UnrealProjectSimulatorConfig,
    ActivationFunctionEnum,
    EnvironmentSettings,
)


@pytest.mark.parametrize("gym_id", ["CartPole-v1", "MountainCar-v0"])
def test_launch_script(make_vec_env_server, gym_id):
    """Test that the PPO script can be launched with a gRPC server."""
    N_ENVS = 3
    env_server_port = make_vec_env_server([gym.make(gym_id) for _ in range(N_ENVS)])

    # Create Sb3ScriptSettings with protocol nested under environment_settings
    args = Sb3TrainScriptSettings(
        environment_settings=EnvironmentSettings(
            protocol_settings=GrpcProtocolConfig(port=env_server_port, url="localhost")
        ),
        algorithm_settings=PPOTrainSettings(),
    )
    main(args)


def test_launch_script_with_dict_action_space(make_vec_env_server):
    """Test that the PPO script can be launched with a gRPC server and dictionary action space."""
    N_ENVS = 3
    env_server_port = make_vec_env_server(
        [make_dict_action_env(DictActionBoxEnv, False) for _ in range(N_ENVS)]
    )
    args = Sb3TrainScriptSettings(
        environment_settings=EnvironmentSettings(
            protocol_settings=GrpcProtocolConfig(port=env_server_port, url="localhost")
        ),
        algorithm_settings=PPOTrainSettings(),
    )
    main(args)


# CLI Mocking Tests - verify CLI argument parsing creates correct settings classes
@pytest.fixture
def mock_main(mocker):
    """Mock the main method."""
    return mocker.patch("schola.scripts.sb3.train.train.main")


@pytest.fixture
def mock_app(mock_main):
    """Mock the SB3 app."""
    # mock the main method
    app = App(name="train", help="Train a model using StableBaselines3")
    logger = logging.getLogger(__name__)
    app = MetaTrainSB3Command(app, Sb3TrainScriptSettings, mock_main, logger).make()
    return app.meta


def test_ppo_cli_default_args(mock_app, mock_main):
    """Test PPO command with default arguments creates correct settings."""

    # Parse CLI args with Cyclopts (returns tuple)
    command, bound, _ = mock_app.parse_args(["ppo"], exit_on_error=False)

    # Execute the parsed command (which calls the mocked main)
    command(*bound.args, **bound.kwargs)

    # Verify main was called once
    assert mock_main.call_count == 1

    # Get the Sb3ScriptSettings that was passed to main
    args = mock_main.call_args[0][0]

    # Verify it's the correct type
    assert isinstance(args, Sb3TrainScriptSettings)

    # Verify algorithm settings is PPO
    assert isinstance(args.algorithm_settings, PPOTrainSettings)

    # Verify default PPO parameters
    assert args.algorithm_settings.learning_rate == 0.0003
    assert args.algorithm_settings.n_steps == 2048
    assert args.algorithm_settings.batch_size == 64
    assert args.algorithm_settings.n_epochs == 10
    assert args.algorithm_settings.gamma == 0.99
    assert args.algorithm_settings.gae_lambda == 0.95
    assert args.algorithm_settings.clip_range == 0.2
    assert args.algorithm_settings.normalize_advantage == True
    assert args.algorithm_settings.ent_coef == 0.0
    assert args.algorithm_settings.vf_coef == 0.5


def test_ppo_cli_custom_args(mock_app, mock_main):
    """Test PPO command with custom arguments."""

    # Parse CLI args with custom PPO parameters
    command, bound, _ = mock_app.parse_args(
        [
            "ppo",
            "--learning-rate",
            "0.001",
            "--n-steps",
            "1024",
            "--batch-size",
            "128",
            "--n-epochs",
            "5",
            "--gamma",
            "0.95",
            "--gae-lambda",
            "0.9",
            "--clip-range",
            "0.3",
            "--ent-coef",
            "0.01",
            "--vf-coef",
            "0.25",
            "--timesteps",
            "10000",
        ],
        exit_on_error=False,
    )

    command(*bound.args, **bound.kwargs)

    assert mock_main.call_count == 1
    args = mock_main.call_args[0][0]

    # Verify algorithm settings is PPO
    assert isinstance(args.algorithm_settings, PPOTrainSettings)

    # Verify custom PPO parameters

    assert args.algorithm_settings.learning_rate == 0.001
    assert args.algorithm_settings.n_steps == 1024
    assert args.algorithm_settings.batch_size == 128
    assert args.algorithm_settings.n_epochs == 5
    assert args.algorithm_settings.gamma == 0.95
    assert args.algorithm_settings.gae_lambda == 0.9
    assert args.algorithm_settings.clip_range == 0.3
    assert args.algorithm_settings.ent_coef == 0.01
    assert args.algorithm_settings.vf_coef == 0.25

    # Verify top-level args
    assert args.training_settings.timesteps == 10000


def test_sac_cli_default_args(mock_app, mock_main):
    """Test SAC command with default arguments creates correct settings."""
    command, bound, _ = mock_app.parse_args(["sac"], exit_on_error=False)
    command(*bound.args, **bound.kwargs)

    assert mock_main.call_count == 1
    args = mock_main.call_args[0][0]

    # Verify it's the correct type
    assert isinstance(args, Sb3TrainScriptSettings)

    # Verify algorithm settings is SAC
    assert isinstance(args.algorithm_settings, SACTrainSettings)

    # Verify default SAC parameters
    assert args.algorithm_settings.learning_rate == 0.0003
    assert args.algorithm_settings.buffer_size == 1000000
    assert args.algorithm_settings.learning_starts == 100
    assert args.algorithm_settings.batch_size == 256
    assert args.algorithm_settings.tau == 0.005
    assert args.algorithm_settings.gamma == 0.99
    assert args.algorithm_settings.train_freq == 1
    assert args.algorithm_settings.gradient_steps == 1


def test_sac_cli_custom_args(mock_app, mock_main):
    """Test SAC command with custom arguments."""
    command, bound, _ = mock_app.parse_args(
        [
            "sac",
            "--learning-rate",
            "0.0005",
            "--buffer-size",
            "500000",
            "--learning-starts",
            "200",
            "--batch-size",
            "512",
            "--tau",
            "0.01",
            "--gamma",
            "0.98",
            "--train-freq",
            "2",
            "--gradient-steps",
            "2",
            "--timesteps",
            "50000",
        ],
        exit_on_error=False,
    )

    command(*bound.args, **bound.kwargs)

    assert mock_main.call_count == 1
    args = mock_main.call_args[0][0]

    # Verify algorithm settings is SAC
    assert isinstance(args.algorithm_settings, SACTrainSettings)

    # Verify custom SAC parameters
    assert args.algorithm_settings.learning_rate == 0.0005
    assert args.algorithm_settings.buffer_size == 500000
    assert args.algorithm_settings.learning_starts == 200
    assert args.algorithm_settings.batch_size == 512
    assert args.algorithm_settings.tau == 0.01
    assert args.algorithm_settings.gamma == 0.98
    assert args.algorithm_settings.train_freq == 2
    assert args.algorithm_settings.gradient_steps == 2

    assert args.training_settings.timesteps == 50000


def test_ppo_network_architecture_args(mock_app, mock_main):
    """Test that network architecture arguments are correctly parsed."""
    command, bound, _ = mock_app.parse_args(
        [
            "ppo",
            "--policy-parameters",
            "128",
            "128",
            "64",
            "--critic-parameters",
            "256",
            "128",
            "--activation",
            "TanH",
        ],
        exit_on_error=False,
    )

    command(*bound.args, **bound.kwargs)

    args = mock_main.call_args[0][0]

    # Verify network architecture settings
    assert args.network_architecture_settings.policy_parameters == [128, 128, 64]
    assert args.network_architecture_settings.critic_parameters == [256, 128]
    assert args.network_architecture_settings.activation == ActivationFunctionEnum.TanH


def test_ppo_logging_args(mock_app, mock_main):
    """Test that logging arguments are correctly parsed."""

    command, bound, _ = mock_app.parse_args(
        [
            "ppo",
            "--enable-tensorboard",
            "--log-dir",
            "./test_logs",
            "--log-freq",
            "100",
            "--callback-verbosity",
            "2",
            "--schola-verbosity",
            "1",
            "--sb3-verbosity",
            "2",
        ],
        exit_on_error=False,
    )

    command(*bound.args, **bound.kwargs)

    args = mock_main.call_args[0][0]

    # Verify logging settings
    assert args.logging_settings.enable_tensorboard == True
    assert str(args.logging_settings.log_dir) == "test_logs"
    assert args.logging_settings.log_freq == 100
    assert args.logging_settings.callback_verbosity == 2
    assert args.logging_settings.schola_verbosity == 1
    assert args.logging_settings.sb3_verbosity == 2


def test_ppo_checkpoint_args(mock_app, mock_main):
    """Test that checkpoint arguments are correctly parsed."""
    command, bound, _ = mock_app.parse_args(
        [
            "ppo",
            "--save-freq",
            "5000",
            "--save-replay-buffer",
            "--save-vecnormalize",
        ],
        exit_on_error=False,
    )

    command(*bound.args, **bound.kwargs)

    args = mock_main.call_args[0][0]

    # Verify checkpoint settings
    assert args.checkpoint_settings.save_freq == 5000
    assert args.checkpoint_settings.save_replay_buffer == True
    assert args.checkpoint_settings.save_vecnormalize == True


def test_ppo_options_default_is_empty_dict(mock_app, mock_main):
    command, bound, _ = mock_app.parse_args(["ppo"], exit_on_error=False)
    command(*bound.args, **bound.kwargs)
    args = mock_main.call_args[0][0]
    assert args.environment_settings.env_options == {}


def test_ppo_env_options_dotted_syntax(mock_app, mock_main):
    command, bound, _ = mock_app.parse_args(
        [
            "ppo",
            "--env-options.level=1",
            "--env-options.curriculum=easy",
        ],
        exit_on_error=False,
    )
    command(*bound.args, **bound.kwargs)
    args = mock_main.call_args[0][0]
    assert args.environment_settings.env_options == {
        "level": "1",
        "curriculum": "easy",
    }


def test_sac_with_executable_simulator(mock_app, mock_main, tmp_path):
    """Test SAC command with executable simulator type and executable path is correctly parsed."""
    # Create a fake executable file
    fake_executable = tmp_path / "MyExe.exe"
    fake_executable.write_text("#!/bin/bash\necho 'hello'")

    command, bound, _ = mock_app.parse_args(
        ["sac", "executable", str(fake_executable)], exit_on_error=False
    )

    command(*bound.args, **bound.kwargs)

    args = mock_main.call_args[0][0]

    # Should now be UnrealExecutableSimulatorConfig with correct path
    assert isinstance(
        args.environment_settings.simulator_settings, UnrealExecutableSimulatorConfig
    )
    assert (
        args.environment_settings.simulator_settings.executable_path == fake_executable
    )


def test_ppo_with_external_simulator(mock_app, mock_main):
    """Test that external simulator type is correctly parsed."""
    command, bound, _ = mock_app.parse_args(["ppo", "external"], exit_on_error=False)

    command(*bound.args, **bound.kwargs)

    args = mock_main.call_args[0][0]

    assert isinstance(
        args.environment_settings.simulator_settings, ExternalSimulatorConfig
    )
    assert args.environment_settings.simulator_settings.num_simulators == 1


def test_executable_num_simulators_parsed(mock_app, mock_main, tmp_path):
    """Test that num_simulators is parsed for executable simulator."""
    fake_executable = tmp_path / "MyExe.exe"
    fake_executable.write_text("#!/bin/bash\necho 'hello'")
    command, bound, _ = mock_app.parse_args(
        [
            "sac",
            "executable",
            str(fake_executable),
            "--num-simulators",
            "3",
        ],
        exit_on_error=False,
    )
    command(*bound.args, **bound.kwargs)
    args = mock_main.call_args[0][0]
    assert isinstance(
        args.environment_settings.simulator_settings, UnrealExecutableSimulatorConfig
    )
    assert args.environment_settings.simulator_settings.num_simulators == 3


def test_project_num_simulators_parsed(mock_app, mock_main, tmp_path):
    """Test that num_simulators is parsed for project simulator."""
    uproject = tmp_path / "MyGame.uproject"
    uproject.write_text("{}")
    command, bound, _ = mock_app.parse_args(
        [
            "ppo",
            "project",
            str(uproject),
            "--num-simulators",
            "2",
        ],
        exit_on_error=False,
    )
    command(*bound.args, **bound.kwargs)
    args = mock_main.call_args[0][0]
    assert isinstance(
        args.environment_settings.simulator_settings, UnrealProjectSimulatorConfig
    )
    assert args.environment_settings.simulator_settings.num_simulators == 2


@pytest.mark.parametrize(
    "algorithm,settings_class",
    [
        ("ppo", PPOTrainSettings),
        ("sac", SACTrainSettings),
    ],
)
def test_algorithm_settings_type(mock_app, mock_main, algorithm, settings_class):
    """Parametrized test to verify correct settings class for each algorithm."""

    command, bound, _ = mock_app.parse_args([algorithm], exit_on_error=False)
    command(*bound.args, **bound.kwargs)

    args = mock_main.call_args[0][0]
    assert isinstance(args.algorithm_settings, settings_class)


def test_ppo_with_pbar(mock_app, mock_main):
    """Test that progress bar flag is correctly parsed."""

    command, bound, _ = mock_app.parse_args(
        [
            "ppo",
            "--pbar",
        ],
        exit_on_error=False,
    )

    command(*bound.args, **bound.kwargs)

    args = mock_main.call_args[0][0]
    assert args.training_settings.pbar == True


def test_ppo_with_disable_eval(mock_app, mock_main):
    """Test that disable-eval flag is correctly parsed."""

    command, bound, _ = mock_app.parse_args(
        [
            "ppo",
            "--disable-eval",
        ],
        exit_on_error=False,
    )

    command(*bound.args, **bound.kwargs)

    args = mock_main.call_args[0][0]
    assert args.training_settings.disable_eval == True


def test_sac_with_sde(mock_app, mock_main):
    """Test SAC with state-dependent exploration arguments."""
    command, bound, _ = mock_app.parse_args(
        [
            "sac",
            "--use-sde",
            "--sde-sample-freq",
            "10",
        ],
        exit_on_error=False,
    )

    command(*bound.args, **bound.kwargs)

    args = mock_main.call_args[0][0]
    assert args.algorithm_settings.use_sde == True
    assert args.algorithm_settings.sde_sample_freq == 10


def test_ppo_complex_scenario(mock_app, mock_main):
    """Test PPO with a complex combination of arguments."""
    command, bound, _ = mock_app.parse_args(
        [
            "ppo",
            "--timesteps",
            "100000",
            "--learning-rate",
            "0.0001",
            "--n-steps",
            "512",
            "--batch-size",
            "128",
            "--n-epochs",
            "20",
            "--policy-parameters",
            "64",
            "64",
            "--critic-parameters",
            "64",
            "64",
            "--activation",
            "Sigmoid",
            "--enable-tensorboard",
            "--log-dir",
            "./complex_logs",
            "--save-freq",
            "10000",
            "--pbar",
        ],
        exit_on_error=False,
    )

    command(*bound.args, **bound.kwargs)

    args = mock_main.call_args[0][0]

    # Verify multiple settings are correctly combined
    assert args.training_settings.timesteps == 100000
    assert args.algorithm_settings.learning_rate == 0.0001
    assert args.algorithm_settings.n_steps == 512
    assert args.algorithm_settings.batch_size == 128
    assert args.network_architecture_settings.policy_parameters == [64, 64]
    assert (
        args.network_architecture_settings.activation == ActivationFunctionEnum.Sigmoid
    )
    assert args.logging_settings.enable_tensorboard == True
    assert args.checkpoint_settings.save_freq == 10000
    assert args.training_settings.pbar == True
