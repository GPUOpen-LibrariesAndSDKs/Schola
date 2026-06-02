# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for the rllib cli"""

from copy import deepcopy
import logging
import pickle
from cyclopts import App
import pytest
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.env_context import EnvContext

from schola.scripts.common.settings import UnrealExecutableSimulatorConfig
from schola.scripts.rllib.train.train import (
    app as train_app,
    RllibTrainCommand,
    _get_restored_env_steps,
    _make_stop_criterion,
)
from schola.core.utils.id_manager import IdManager
from schola.rllib.env import BaseRayEnv
from schola.scripts.rllib.settings import (
    APPOSettings,
    PPOSettings,
    SACSettings,
    IMPALASettings,
)
from schola.scripts.rllib.train.settings import RllibScriptSettings, TrainingSettings
from schola.scripts.common.settings import (
    UnrealProjectSimulatorConfig,
    ActivationFunctionEnum,
)


class _PolicyMappingEnv(BaseRayEnv):
    """Minimal concrete BaseRayEnv for policy mapping helper tests."""

    def _init_agent_tracking(self):
        pass

    def _define_environment(self):
        pass

    def reset(self, **kwargs):
        pass

    def step(self, actions):
        pass


@pytest.fixture
def mock_main(mocker):
    """Mock the main training function to prevent actual training."""
    return mocker.patch("schola.scripts.rllib.train.train.main")


@pytest.fixture
def mock_app(mock_main):
    """Build a fresh app with mocked main (no global injection)."""
    app = App(name="train", help="Train a Model using ray")
    logger = logging.getLogger(__name__)
    app = RllibTrainCommand(app, RllibScriptSettings, mock_main, logger).make()
    return app


def test_agent_type_policy_mapping_fn():
    """AgentType groups compatible agents and falls back to agent IDs."""
    env = object.__new__(_PolicyMappingEnv)
    env.id_manager = IdManager(
        [["Tagger_0", "Tagger_1", "Runner_0", "Solo_0"]],
        {
            0: {
                "Tagger_0": "Tagger",
                "Tagger_1": "Tagger",
                "Runner_0": " Runner ",
                "Solo_0": "",
            }
        },
    )
    policy_mapping_fn = env.make_policy_mapping_fn()

    assert policy_mapping_fn("Tagger_0") == "Tagger"
    assert policy_mapping_fn("Tagger_1") == "Tagger"
    assert policy_mapping_fn("Runner_0") == "Runner"
    assert policy_mapping_fn("Solo_0") == "Solo_0"
    assert policy_mapping_fn("Unknown_0") == "Unknown_0"


def test_get_restored_env_steps_reads_checkpoint_state(tmp_path):
    """Restored RLlib steps are read from trusted checkpoint metadata."""
    checkpoint_dir = tmp_path / "checkpoint_000001"
    env_runner_dir = checkpoint_dir / "env_runner"
    env_runner_dir.mkdir(parents=True)
    with (env_runner_dir / "state.pkl").open("wb") as state_file:
        pickle.dump({"num_env_steps_sampled_lifetime": 1234}, state_file)

    assert _get_restored_env_steps(checkpoint_dir) == 1234


def test_get_restored_env_steps_missing_state_falls_back(tmp_path):
    """Missing env runner state keeps --timesteps as the lifetime target."""
    checkpoint_dir = tmp_path / "checkpoint_000001"
    checkpoint_dir.mkdir()

    assert _get_restored_env_steps(checkpoint_dir) == 0


def test_get_restored_env_steps_invalid_state_file_falls_back(tmp_path):
    """Unreadable checkpoint metadata does not change the stop target."""
    checkpoint_dir = tmp_path / "checkpoint_000001"
    env_runner_dir = checkpoint_dir / "env_runner"
    env_runner_dir.mkdir(parents=True)
    (env_runner_dir / "state.pkl").write_bytes(b"not a pickle")

    assert _get_restored_env_steps(checkpoint_dir) == 0


@pytest.mark.parametrize(
    "restored_value",
    ["not-an-int", -1],
)
def test_get_restored_env_steps_invalid_value_falls_back(tmp_path, restored_value):
    """Invalid restored timestep values are ignored."""
    checkpoint_dir = tmp_path / "checkpoint_000001"
    env_runner_dir = checkpoint_dir / "env_runner"
    env_runner_dir.mkdir(parents=True)
    with (env_runner_dir / "state.pkl").open("wb") as state_file:
        pickle.dump({"num_env_steps_sampled_lifetime": restored_value}, state_file)

    assert _get_restored_env_steps(checkpoint_dir) == 0


def test_make_stop_criterion_uses_absolute_timesteps(tmp_path):
    """Stop target is --timesteps as lifetime cap (default; same fresh vs resume)."""
    checkpoint_dir = tmp_path / "checkpoint_000001"
    env_runner_dir = checkpoint_dir / "env_runner"
    env_runner_dir.mkdir(parents=True)
    with (env_runner_dir / "state.pkl").open("wb") as state_file:
        pickle.dump({"num_env_steps_sampled_lifetime": 1000}, state_file)

    assert _make_stop_criterion(250, checkpoint_dir) == {
        "num_env_steps_sampled_lifetime": 250,
    }
    assert _make_stop_criterion(250, None) == {
        "num_env_steps_sampled_lifetime": 250,
    }


def test_make_stop_criterion_adds_restored_steps_when_reset(tmp_path):
    """When reset_timestep=True, stop target is restored_steps + timesteps."""
    checkpoint_dir = tmp_path / "checkpoint_000001"
    env_runner_dir = checkpoint_dir / "env_runner"
    env_runner_dir.mkdir(parents=True)
    with (env_runner_dir / "state.pkl").open("wb") as state_file:
        pickle.dump({"num_env_steps_sampled_lifetime": 1000}, state_file)

    assert _make_stop_criterion(250, checkpoint_dir, reset_timestep=True) == {
        "num_env_steps_sampled_lifetime": 1250,
    }


def test_make_stop_criterion_reset_no_checkpoint_uses_timesteps():
    """When reset_timestep=True but no checkpoint is given, stop target is just --timesteps."""
    assert _make_stop_criterion(500, None, reset_timestep=True) == {
        "num_env_steps_sampled_lifetime": 500,
    }


def test_ppo_default_arguments(mock_app, mock_main):
    """Test PPO command with default arguments."""
    mock_app.meta(["ppo"], result_action="return_value")

    # Verify main was called once
    mock_main.assert_called_once()

    # Extract the arguments passed to main
    args: RllibScriptSettings = mock_main.call_args[0][0]

    # Verify it's the correct dataclass
    assert isinstance(args, RllibScriptSettings)

    # Verify algorithm settings
    assert isinstance(args.algorithm_settings, PPOSettings)
    assert args.algorithm_settings.gae_lambda == 0.95
    assert args.algorithm_settings.clip_param == 0.2
    assert args.algorithm_settings.use_gae is True

    # Verify default training settings
    assert isinstance(args.training_settings, TrainingSettings)
    assert args.training_settings.timesteps == 3000
    assert args.training_settings.learning_rate == 0.0003
    assert args.training_settings.gamma == 0.99

    # Verify default simulator is external and num_simulators defaults to 1
    from schola.scripts.common.settings import ExternalSimulatorConfig

    assert isinstance(
        args.environment_settings.simulator_settings, ExternalSimulatorConfig
    )
    assert args.environment_settings.simulator_settings.num_simulators == 1


def test_ppo_custom_training_parameters(mock_app, mock_main):
    """Test PPO command with custom training parameters."""
    mock_app.meta(
        [
            "ppo",
            "--timesteps",
            "10000",
            "--learning-rate",
            "0.001",
            "--gamma",
            "0.95",
            "--minibatch-size",
            "64",
            "--train-batch-size-per-learner",
            "256",
            "--num-sgd-iter",
            "10",
        ],
        result_action="return_value",
    )

    mock_main.assert_called_once()
    args: RllibScriptSettings = mock_main.call_args[0][0]

    # Verify custom training settings
    assert args.training_settings.timesteps == 10000
    assert args.training_settings.learning_rate == 0.001
    assert args.training_settings.gamma == 0.95
    assert args.training_settings.minibatch_size == 64
    assert args.training_settings.train_batch_size_per_learner == 256
    assert args.training_settings.num_epochs == 10


def test_ppo_custom_algorithm_parameters(mock_app, mock_main):
    """Test PPO command with custom PPO-specific parameters."""
    mock_app.meta(
        ["ppo", "--gae-lambda", "0.90", "--clip-param", "0.3", "--no-use-gae"],
        result_action="return_value",
    )

    mock_main.assert_called_once()
    args: RllibScriptSettings = mock_main.call_args[0][0]

    # Verify custom PPO settings
    assert isinstance(args.algorithm_settings, PPOSettings)
    assert args.algorithm_settings.gae_lambda == 0.90
    assert args.algorithm_settings.clip_param == 0.3
    assert args.algorithm_settings.use_gae is False


def test_sac_default_arguments(mock_app, mock_main):
    """Test SAC command with default arguments."""
    mock_app.meta(["sac"], result_action="return_value")

    mock_main.assert_called_once()
    args: RllibScriptSettings = mock_main.call_args[0][0]

    # Verify it's using SAC settings
    assert isinstance(args.algorithm_settings, SACSettings)
    assert args.algorithm_settings.tau == 0.005
    assert args.algorithm_settings.target_entropy == "auto"
    assert args.algorithm_settings.initial_alpha == 1.0
    assert args.algorithm_settings.n_step == 1
    assert args.algorithm_settings.twin_q is True


def test_sac_custom_parameters(mock_app, mock_main):
    """Test SAC command with custom SAC-specific parameters."""
    mock_app.meta(
        [
            "sac",
            "--tau",
            "0.01",
            "--initial-alpha",
            "0.5",
            "--n-step",
            "3",
            "--no-twin-q",
        ],
        result_action="return_value",
    )

    mock_main.assert_called_once()
    args: RllibScriptSettings = mock_main.call_args[0][0]

    # Verify custom SAC settings
    assert isinstance(args.algorithm_settings, SACSettings)
    assert args.algorithm_settings.tau == 0.01
    assert args.algorithm_settings.initial_alpha == 0.5
    assert args.algorithm_settings.n_step == 3
    assert args.algorithm_settings.twin_q is False


def test_impala_default_arguments(mock_app, mock_main):
    """Test IMPALA command with default arguments."""
    mock_app.meta(["impala"], result_action="return_value")

    mock_main.assert_called_once()
    args: RllibScriptSettings = mock_main.call_args[0][0]

    # Verify it's using IMPALA settings
    assert isinstance(args.algorithm_settings, IMPALASettings)
    assert args.algorithm_settings.vtrace is True
    assert args.algorithm_settings.vtrace_clip_rho_threshold == 1.0
    assert args.algorithm_settings.vtrace_clip_pg_rho_threshold == 1.0


def test_impala_custom_parameters(mock_app, mock_main):
    """Test IMPALA command with custom IMPALA-specific parameters."""
    mock_app.meta(
        [
            "impala",
            "--no-vtrace",
            "--vtrace-clip-rho-threshold",
            "2.0",
            "--vtrace-clip-pg-rho-threshold",
            "1.5",
        ],
        result_action="return_value",
    )

    mock_main.assert_called_once()
    args: RllibScriptSettings = mock_main.call_args[0][0]

    # Verify custom IMPALA settings
    assert isinstance(args.algorithm_settings, IMPALASettings)
    assert args.algorithm_settings.vtrace is False
    assert args.algorithm_settings.vtrace_clip_rho_threshold == 2.0
    assert args.algorithm_settings.vtrace_clip_pg_rho_threshold == 1.5


def test_resource_settings(mock_app, mock_main):
    """Test resource allocation parameters."""
    mock_app.meta(
        [
            "ppo",
            "--num-gpus",
            "2",
            "--num-cpus",
            "8",
            "--num-learners",
            "4",
            "--num-cpus-per-learner",
            "2",
            "--num-gpus-per-learner",
            "1",
        ],
        result_action="return_value",
    )

    mock_main.assert_called_once()
    args: RllibScriptSettings = mock_main.call_args[0][0]

    # Verify resource settings
    assert args.resource_settings.num_gpus == 2
    assert args.resource_settings.num_cpus == 8
    assert args.resource_settings.num_learners == 4
    assert args.resource_settings.num_cpus_per_learner == 2
    assert args.resource_settings.num_gpus_per_learner == 1


def test_network_architecture_settings(mock_app, mock_main):
    """Test network architecture parameters."""
    mock_app.meta(
        [
            "ppo",
            "--activation",
            "TanH",
            "--use-lstm",
            "--lstm-cell-size",
            "128",
            "--max-seq-len",
            "10",
            "--fcnet-hiddens",
            "256",
            "256",
        ],
        result_action="return_value",
    )

    mock_main.assert_called_once()
    args: RllibScriptSettings = mock_main.call_args[0][0]

    # Verify network architecture settings
    assert (
        args.network_architecture_settings.activation == ActivationFunctionEnum.TanH
    ), "Activation should be TanH"
    assert (
        args.network_architecture_settings.use_lstm is True
    ), "Use LSTM should be True"
    assert (
        args.network_architecture_settings.lstm_cell_size == 128
    ), "LSTM cell size should be 128"
    assert (
        args.network_architecture_settings.max_seq_len == 10
    ), "max_seq_len should be 10"
    assert args.network_architecture_settings.fcnet_hiddens == [
        256,
        256,
    ], "FCNet hiddens should be [256, 256]"


def test_logging_settings(mock_app, mock_main):
    """Test logging verbosity parameters."""
    mock_app.meta(
        ["ppo", "--schola-verbosity", "2", "--rllib-verbosity", "3"],
        result_action="return_value",
    )

    mock_main.assert_called_once()
    args: RllibScriptSettings = mock_main.call_args[0][0]

    # Verify logging settings
    assert args.logging_settings.schola_verbosity == 2
    assert args.logging_settings.rllib_verbosity == 3


def test_checkpoint_settings(mock_app, mock_main, tmp_path):
    """Test checkpoint configuration parameters."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    mock_app.meta(
        [
            "ppo",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--save-freq",
            "1000",
            "--enable-checkpoints",
            "--save-final-policy",
            "--export-onnx",
        ],
        result_action="return_value",
    )

    mock_main.assert_called_once()
    args: RllibScriptSettings = mock_main.call_args[0][0]

    # Verify checkpoint settings
    assert args.checkpoint_settings.checkpoint_dir == checkpoint_dir
    assert args.checkpoint_settings.save_freq == 1000
    assert args.checkpoint_settings.enable_checkpoints is True
    assert args.checkpoint_settings.save_final_policy is True
    assert args.checkpoint_settings.export_onnx is True


def test_ppo_with_executable_simulator(mock_app, mock_main, tmp_path):
    """Test executable simulator type is correctly parsed."""
    executable_path = tmp_path / "UnrealGame.exe"
    executable_path.touch()  # Create fake executable
    mock_app.meta(
        ["ppo", "executable", "--executable-path", str(executable_path), "--headless"],
        result_action="return_value",
    )

    mock_main.assert_called_once()
    args: RllibScriptSettings = mock_main.call_args[0][0]

    assert isinstance(
        args.environment_settings.simulator_settings, UnrealExecutableSimulatorConfig
    )
    assert (
        args.environment_settings.simulator_settings.executable_path == executable_path
    )


def test_executable_num_simulators_parsed(mock_app, mock_main, tmp_path):
    """Test num_simulators is parsed for executable simulator."""
    executable_path = tmp_path / "UnrealGame.exe"
    executable_path.touch()
    mock_app.meta(
        [
            "ppo",
            "executable",
            "--executable-path",
            str(executable_path),
            "--num-simulators",
            "4",
        ],
        result_action="return_value",
    )
    mock_main.assert_called_once()
    args: RllibScriptSettings = mock_main.call_args[0][0]
    assert isinstance(
        args.environment_settings.simulator_settings, UnrealExecutableSimulatorConfig
    )
    assert args.environment_settings.simulator_settings.num_simulators == 4


def test_project_num_simulators_parsed(mock_app, mock_main, tmp_path):
    """Test num_simulators is parsed for project simulator."""
    uproject = tmp_path / "MyGame.uproject"
    uproject.write_text("{}")
    mock_app.meta(
        [
            "ppo",
            "project",
            str(uproject),
            "--num-simulators",
            "2",
        ],
        result_action="return_value",
    )
    mock_main.assert_called_once()
    args: RllibScriptSettings = mock_main.call_args[0][0]
    assert isinstance(
        args.environment_settings.simulator_settings, UnrealProjectSimulatorConfig
    )
    assert args.environment_settings.simulator_settings.num_simulators == 2


def test_protocol_settings(mock_app, mock_main):
    """Test protocol configuration parameters."""
    mock_app.meta(["ppo", "--port", "12345"], result_action="return_value")

    mock_main.assert_called_once()
    args: RllibScriptSettings = mock_main.call_args[0][0]

    # Verify protocol settings
    assert args.environment_settings.protocol_settings.port == 12345


def test_ppo_env_options_default_is_empty_dict(mock_app, mock_main):
    """Without ``--env-options.k=v`` the field defaults to an empty dict."""
    mock_app.meta(["ppo"], result_action="return_value")
    args: RllibScriptSettings = mock_main.call_args[0][0]
    assert args.environment_settings.env_options == {}


def test_ppo_env_options_dotted_syntax(mock_app, mock_main):
    """Cyclopts dotted syntax populates ``env_options`` with str values."""
    mock_app.meta(
        [
            "ppo",
            "--env-options.level=1",
            "--env-options.curriculum=easy",
        ],
        result_action="return_value",
    )
    args: RllibScriptSettings = mock_main.call_args[0][0]
    assert args.environment_settings.env_options == {
        "level": "1",
        "curriculum": "easy",
    }


def test_options_thread_through_rllib_env_config_recipe(mock_protocol_and_simulator):
    """``env_config["options"]`` survives the RLlib config plumbing the train
    CLI uses.

    ``train.main`` wires CLI ``--env-options.*`` values into the RLlib config via
    the standard ``AlgorithmConfig.environment(env_config={"options": ...})``
    recipe -- the same recipe a user would arrive at by following the RLlib
    docs example:
    https://github.com/ray-project/ray/blob/master/rllib/examples/envs/custom_gym_env.py

    At env-build time RLlib wraps ``config.env_config`` in an ``EnvContext``
    (a ``dict`` subclass) and hands that to the env class. Schola's
    ``BaseRayEnv.__init__`` then reads ``cfg.get("options")`` to seed the
    one-shot options cache. This test mirrors that build path directly and
    asserts the options dict the user passed lands in ``BaseRayEnv._options``.

    Catches regressions where:

    * ``AlgorithmConfig.environment(env_config=...)`` drops or mutates the
      user's dict before it reaches the env constructor,
    * ``EnvContext`` ceases to behave like a plain ``dict`` (so the
      ``cfg.get("options")`` lookup in ``BaseRayEnv.__init__`` would break), or
    * Schola's ``env_config["options"]`` contract drifts away from the shape
      RLlib actually delivers at env-build time.
    """
    protocol, simulator = mock_protocol_and_simulator

    expected_options = {"level": "hard", "schedule": ["a", "b"]}
    config = PPOConfig().environment(env_config={"options": expected_options})

    # The user's env_config must round-trip through .environment() unchanged.
    assert config.env_config == {"options": expected_options}

    # Mirror what RLlib's default env runner does at env-build time: wrap
    # env_config in an EnvContext (dict subclass) and pass it to the env class.
    env_ctx = EnvContext(config.env_config, worker_index=0, num_workers=0, remote=False)
    env = _PolicyMappingEnv(protocol, simulator, env_config=env_ctx)

    assert env._options == expected_options, (
        "Schola's env_config['options'] cache was not populated when the env "
        "was constructed via the RLlib docs recipe; a user following "
        "https://github.com/ray-project/ray/blob/master/rllib/examples/envs/"
        "custom_gym_env.py would see options silently dropped on reset."
    )
    # The deepcopy guarantee from BaseRayEnv.__init__ must still hold:
    # mutating the user's source dict after construction must not leak in.
    expected_options["level"] = "MUTATED"
    assert env._options["level"] == "hard"


def test_multiple_algorithms_return_different_settings(mock_app, mock_main):
    """Test that different algorithm commands create different settings."""
    mock_app.meta(["ppo"], result_action="return_value")
    ppo_args: RllibScriptSettings = deepcopy(mock_main.call_args[0][0])

    mock_main.reset_mock()
    mock_app.meta(["sac"], result_action="return_value")
    sac_args: RllibScriptSettings = deepcopy(mock_main.call_args[0][0])

    mock_main.reset_mock()
    mock_app.meta(["impala"], result_action="return_value")
    impala_args: RllibScriptSettings = deepcopy(mock_main.call_args[0][0])

    mock_main.reset_mock()
    mock_app.meta(["appo"], result_action="return_value")
    appo_args: RllibScriptSettings = deepcopy(mock_main.call_args[0][0])

    # Verify different algorithm types are set correctly
    assert isinstance(ppo_args.algorithm_settings, PPOSettings)
    assert isinstance(sac_args.algorithm_settings, SACSettings)
    assert isinstance(impala_args.algorithm_settings, IMPALASettings)
    assert isinstance(appo_args.algorithm_settings, APPOSettings)


def test_complex_configuration(mock_app, mock_main, tmp_path):
    """Test a complex configuration with many parameters."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    mock_app.meta(
        [
            "ppo",
            # Training settings
            "--timesteps",
            "50000",
            "--learning-rate",
            "0.0005",
            "--gamma",
            "0.98",
            "--minibatch-size",
            "32",
            "--train-batch-size-per-learner",
            "128",
            "--num-sgd-iter",
            "8",
            # PPO settings
            "--gae-lambda",
            "0.92",
            "--clip-param",
            "0.25",
            # Resource settings
            "--num-gpus",
            "1",
            "--num-cpus",
            "4",
            # Network architecture
            "--activation",
            "ReLU",
            # Logging
            "--schola-verbosity",
            "1",
            "--rllib-verbosity",
            "2",
            # Checkpoint settings
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--save-freq",
            "5000",
            "--enable-checkpoints",
            # Protocol settings
            "--port",
            "50051",
        ],
        result_action="return_value",
    )

    mock_main.assert_called_once()
    args: RllibScriptSettings = mock_main.call_args[0][0]

    # Verify all settings were applied correctly
    assert args.training_settings.timesteps == 50000
    assert args.training_settings.learning_rate == 0.0005
    assert args.training_settings.gamma == 0.98
    assert args.algorithm_settings.gae_lambda == 0.92  # type: ignore
    assert args.algorithm_settings.clip_param == 0.25  # type: ignore
    assert args.resource_settings.num_gpus == 1
    assert args.resource_settings.num_cpus == 4
    assert args.network_architecture_settings.activation == ActivationFunctionEnum.ReLU
    assert args.logging_settings.schola_verbosity == 1
    assert args.logging_settings.rllib_verbosity == 2
    assert args.checkpoint_settings.save_freq == 5000
    assert args.environment_settings.protocol_settings.port == 50051


@pytest.mark.xdist_group(name="ray-cluster")
def test_train_cli_with_unreal_editor(
    make_vec_env_server, make_env, ray_cluster, tmp_path
):
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    env_server_port = make_vec_env_server(
        [make_env("CartPole-v1", i) for i in range(2)]
    )

    train_app.meta(
        [
            "ppo",
            # Training settings
            "--timesteps",
            "5000",
            "--learning-rate",
            "0.0005",
            "--gamma",
            "0.98",
            "--minibatch-size",
            "32",
            "--train-batch-size-per-learner",
            "128",
            "--num-epochs",
            "8",
            # PPO settings
            "--gae-lambda",
            "0.92",
            "--clip-param",
            "0.25",
            # Resource settings
            "--using-cluster",
            # Network architecture
            "--activation",
            "ReLU",
            # Logging
            "--schola-verbosity",
            "2",
            "--rllib-verbosity",
            "2",
            # Checkpoint settings
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--no-enable-checkpoints",
            "--no-export-onnx",
            # Protocol settings
            "--port",
            f"{env_server_port}",
        ],
        result_action="return_value",
    )
