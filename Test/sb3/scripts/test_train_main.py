# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for SB3 train script ``main()`` branches and ``warn_if_small_image_observation``."""

from __future__ import annotations

import builtins
import logging
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import pytest
from cyclopts import App
from stable_baselines3.common.callbacks import CheckpointCallback

from schola.scripts.common.settings import EnvironmentSettings, GrpcProtocolConfig
from schola.scripts.sb3.train.settings import (
    PPOTrainSettings,
    SACTrainSettings,
    Sb3CheckpointSettings,
    Sb3LoggingSettings,
    Sb3ResumeSettings,
    Sb3TrainScriptSettings,
    Sb3TrainingSettings,
)
from schola.scripts.sb3.train.train import (
    MetaTrainSB3Command,
    main,
    warn_if_small_image_observation,
)


@pytest.fixture
def mock_main(mocker):
    return mocker.patch("schola.scripts.sb3.train.train.main")


@pytest.fixture
def mock_app(mock_main):
    app_obj = App(name="train", help="Train a model using StableBaselines3")
    logger = logging.getLogger("test_train_main_branches")
    return (
        MetaTrainSB3Command(app_obj, Sb3TrainScriptSettings, mock_main, logger)
        .make()
        .meta
    )


@pytest.fixture(scope="function")
def make_train_args(tmp_path):
    """Factory fixture for ``Sb3TrainScriptSettings`` that wraps the in-file
    ``_train_args`` helper. Pre-existing tests use ``_train_args`` directly; new
    tests should prefer this fixture for a pytest-idiomatic call shape.
    """

    def _factory(**kwargs) -> Sb3TrainScriptSettings:
        return _train_args(tmp_path, **kwargs)

    return _factory


@pytest.fixture(scope="function")
def mock_vec_env():
    """A fresh minimal MagicMock ``VecEnv``, function-scoped so per-test call
    counts (``set_options``, ``close``) cannot bleed across tests."""
    return _mock_vec_env()


@pytest.fixture(scope="function")
def patch_sb3_ppo_train_deps(mocker, mock_vec_env):
    """Patch the dependencies that ``schola.scripts.sb3.train.train.main``
    reaches into for the PPO "no-checkpoint" happy path.

    Yields the underlying mock ``VecEnv`` so tests can assert against it
    (``mock_env.set_options.assert_*``, etc.).
    """
    mocker.patch("schola.sb3.env.VecEnv", autospec=True, return_value=mock_vec_env)

    mock_model = MagicMock()
    mock_model.get_vec_normalize_env.return_value = None
    mock_ppo = mocker.patch("stable_baselines3.PPO")
    mock_ppo.load.side_effect = Exception("no checkpoint")
    mock_ppo.return_value = mock_model

    return mock_vec_env


def _train_args(
    tmp_path: Path,
    *,
    timesteps: int = 8,
    disable_eval: bool = True,
    enable_tensorboard: bool = False,
    enable_checkpoints: bool = False,
    save_final_policy: bool = False,
    export_onnx: bool = False,
    resume_from: Path | None = None,
    load_vecnormalize: Path | None = None,
    load_replay_buffer: Path | None = None,
    pbar: bool = False,
    algorithm_settings: PPOTrainSettings | SACTrainSettings | None = None,
    policy_kwargs_network: bool = False,
    env_options: dict[str, str] | None = None,
) -> Sb3TrainScriptSettings:
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume_settings = Sb3ResumeSettings(
        resume_from=resume_from,
        load_vecnormalize=load_vecnormalize,
        load_replay_buffer=load_replay_buffer,
    )
    defaults = Sb3TrainScriptSettings()
    if policy_kwargs_network:
        net = replace(
            defaults.network_architecture_settings,
            policy_parameters=[64, 64],
        )
    else:
        net = defaults.network_architecture_settings

    log_dir = tmp_path / "tb_logs"
    logging_settings = replace(
        Sb3LoggingSettings(),
        enable_tensorboard=enable_tensorboard,
        log_dir=log_dir,
    )

    return Sb3TrainScriptSettings(
        environment_settings=EnvironmentSettings(
            protocol_settings=GrpcProtocolConfig(port=1, url="localhost"),
            env_options=env_options or {},
        ),
        algorithm_settings=algorithm_settings or PPOTrainSettings(),
        training_settings=replace(
            Sb3TrainingSettings(),
            timesteps=timesteps,
            disable_eval=disable_eval,
            pbar=pbar,
        ),
        logging_settings=logging_settings,
        resume_settings=resume_settings,
        checkpoint_settings=replace(
            Sb3CheckpointSettings(),
            checkpoint_dir=ckpt_dir,
            enable_checkpoints=enable_checkpoints,
            save_freq=2,
            save_final_policy=save_final_policy,
            export_onnx=export_onnx,
        ),
        network_architecture_settings=net,
    )


def _mock_vec_env():
    env = MagicMock()
    env.num_envs = 1
    env.observation_space = gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
    env.action_space = gym.spaces.Discrete(2)
    env.close = MagicMock()
    return env


@pytest.mark.parametrize(
    "obs_space,threshold,calls",
    [
        (gym.spaces.Box(0, 1, (32, 32), dtype=np.float32), 64, True),
        (gym.spaces.Box(0, 1, (3, 48, 48), dtype=np.float32), 64, True),
        (gym.spaces.Box(0, 1, (128, 128), dtype=np.float32), 64, False),
        (gym.spaces.Box(0, 1, (10,), dtype=np.float32), 64, False),
    ],
)
def test_warn_if_small_image_observation(obs_space, threshold, calls):
    with patch("schola.scripts.sb3.train.train.print_error") as mock_err:
        warn_if_small_image_observation(obs_space, threshold=threshold)
        if calls:
            mock_err.assert_called_once()
        else:
            mock_err.assert_not_called()


def test_warn_if_small_image_nested_dict():
    inner = gym.spaces.Box(0, 1, (20, 20), dtype=np.float32)
    obs = gym.spaces.Dict({"rgb": inner})
    with patch("schola.scripts.sb3.train.train.print_error") as mock_err:
        warn_if_small_image_observation(obs, threshold=64)
        mock_err.assert_called_once()


@patch(
    "stable_baselines3.common.vec_env.vec_monitor.VecMonitor", side_effect=lambda e: e
)
@patch("stable_baselines3.common.evaluation.evaluate_policy", return_value=(1.5, 0.25))
@patch("schola.sb3.env.VecEnv")
@patch("stable_baselines3.PPO")
def test_main_runs_eval_when_enabled(mock_ppo, mock_vec_cls, mock_eval, _vm, tmp_path):
    mock_env = _mock_vec_env()
    mock_vec_cls.return_value = mock_env
    mock_model = MagicMock()
    mock_model.get_vec_normalize_env.return_value = None
    mock_ppo.load.side_effect = Exception("no checkpoint")
    mock_ppo.return_value = mock_model

    args = _train_args(tmp_path, timesteps=4, disable_eval=False)
    result = main(args)

    mock_eval.assert_called_once()
    assert result == (1.5, 0.25)
    mock_env.close.assert_called()


@patch("stable_baselines3.common.evaluation.evaluate_policy")
@patch("schola.sb3.env.VecEnv")
@patch("stable_baselines3.PPO")
def test_main_skips_eval_when_disabled(mock_ppo, mock_vec_cls, mock_eval, tmp_path):
    mock_env = _mock_vec_env()
    mock_vec_cls.return_value = mock_env
    mock_model = MagicMock()
    mock_model.get_vec_normalize_env.return_value = None
    mock_ppo.load.side_effect = Exception("no checkpoint")
    mock_ppo.return_value = mock_model

    args = _train_args(tmp_path, timesteps=4, disable_eval=True)
    result = main(args)

    mock_eval.assert_not_called()
    assert result is None
    mock_env.close.assert_called()


@patch("schola.sb3.env.VecEnv")
@patch("stable_baselines3.PPO")
def test_main_resume_load_failure_trains_from_scratch(mock_ppo, mock_vec_cls, tmp_path):
    mock_env = _mock_vec_env()
    mock_vec_cls.return_value = mock_env
    ckpt = tmp_path / "bad.zip"
    ckpt.write_bytes(b"not a zip")

    mock_model = MagicMock()
    mock_model.get_vec_normalize_env.return_value = None
    mock_ppo.load.side_effect = RuntimeError("corrupt")
    mock_ppo.return_value = mock_model

    args = _train_args(
        tmp_path,
        timesteps=2,
        resume_from=ckpt,
    )
    main(args)

    mock_ppo.load.assert_called_once()
    mock_ppo.assert_called_once()
    mock_model.learn.assert_called_once()


@patch("schola.sb3.env.VecEnv")
@patch("stable_baselines3.PPO")
def test_main_resume_load_success_skips_new_model(mock_ppo, mock_vec_cls, tmp_path):
    mock_env = _mock_vec_env()
    mock_vec_cls.return_value = mock_env
    ckpt = tmp_path / "ok.zip"
    ckpt.write_bytes(b"x")

    mock_model = MagicMock()
    mock_model.get_vec_normalize_env.return_value = None
    mock_ppo.load.return_value = mock_model

    args = _train_args(tmp_path, timesteps=2, resume_from=ckpt)
    main(args)

    mock_ppo.load.assert_called_once()
    mock_ppo.assert_not_called()
    mock_model.learn.assert_called_once()


@patch("stable_baselines3.common.vec_env.VecNormalize")
@patch("schola.sb3.env.VecEnv")
@patch("stable_baselines3.PPO")
def test_main_load_vecnormalize_on_resume(mock_ppo, mock_vec_cls, mock_vn, tmp_path):
    mock_env = _mock_vec_env()
    mock_vec_cls.return_value = mock_env
    mock_model = MagicMock()
    mock_model.get_vec_normalize_env.return_value = None
    mock_ppo.load.side_effect = Exception("x")
    mock_ppo.return_value = mock_model

    vn_path = tmp_path / "vn.pkl"
    vn_path.write_bytes(b"{}")
    mock_vn.load.return_value = MagicMock()

    args = _train_args(
        tmp_path,
        timesteps=2,
        load_vecnormalize=vn_path,
    )
    main(args)

    mock_vn.load.assert_called_once()


@patch("schola.sb3.env.VecEnv")
@patch("stable_baselines3.PPO")
def test_main_save_final_and_export_onnx(mock_ppo, mock_vec_cls, tmp_path):
    mock_env = _mock_vec_env()
    mock_vec_cls.return_value = mock_env
    mock_model = MagicMock()
    mock_model.get_vec_normalize_env.return_value = None
    mock_ppo.load.side_effect = Exception("x")
    mock_ppo.return_value = mock_model

    args = _train_args(
        tmp_path,
        timesteps=2,
        save_final_policy=True,
        export_onnx=True,
    )
    with patch("schola.sb3.export.save_model_as_onnx") as mock_onnx:
        main(args)

    mock_model.save.assert_called_once()
    mock_onnx.assert_called_once()


@patch("schola.sb3.env.VecEnv")
@patch("stable_baselines3.PPO")
def test_main_enable_checkpoints_passes_callback(mock_ppo, mock_vec_cls, tmp_path):
    mock_env = _mock_vec_env()
    mock_vec_cls.return_value = mock_env
    mock_model = MagicMock()
    mock_model.get_vec_normalize_env.return_value = None
    mock_ppo.load.side_effect = Exception("x")
    mock_ppo.return_value = mock_model

    args = _train_args(tmp_path, timesteps=4, enable_checkpoints=True)
    main(args)

    kwargs = mock_model.learn.call_args.kwargs
    cbs = kwargs.get("callback") or []
    assert any(isinstance(c, CheckpointCallback) for c in cbs)


def test_main_forwards_env_options_to_env(patch_sb3_ppo_train_deps, make_train_args):
    """When ``environment_settings.env_options`` is non-empty, ``main`` should
    forward it to the env via ``set_options`` before ``model.learn``."""
    opts = {"level": "1", "curriculum": "easy"}
    main(make_train_args(timesteps=2, env_options=opts))

    patch_sb3_ppo_train_deps.set_options.assert_called_once_with(options=opts)


def test_main_skips_set_options_when_env_options_empty(
    patch_sb3_ppo_train_deps, make_train_args
):
    """When ``environment_settings.env_options`` is empty, ``main`` should not
    call ``set_options``."""
    main(make_train_args(timesteps=2, env_options={}))

    patch_sb3_ppo_train_deps.set_options.assert_not_called()


@patch("schola.sb3.env.VecEnv")
@patch("stable_baselines3.PPO")
def test_main_ppo_with_dict_obs_sets_policy_kwargs(mock_ppo, mock_vec_cls, tmp_path):
    mock_env = _mock_vec_env()
    mock_env.observation_space = gym.spaces.Dict(
        {"x": gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)}
    )
    mock_vec_cls.return_value = mock_env
    mock_model = MagicMock()
    mock_model.get_vec_normalize_env.return_value = None
    mock_ppo.load.side_effect = Exception("x")
    mock_ppo.return_value = mock_model

    args = _train_args(
        tmp_path,
        timesteps=2,
        policy_kwargs_network=True,
    )
    main(args)

    call_kw = mock_ppo.call_args.kwargs
    assert call_kw["policy"] == "MultiInputPolicy"
    pk = call_kw.get("policy_kwargs") or {}
    assert pk.get("features_extractor_kwargs") == {"normalized_image": True}


@patch("schola.sb3.env.VecEnv")
@patch("stable_baselines3.SAC")
def test_main_sac_load_replay_buffer_skipped_without_method(
    mock_sac, mock_vec_cls, tmp_path
):
    mock_env = _mock_vec_env()
    mock_vec_cls.return_value = mock_env
    mock_model = MagicMock(
        spec=["learn", "set_logger", "get_vec_normalize_env", "save"],
    )
    mock_model.get_vec_normalize_env.return_value = None
    mock_sac.load.side_effect = Exception("x")
    mock_sac.return_value = mock_model

    buf = tmp_path / "buf.pkl"
    buf.write_bytes(b"x")

    args = _train_args(
        tmp_path,
        timesteps=2,
        algorithm_settings=SACTrainSettings(
            buffer_size=10000,
            learning_starts=100,
            batch_size=256,
        ),
        load_replay_buffer=buf,
    )
    main(args)

    mock_model.learn.assert_called_once()


def test_main_disables_tensorboard_when_import_fails(tmp_path):
    real_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tensorboard":
            raise ImportError("simulated missing tensorboard")
        return real_import(name, globals, locals, fromlist, level)

    mock_env = _mock_vec_env()
    mock_model = MagicMock()
    mock_model.get_vec_normalize_env.return_value = None

    args = _train_args(
        tmp_path,
        timesteps=2,
        enable_tensorboard=True,
    )
    assert args.logging_settings.enable_tensorboard is True

    with (
        patch("builtins.__import__", side_effect=_import),
        patch("schola.sb3.env.VecEnv", return_value=mock_env),
        patch("stable_baselines3.PPO") as mock_ppo,
    ):
        mock_ppo.load.side_effect = Exception("x")
        mock_ppo.return_value = mock_model
        main(args)

    assert args.logging_settings.enable_tensorboard is False


def test_main_disables_pbar_when_tqdm_missing(tmp_path):
    real_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tqdm":
            raise ModuleNotFoundError("simulated missing tqdm")
        return real_import(name, globals, locals, fromlist, level)

    mock_env = _mock_vec_env()
    mock_model = MagicMock()
    mock_model.get_vec_normalize_env.return_value = None

    args = _train_args(tmp_path, timesteps=2, pbar=True)
    assert args.training_settings.pbar is True

    with (
        patch("builtins.__import__", side_effect=_import),
        patch("schola.sb3.env.VecEnv", return_value=mock_env),
        patch("stable_baselines3.PPO") as mock_ppo,
    ):
        mock_ppo.load.side_effect = Exception("x")
        mock_ppo.return_value = mock_model
        main(args)

    assert args.training_settings.pbar is False


def test_cli_ppo_invalid_batch_vs_n_steps_raises(mock_app, mock_main):
    """Invalid PPO hyperparameters surface as ValueError when the command runs."""
    command, bound, _ = mock_app.parse_args(
        ["ppo", "--n-steps", "100", "--batch-size", "33"],
        exit_on_error=False,
    )
    with pytest.raises(ValueError, match="batch_size"):
        command(*bound.args, **bound.kwargs)


def test_cli_sac_invalid_buffer_size_vs_batch_raises(mock_app, mock_main):
    command, bound, _ = mock_app.parse_args(
        ["sac", "--buffer-size", "100", "--batch-size", "256"],
        exit_on_error=False,
    )
    with pytest.raises(ValueError, match="batch_size"):
        command(*bound.args, **bound.kwargs)


def test_sac_train_settings_rejects_unknown_replay_buffer_kwargs():
    """SAC dataclass validation rejects unsupported replay_buffer_kwargs keys."""
    with pytest.raises(KeyError, match="Unsupported keys"):
        SACTrainSettings(replay_buffer_kwargs={"unknown": True})
