# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for the SB3 eval CLI."""

import logging
import math
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest
from cyclopts import App

from schola.scripts.common.settings import EnvironmentSettings, GrpcProtocolConfig
from schola.scripts.sb3.eval.eval import MetaEvalSB3Command, main as eval_main
from schola.scripts.sb3.eval.settings import Sb3EvalScriptSettings


@pytest.fixture
def dummy_sb3_policy_zip(tmp_path: Path) -> Path:
    """
    Train a tiny PPO on ``CartPole-v1`` (in-process ``DummyVecEnv``) and save ``.zip``.

    Observation and action spaces match the gRPC CartPole vec server used in tests.
    """
    pytest.importorskip("stable_baselines3")
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    save_stem = tmp_path / "dummy_cartpole_policy"
    venv = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    try:
        model = PPO(
            "MlpPolicy",
            venv,
            n_steps=64,
            batch_size=64,
            ent_coef=0.0,
            verbose=0,
        )
        model.learn(total_timesteps=512, progress_bar=False)
        model.save(str(save_stem))
    finally:
        venv.close()
    out = save_stem.with_suffix(".zip")
    assert out.is_file()
    return out


@pytest.fixture
def mock_main(mocker):
    return mocker.patch("schola.scripts.sb3.eval.eval.main")


@pytest.fixture
def mock_eval_app(mock_main):
    base = App(name="eval", help="Evaluate a trained Stable-Baselines3 policy")
    logger = logging.getLogger(__name__)
    return MetaEvalSB3Command(base, Sb3EvalScriptSettings, mock_main, logger).make()


@pytest.fixture
def sb3_eval_meta_app():
    """Real ``schola sb3 eval`` Cyclopts meta-app (invokes ``eval_main``)."""
    base = App(name="eval", help="Evaluate a trained Stable-Baselines3 policy")
    logger = logging.getLogger(__name__)
    return MetaEvalSB3Command(base, Sb3EvalScriptSettings, eval_main, logger).make()


def test_eval_cli_forwards_checkpoint_and_defaults(
    mock_eval_app, mock_main, tmp_path: Path
):
    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"x")
    mock_eval_app.meta(
        ["ppo", "--checkpoint", str(checkpoint)], result_action="return_value"
    )
    mock_main.assert_called_once()
    args = mock_main.call_args[0][0]
    assert isinstance(args, Sb3EvalScriptSettings)
    assert args.checkpoint == checkpoint
    assert args.n_eval_episodes == 10
    assert args.deterministic is True


def test_eval_cli_custom_episodes(mock_eval_app, mock_main, tmp_path: Path):
    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"x")
    mock_eval_app.meta(
        [
            "ppo",
            "--checkpoint",
            str(checkpoint),
            "--n-eval-episodes",
            "3",
            "--no-deterministic",
        ],
        result_action="return_value",
    )
    args = mock_main.call_args[0][0]
    assert args.n_eval_episodes == 3
    assert args.deterministic is False


def test_eval_cli_env_options_default_is_empty_dict(
    mock_eval_app, mock_main, tmp_path: Path
):
    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"x")
    mock_eval_app.meta(
        ["ppo", "--checkpoint", str(checkpoint)],
        result_action="return_value",
    )
    args = mock_main.call_args[0][0]
    assert args.environment_settings.env_options == {}


def test_eval_cli_env_options_dotted_syntax(mock_eval_app, mock_main, tmp_path: Path):
    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"x")
    mock_eval_app.meta(
        [
            "ppo",
            "--checkpoint",
            str(checkpoint),
            "--env-options.level=1",
            "--env-options.curriculum=easy",
        ],
        result_action="return_value",
    )
    args = mock_main.call_args[0][0]
    assert args.environment_settings.env_options == {
        "level": "1",
        "curriculum": "easy",
    }


def test_sb3_eval_main_on_real_vec_env(dummy_sb3_policy_zip, make_vec_env_server):
    pytest.importorskip("stable_baselines3")
    import gymnasium as gym

    port = make_vec_env_server([lambda: gym.make("CartPole-v1")])
    args = Sb3EvalScriptSettings(
        checkpoint=dummy_sb3_policy_zip,
        n_eval_episodes=2,
        environment_settings=EnvironmentSettings(
            protocol_settings=GrpcProtocolConfig(url="localhost", port=port),
        ),
    )
    mean_r, std_r = eval_main(args)
    assert isinstance(mean_r, (int, float)) and isinstance(std_r, (int, float))
    assert math.isfinite(mean_r) and math.isfinite(std_r)


def test_sb3_eval_cli_on_real_vec_env(
    dummy_sb3_policy_zip, make_vec_env_server, sb3_eval_meta_app
):
    """End-to-end ``schola sb3 eval`` parsing and ``eval_main`` on a live gRPC vec env."""
    pytest.importorskip("stable_baselines3")
    import gymnasium as gym

    port = make_vec_env_server([lambda: gym.make("CartPole-v1")])
    mean_r, std_r = sb3_eval_meta_app.meta(
        [
            "ppo",
            "--checkpoint",
            str(dummy_sb3_policy_zip),
            "--port",
            str(port),
            "--url",
            "localhost",
            "--n-eval-episodes",
            "2",
        ],
        result_action="return_value",
    )
    assert math.isfinite(mean_r) and math.isfinite(std_r)


@pytest.fixture(scope="function")
def mock_vec_env_for_eval():
    """Minimal MagicMock ``VecEnv`` matching what eval ``main`` introspects."""
    env = MagicMock()
    env.num_envs = 1
    env.observation_space = gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
    env.action_space = gym.spaces.Discrete(2)
    env.close = MagicMock()
    return env


@pytest.fixture(scope="function")
def patch_sb3_eval_deps(mocker, mock_vec_env_for_eval):
    """Patch the SB3 dependencies that eval ``main`` reaches into.

    Returns the underlying mock ``VecEnv`` so tests can make per-test assertions
    against it (e.g. ``mock_env.set_options.assert_*``).
    """
    mocker.patch("schola.sb3.env.VecEnv", return_value=mock_vec_env_for_eval)

    mock_model = MagicMock()
    mock_ppo = mocker.patch("stable_baselines3.PPO")
    mock_ppo.load.return_value = mock_model
    mocker.patch(
        "stable_baselines3.common.evaluation.evaluate_policy",
        return_value=([1.0, 1.0], [10, 10]),
    )
    mocker.patch(
        "stable_baselines3.common.vec_env.vec_monitor.VecMonitor",
        side_effect=lambda e: e,
    )
    return mock_vec_env_for_eval


@pytest.fixture(scope="function")
def make_eval_args(tmp_path: Path):
    """Factory fixture for ``Sb3EvalScriptSettings`` with sensible defaults.

    Tests pass only the fields they care about (typically just ``env_options``);
    everything else gets a working default so test bodies stay focused.
    """

    def _factory(
        *,
        env_options: Optional[dict[str, Any]] = None,
        n_eval_episodes: int = 2,
    ) -> Sb3EvalScriptSettings:
        checkpoint = tmp_path / "policy.zip"
        checkpoint.write_bytes(b"x")
        return Sb3EvalScriptSettings(
            checkpoint=checkpoint,
            n_eval_episodes=n_eval_episodes,
            environment_settings=EnvironmentSettings(
                protocol_settings=GrpcProtocolConfig(url="localhost", port=1),
                env_options=env_options or {},
            ),
        )

    return _factory


def test_eval_main_forwards_env_options_to_env(patch_sb3_eval_deps, make_eval_args):
    """When ``environment_settings.env_options`` is non-empty, eval ``main`` should
    forward it to the env via ``set_options`` before ``evaluate_policy``."""
    opts = {"level": "1", "curriculum": "easy"}
    eval_main(make_eval_args(env_options=opts))

    patch_sb3_eval_deps.set_options.assert_called_once_with(options=opts)


def test_eval_main_skips_set_options_when_env_options_empty(
    patch_sb3_eval_deps, make_eval_args
):
    """When ``environment_settings.env_options`` is empty, eval ``main`` should not
    call ``set_options``."""
    eval_main(make_eval_args(env_options={}))

    patch_sb3_eval_deps.set_options.assert_not_called()
