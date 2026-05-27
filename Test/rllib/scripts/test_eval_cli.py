# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for the RLlib eval CLI."""

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from cyclopts import App

from schola.scripts.rllib.eval.eval import (
    RllibEvalCommand,
    _apply_env_options,
    main as eval_main,
)
from schola.scripts.rllib.eval.settings import RllibEvalScriptSettings
from schola.scripts.rllib.settings import ResourceSettings


@pytest.fixture
def mock_main(mocker):
    return mocker.patch("schola.scripts.rllib.eval.eval.main")


@pytest.fixture
def mock_eval_app(mock_main):
    base = App(name="eval", help="Evaluate a trained RLlib policy from a checkpoint")
    logger = logging.getLogger(__name__)
    return RllibEvalCommand(base, RllibEvalScriptSettings, mock_main, logger).make()


@pytest.fixture
def rllib_eval_meta_app():
    """Real ``schola rllib eval`` Cyclopts meta-app (invokes ``eval_main``)."""
    base = App(name="eval", help="Evaluate a trained RLlib policy from a checkpoint")
    logger = logging.getLogger(__name__)
    return RllibEvalCommand(base, RllibEvalScriptSettings, eval_main, logger).make()


@pytest.fixture
def dummy_rllib_checkpoint_dir(tmp_path: Path, ray_cluster):
    """
    Train a tiny PPO on ``CartPole-v1`` and save a checkpoint directory.

    Uses the session ``ray_cluster`` so ``eval_main`` can run with
    ``ResourceSettings(using_cluster=True)`` without double ``ray.init``.
    """
    pytest.importorskip("ray")
    from ray.rllib.algorithms.ppo import PPOConfig

    config = (
        PPOConfig()
        .environment("CartPole-v1")
        .env_runners(num_env_runners=0)
        .training(
            train_batch_size=200,
            minibatch_size=200,
            num_sgd_iter=1,
        )
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .learners(num_learners=0)
    )
    algo = config.build_algo()
    try:
        algo.train()
        ckpt = tmp_path / "rllib_eval_ckpt"
        algo.save(str(ckpt))
        return ckpt
    finally:
        algo.stop()


def test_eval_cli_forwards_checkpoint_and_defaults(
    mock_eval_app, mock_main, tmp_path: Path
):
    ckpt = tmp_path / "checkpoint_000001"
    ckpt.mkdir()
    mock_eval_app.meta(["--checkpoint", str(ckpt)], result_action="return_value")
    mock_main.assert_called_once()
    args = mock_main.call_args[0][0]
    assert isinstance(args, RllibEvalScriptSettings)
    assert args.checkpoint == ckpt
    assert args.n_eval_episodes == 10


def test_eval_cli_custom_episodes(mock_eval_app, mock_main, tmp_path: Path):
    ckpt = tmp_path / "c"
    ckpt.mkdir()
    mock_eval_app.meta(
        ["--checkpoint", str(ckpt), "--n-eval-episodes", "5"],
        result_action="return_value",
    )
    args = mock_main.call_args[0][0]
    assert args.n_eval_episodes == 5


def test_eval_cli_env_options_default_is_empty_dict(
    mock_eval_app, mock_main, tmp_path: Path
):
    """Without ``--env-options.k=v`` the field defaults to an empty dict."""
    ckpt = tmp_path / "c"
    ckpt.mkdir()
    mock_eval_app.meta(
        ["--checkpoint", str(ckpt)],
        result_action="return_value",
    )
    args = mock_main.call_args[0][0]
    assert args.environment_settings.env_options == {}


def test_eval_cli_env_options_dotted_syntax(mock_eval_app, mock_main, tmp_path: Path):
    """Cyclopts dotted syntax populates ``env_options`` with str values."""
    ckpt = tmp_path / "c"
    ckpt.mkdir()
    mock_eval_app.meta(
        [
            "--checkpoint",
            str(ckpt),
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


# ---- _apply_env_options unit tests -----------------------------------------


class FakeEnv:
    """Minimal stand-in for an env exposing ``set_options``. Records every
    call so tests can assert against ``set_options_calls`` directly."""

    def __init__(self):
        self.set_options_calls = []

    def set_options(self, opts):
        self.set_options_calls.append(opts)


class FakeEnvRunner:
    """Env runner with a real ``FakeEnv`` attached as ``.env``."""

    def __init__(self):
        self.env = FakeEnv()


class FakeEnvRunnerGroup:
    """Implements RLlib's ``foreach_env_runner(fn)`` semantics over an
    in-memory list of runners. ``foreach_calls`` lets tests assert
    whether the group was traversed at all."""

    def __init__(self, runners=()):
        self._runners = list(runners)
        self.foreach_calls = 0

    def foreach_env_runner(self, fn):
        self.foreach_calls += 1
        for r in self._runners:
            fn(r)


class FakeAlgo:
    """Tiny algorithm stand-in exposing just the two attributes that
    ``_apply_env_options`` reads."""

    def __init__(self, env_runner_group, eval_env_runner_group=None):
        self.env_runner_group = env_runner_group
        self.eval_env_runner_group = eval_env_runner_group


def test_apply_env_options_noops_on_empty_dict():
    """An empty ``env_options`` must not touch the algorithm at all -- no
    env-runner traversal, no ``set_options`` calls."""
    runner = FakeEnvRunner()
    group = FakeEnvRunnerGroup([runner])
    algo = FakeAlgo(group)

    _apply_env_options(algo, {})

    assert group.foreach_calls == 0
    assert runner.env.set_options_calls == []


def test_apply_env_options_stages_via_foreach_env_runner():
    """``_apply_env_options`` must dispatch through ``foreach_env_runner`` on
    the training group; the staged callable then resolves
    ``runner.env.set_options(opts)`` on each runner."""
    runner = FakeEnvRunner()
    algo = FakeAlgo(FakeEnvRunnerGroup([runner]))

    opts = {"level": "1"}
    _apply_env_options(algo, opts)

    assert runner.env.set_options_calls == [opts]


def test_apply_env_options_also_stages_on_eval_env_runner_group():
    """When the checkpoint was trained with ``evaluation_num_env_runners > 0``
    the algorithm exposes a separate ``eval_env_runner_group`` whose envs
    are the ones ``algo.evaluate()`` actually drives; both groups must be
    visited."""
    train_runner = FakeEnvRunner()
    eval_runner = FakeEnvRunner()
    algo = FakeAlgo(
        env_runner_group=FakeEnvRunnerGroup([train_runner]),
        eval_env_runner_group=FakeEnvRunnerGroup([eval_runner]),
    )

    opts = {"level": "1"}
    _apply_env_options(algo, opts)

    assert train_runner.env.set_options_calls == [opts]
    assert eval_runner.env.set_options_calls == [opts]


# ---- eval.main forwarding tests --------------------------------------------


@pytest.fixture
def patch_rllib_eval_deps(mocker):
    """Patch the RLlib + ray dependencies that ``eval.main`` reaches into so
    the test runs without actually loading a checkpoint or starting Ray.

    Returns the mock algorithm so per-test assertions can be made against
    its env-runner group's ``foreach_env_runner`` (which is what
    ``_apply_env_options`` ultimately drives)."""
    mocker.patch("ray.init")
    mocker.patch("ray.shutdown")

    mock_algo = MagicMock()
    mock_algo.evaluate.return_value = {"env_runners": {"episode_reward_mean": 1.0}}

    runner = MagicMock()
    mock_algo._captured_env = runner.env  # exposed for assertions
    mock_algo.env_runner_group.foreach_env_runner.side_effect = lambda fn: fn(runner)
    mock_algo.eval_env_runner_group = None

    mocker.patch(
        "ray.rllib.algorithms.algorithm.Algorithm.from_checkpoint",
        return_value=mock_algo,
    )
    return mock_algo


def _make_eval_args(
    tmp_path: Path, env_options: dict | None = None
) -> RllibEvalScriptSettings:
    from schola.scripts.common.settings import EnvironmentSettings, GrpcProtocolConfig

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    return RllibEvalScriptSettings(
        checkpoint=ckpt,
        n_eval_episodes=2,
        environment_settings=EnvironmentSettings(
            protocol_settings=GrpcProtocolConfig(url="localhost", port=1),
            env_options=env_options or {},
        ),
    )


def test_eval_main_forwards_env_options_to_env(patch_rllib_eval_deps, tmp_path):
    """When ``env_options`` is non-empty, ``main`` should stage it on every
    env runner via ``set_options`` before invoking ``algo.evaluate()``."""
    opts = {"level": "1", "curriculum": "easy"}
    eval_main(_make_eval_args(tmp_path, env_options=opts))

    patch_rllib_eval_deps._captured_env.set_options.assert_called_once_with(opts)
    patch_rllib_eval_deps.evaluate.assert_called_once()


def test_eval_main_skips_set_options_when_env_options_empty(
    patch_rllib_eval_deps, tmp_path
):
    """When ``env_options`` is empty, ``main`` should not call
    ``foreach_env_runner`` or ``set_options`` at all."""
    eval_main(_make_eval_args(tmp_path, env_options={}))

    patch_rllib_eval_deps.env_runner_group.foreach_env_runner.assert_not_called()


@pytest.mark.xdist_group(name="ray-cluster")
@pytest.mark.timeout(180)
def test_rllib_eval_main_on_real_checkpoint(dummy_rllib_checkpoint_dir):
    pytest.importorskip("ray")
    args = RllibEvalScriptSettings(
        checkpoint=dummy_rllib_checkpoint_dir,
        n_eval_episodes=2,
        resource_settings=ResourceSettings(using_cluster=True),
    )
    results = eval_main(args)
    assert isinstance(results, dict)
    env_metrics = results.get("env_runners") or results.get("evaluation")
    assert env_metrics is not None


@pytest.mark.xdist_group(name="ray-cluster")
@pytest.mark.timeout(180)
def test_rllib_eval_cli_on_real_checkpoint(
    dummy_rllib_checkpoint_dir, rllib_eval_meta_app
):
    """End-to-end ``schola rllib eval`` parsing and ``eval_main`` on a real checkpoint."""
    pytest.importorskip("ray")
    results = rllib_eval_meta_app.meta(
        [
            "--checkpoint",
            str(dummy_rllib_checkpoint_dir),
            "--n-eval-episodes",
            "2",
            "--using-cluster",
        ],
        result_action="return_value",
    )
    assert isinstance(results, dict)
    env_metrics = results.get("env_runners") or results.get("evaluation")
    assert env_metrics is not None
