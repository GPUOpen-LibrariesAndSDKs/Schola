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


# ---- eval.main forwarding tests --------------------------------------------
# Mock-based orchestration tests; real-object drift coverage lives in the
# test_apply_env_options_reaches_real_* tests.


@pytest.fixture
def patch_rllib_eval_deps(mocker):
    """Patch the RLlib + ray dependencies that ``eval.main`` reaches into so
    the test runs without actually loading a checkpoint or starting Ray.

    Returns the mock algorithm so per-test assertions can be made against
    its env-runner group's ``foreach_env_runner`` (which is what
    ``_apply_env_options`` ultimately drives)."""
    mocker.patch("ray.init")
    mocker.patch("ray.shutdown")

    runner = MagicMock()
    mock_algo = MagicMock()
    # foreach_env_runner(fn) applies fn to the single runner, mirroring RLlib.
    mock_algo.env_runner_group.foreach_env_runner.side_effect = lambda fn: fn(runner)
    # Set explicitly so the getattr(algo, "eval_env_runner_group", None) lookup
    # in _apply_env_options resolves to a real None, not an auto-created mock.
    mock_algo.eval_env_runner_group = None
    mock_algo.evaluate.return_value = {"env_runners": {"episode_reward_mean": 1.0}}
    mock_algo._captured_env = runner.env  # exposed for assertions

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


# ---- _apply_env_options real-object contract tests -------------------------
#
# These build a real algo (via the shared ``make_schola_rllib_config`` fixture,
# also used by ``test_rllib_env_runner``) and drive the actual
# ``env_runner_group`` / ``foreach_env_runner``, so a Ray rename of that
# contract -- or a dropped ``RayVecEnv.set_options`` -- fails here rather than
# passing green against fabricated mock attributes.


@pytest.fixture
def build_eval_algo(make_schola_rllib_config):
    """Build real algos from the shared config and ``stop()`` them at teardown,
    so tests don't have to manage cleanup themselves. Teardown runs even if the
    test body raises."""
    pytest.importorskip("ray")
    algos = []

    def _build(*, evaluation=None):
        algo = make_schola_rllib_config(evaluation=evaluation).build_algo()
        algos.append(algo)
        return algo

    yield _build

    for algo in algos:
        algo.stop()


@pytest.mark.xdist_group(name="ray-cluster")
@pytest.mark.timeout(180)
def test_apply_env_options_reaches_real_env_runner_group(build_eval_algo, ray_cluster):
    """Drives the actual ``env_runner_group`` / ``foreach_env_runner`` and lands
    ``set_options`` on a real ``RayVecEnv`` (asserted via its one-shot cache)."""
    algo = build_eval_algo()
    opts = {"level": "1", "curriculum": "easy"}
    _apply_env_options(algo, opts)

    cached = algo.env_runner_group.foreach_env_runner(lambda r: r.env._options)
    assert cached and all(c == opts for c in cached)


@pytest.mark.xdist_group(name="ray-cluster")
@pytest.mark.timeout(180)
def test_apply_env_options_reaches_real_eval_env_runner_group(
    build_eval_algo, ray_cluster
):
    """A separate ``eval_env_runner_group`` must also receive the options.

    It is ``evaluation_interval`` (not ``evaluation_num_env_runners``) that makes
    RLlib build the eval group, so we request a *local* eval env runner
    (``evaluation_num_env_runners=0``) which exercises the same
    ``_apply_env_options`` path. We deliberately avoid a remote eval runner: the
    driver has already loaded gRPC (fork-unsafe) and torch, so Ray spawning a
    remote env-runner actor aborts the process and crashes the xdist worker."""
    algo = build_eval_algo(
        evaluation={"evaluation_num_env_runners": 0, "evaluation_interval": 1}
    )
    opts = {"level": "1"}
    _apply_env_options(algo, opts)

    train_cached = algo.env_runner_group.foreach_env_runner(lambda r: r.env._options)
    eval_cached = algo.eval_env_runner_group.foreach_env_runner(
        lambda r: r.env._options
    )
    assert train_cached and all(c == opts for c in train_cached)
    assert eval_cached and all(c == opts for c in eval_cached)
