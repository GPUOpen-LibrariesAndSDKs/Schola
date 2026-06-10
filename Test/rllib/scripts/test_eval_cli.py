# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for the RLlib eval CLI."""

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from cyclopts import App

from schola.scripts.rllib.eval.eval import (
    RllibEvalCommand,
    _apply_env_config,
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
def dummy_rllib_checkpoint_dir(
    tmp_path: Path, make_vec_env_server, make_env, ray_cluster
):
    """Train a tiny PPO over an in-process Schola gRPC gym server and save a
    checkpoint.

    Yields ``(checkpoint_dir, server_port)``. The gym server stays alive for the
    whole test so ``eval_main`` can reconnect to the same port after rebuilding
    the env (the env is opened once by ``from_checkpoint`` and again by
    ``_apply_env_config``). Uses the session ``ray_cluster`` so ``eval_main`` can
    run with ``ResourceSettings(using_cluster=True)`` without a double ``ray.init``.
    """
    pytest.importorskip("ray")
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.connectors.env_to_module import FlattenObservations
    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from ray.rllib.policy.policy import PolicySpec
    from schola.rllib.env_runner import ScholaEnvRunner
    from schola.scripts.common.settings import (
        EnvironmentSettings,
        GrpcProtocolConfig,
        PortOffsetMode,
    )

    port = make_vec_env_server([make_env("CartPole-v1", 0)])

    # Build the baked-in env_config through the same helper the train/eval CLIs
    # use, so the checkpoint matches a real run. ``fixed`` keeps the single
    # local runner on the in-process server port.
    env_config = EnvironmentSettings(
        protocol_settings=GrpcProtocolConfig(
            url="localhost", port=port, port_offset_mode=PortOffsetMode.FIXED
        ),
    ).make_env_config()

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .training(num_epochs=1, train_batch_size=128, minibatch_size=32)
        .environment(env_config=env_config)
        .framework("torch")
        .env_runners(
            env_runner_cls=ScholaEnvRunner,
            num_env_runners=0,
            env_to_module_connector=lambda env, spaces=None, device=None: FlattenObservations(
                input_observation_space=env.single_observation_space,
                input_action_space=env.single_action_space,
                multi_agent=True,
            ),
        )
        .learners(num_learners=0)
        # Bake in a local eval runner so the reloaded algo.evaluate() has an env
        # runner group to run on.
        .evaluation(
            evaluation_num_env_runners=0,
            evaluation_interval=1,
            evaluation_duration=2,
            evaluation_duration_unit="episodes",
        )
        .multi_agent(
            policies={"shared_policy": PolicySpec()},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "shared_policy": RLModuleSpec(
                        model_config=DefaultModelConfig(
                            fcnet_hiddens=[32, 32], vf_share_layers=True
                        )
                    )
                }
            ),
        )
    )
    algo = config.build_algo()
    try:
        algo.train()
        ckpt = tmp_path / "rllib_eval_ckpt"
        algo.save(str(ckpt))
        yield ckpt, port
    finally:
        algo.stop()


# ---- eval.main end-to-end tests on a real checkpoint -----------------------
#
# These drive the full ``eval_main`` orchestration on the checkpoint built by
# ``dummy_rllib_checkpoint_dir`` above: ``from_checkpoint`` -> ``_apply_env_config``
# (which rebuilds the env from the CLI config) -> ``algo.evaluate()``. Because the
# new eval always rebuilds the env, the CLI protocol must point back at the same
# live server port kept alive by the fixture.


@pytest.mark.xdist_group(name="ray-cluster")
@pytest.mark.timeout(180)
def test_rllib_eval_main_on_real_checkpoint(dummy_rllib_checkpoint_dir):
    """``eval_main`` restores the checkpoint, rebuilds the env from the CLI
    protocol args (pointed at the same live server) and evaluates for real."""
    pytest.importorskip("ray")
    from schola.scripts.common.settings import EnvironmentSettings, GrpcProtocolConfig

    ckpt, port = dummy_rllib_checkpoint_dir
    args = RllibEvalScriptSettings(
        checkpoint=ckpt,
        n_eval_episodes=2,
        environment_settings=EnvironmentSettings(
            protocol_settings=GrpcProtocolConfig(url="localhost", port=port),
        ),
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
    """End-to-end ``schola rllib eval`` parsing and ``eval_main`` on a real
    checkpoint, with the protocol port pointed at the live in-process server."""
    pytest.importorskip("ray")
    ckpt, port = dummy_rllib_checkpoint_dir
    results = rllib_eval_meta_app.meta(
        [
            "--checkpoint",
            str(ckpt),
            "--n-eval-episodes",
            "2",
            "--port",
            str(port),
            "--using-cluster",
        ],
        result_action="return_value",
    )
    assert isinstance(results, dict)
    env_metrics = results.get("env_runners") or results.get("evaluation")
    assert env_metrics is not None


# ---- CLI parsing tests -----------------------------------------------------


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
    mock_eval_app.meta(["--checkpoint", str(ckpt)], result_action="return_value")
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
    assert args.environment_settings.env_options == {"level": "1", "curriculum": "easy"}


# ---- eval.main orchestration tests -----------------------------------------


@pytest.fixture
def make_eval_args(tmp_path: Path):
    """Factory for ``RllibEvalScriptSettings`` with a localhost:1 protocol and an
    existing (empty) checkpoint dir; ``env_options`` is the per-call knob."""
    from schola.scripts.common.settings import EnvironmentSettings, GrpcProtocolConfig

    def _make(env_options: dict | None = None) -> RllibEvalScriptSettings:
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir(exist_ok=True)
        return RllibEvalScriptSettings(
            checkpoint=ckpt,
            n_eval_episodes=2,
            environment_settings=EnvironmentSettings(
                protocol_settings=GrpcProtocolConfig(url="localhost", port=1),
                env_options=env_options or {},
            ),
        )

    return _make


@pytest.fixture
def patch_rllib_eval_deps(mocker):
    """Patch the RLlib + ray dependencies ``eval.main`` reaches into so the test
    runs without loading a checkpoint or starting Ray.

    Exposes the single env runner plus its original/copied config so tests can
    assert that ``_apply_env_config`` unfroze the config, set ``env_config`` and
    rebuilt the env via ``make_env`` (what drives the real eval rebuild)."""
    from ray.rllib.algorithms.algorithm import Algorithm

    mocker.patch("ray.init")
    mocker.patch("ray.shutdown")

    runner = MagicMock()
    # Cache child mocks before _rebuild reassigns ``runner.config``.
    orig_cfg = runner.config
    new_cfg = orig_cfg.copy.return_value

    mock_algo = MagicMock(spec=Algorithm)
    mock_algo.config = MagicMock()

    mock_algo.env_runner_group = MagicMock()
    mock_algo.env_runner_group.foreach_env_runner.side_effect = lambda fn: [fn(runner)]
    # Explicit None so the ``algo.eval_env_runner_group`` lookup resolves to a
    # real None rather than an auto-created child mock.
    mock_algo.eval_env_runner_group = None
    mock_algo.evaluate.return_value = {"env_runners": {"episode_reward_mean": 1.0}}
    mock_algo._runner = runner
    mock_algo._orig_cfg = orig_cfg
    mock_algo._new_cfg = new_cfg

    mocker.patch(
        "ray.rllib.algorithms.algorithm.Algorithm.from_checkpoint",
        autospec=True,
        return_value=mock_algo,
    )
    return mock_algo


def test_eval_main_applies_cli_env_config(patch_rllib_eval_deps, make_eval_args):
    """``main`` unfreezes each runner's config, writes the CLI ``env_config``
    (including ``--env-options``) and rebuilds the env before evaluating."""
    opts = {"level": "1", "curriculum": "easy"}
    eval_main(make_eval_args(env_options=opts))

    algo = patch_rllib_eval_deps
    algo._orig_cfg.copy.assert_called_once_with(copy_frozen=False)
    algo._runner.make_env.assert_called_once()
    env_config = algo._new_cfg.environment.call_args.kwargs["env_config"]
    assert env_config["options"] == opts
    algo.evaluate.assert_called_once()


def test_eval_main_applies_env_config_even_when_options_empty(
    patch_rllib_eval_deps, make_eval_args
):
    """The CLI always wins: even with no ``--env-options`` the env is rebuilt
    from the CLI config (with empty options)."""
    eval_main(make_eval_args(env_options={}))

    algo = patch_rllib_eval_deps
    algo._runner.make_env.assert_called_once()
    env_config = algo._new_cfg.environment.call_args.kwargs["env_config"]
    assert env_config["options"] == {}


# ---- _apply_env_config real-object contract tests --------------------------
#
# These build a real algo (via the shared ``make_schola_rllib_config`` fixture,
# also used by ``test_rllib_env_runner``) and drive the actual
# ``env_runner_group`` / ``foreach_env_runner`` + ``make_env``, so a Ray rename
# of that contract -- or a broken ``copy(copy_frozen=...)`` -- fails here rather
# than passing green against fabricated mock attributes.


def _stub_env_config(protocol_cls, simulator_cls, url):
    return {
        "protocol": protocol_cls,
        "simulator": simulator_cls,
        "protocol_args": {"url": url},
        "simulator_args": {},
        "port_offset_mode": "per_worker",
        "options": {},
    }


@pytest.fixture
def build_eval_algo(make_schola_rllib_config):
    """Build real algos from the shared config and ``stop()`` them at teardown,
    so tests don't manage cleanup themselves. Teardown runs even on failure."""
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
def test_apply_env_options_reaches_real_env_runner_group(
    build_eval_algo, ray_cluster, stub_protocol_class, stub_simulator_class
):
    """Drives the actual ``env_runner_group`` / ``foreach_env_runner`` and
    rebuilds the env, asserting the new protocol ``url`` reached the real env on
    each runner."""
    algo = build_eval_algo()
    new_env_config = _stub_env_config(
        stub_protocol_class, stub_simulator_class, "thisisurl"
    )
    _apply_env_config(algo, new_env_config)

    urls = algo.env_runner_group.foreach_env_runner(
        lambda r: r.env.protocol.init_kwargs.get("url")
    )
    assert urls and all(u == "thisisurl" for u in urls)


@pytest.mark.xdist_group(name="ray-cluster")
@pytest.mark.timeout(180)
def test_apply_env_config_rebuilds_real_eval_env_runner_group(
    build_eval_algo, ray_cluster, stub_protocol_class, stub_simulator_class
):
    """A separate ``eval_env_runner_group`` must also be rebuilt.

    ``evaluation_interval`` (not ``evaluation_num_env_runners``) is what makes
    RLlib build the eval group, so we request a *local* eval env runner
    (``evaluation_num_env_runners=0``). We avoid a remote eval runner: the
    driver has already loaded gRPC (fork-unsafe) and torch, so Ray spawning a
    remote env-runner actor aborts the process and crashes the xdist worker."""
    algo = build_eval_algo(
        evaluation={"evaluation_num_env_runners": 0, "evaluation_interval": 1}
    )
    new_env_config = _stub_env_config(
        stub_protocol_class, stub_simulator_class, "sixseven"
    )
    _apply_env_config(algo, new_env_config)

    train_urls = algo.env_runner_group.foreach_env_runner(
        lambda r: r.env.protocol.init_kwargs.get("url")
    )
    eval_urls = algo.eval_env_runner_group.foreach_env_runner(
        lambda r: r.env.protocol.init_kwargs.get("url")
    )
    assert train_urls and all(u == "sixseven" for u in train_urls)
    assert eval_urls and all(u == "sixseven" for u in eval_urls)
