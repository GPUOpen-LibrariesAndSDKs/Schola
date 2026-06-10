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
    _build_env_config,
    main as eval_main,
)
from schola.scripts.rllib.eval.settings import RllibEvalScriptSettings


@pytest.fixture
def mock_main(mocker):
    return mocker.patch("schola.scripts.rllib.eval.eval.main")


@pytest.fixture
def mock_eval_app(mock_main):
    base = App(name="eval", help="Evaluate a trained RLlib policy from a checkpoint")
    logger = logging.getLogger(__name__)
    return RllibEvalCommand(base, RllibEvalScriptSettings, mock_main, logger).make()


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


# ---- _build_env_config unit tests ------------------------------------------


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


def test_build_env_config_defaults_to_external_simulator(make_eval_args):
    """Default simulator settings yield an ExternalSimulator gRPC config that
    carries the CLI protocol address and env options."""
    from schola.core.protocols.protobuf.grpc_protocol import GrpcProtocol
    from schola.core.simulators.external_simulator import ExternalSimulator

    cfg = _build_env_config(make_eval_args(env_options={"k": "v"}))

    assert cfg["protocol"] is GrpcProtocol
    assert cfg["simulator"] is ExternalSimulator
    assert cfg["protocol_args"]["url"] == "localhost"
    assert cfg["protocol_args"]["port"] == 1
    assert cfg["options"] == {"k": "v"}


def test_build_env_config_uses_executable_simulator(tmp_path):
    """An executable simulator config serializes via ``get_executable_args``
    (renamed kwargs + ``validate_path=False`` for remote reconstruction)."""
    from schola.core.simulators.unreal.executable_simulator import UnrealExecutable
    from schola.scripts.common.settings import (
        EnvironmentSettings,
        UnrealExecutableSimulatorConfig,
    )

    exe = tmp_path / "game.exe"
    exe.write_text("")
    args = RllibEvalScriptSettings(
        checkpoint=tmp_path,
        environment_settings=EnvironmentSettings(
            simulator_settings=UnrealExecutableSimulatorConfig(executable_path=exe),
        ),
    )

    cfg = _build_env_config(args)

    assert cfg["simulator"] is UnrealExecutable
    assert cfg["simulator_args"]["executable_path"] == exe
    assert cfg["simulator_args"]["validate_path"] is False


# ---- eval.main orchestration tests -----------------------------------------
# Mock-based orchestration tests; real-object drift coverage lives in the
# test_apply_env_config_rebuilds_real_* tests.


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
