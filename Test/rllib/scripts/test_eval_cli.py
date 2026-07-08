# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for the RLlib eval CLI."""

import logging
from pathlib import Path

import pytest
from cyclopts import App

from schola.scripts.rllib.eval.eval import (
    RllibEvalCommand,
    _build_eval_config,
    _shape_env_runner_metrics,
    main as eval_main,
)
from schola.rllib.checkpoint import rl_module_dir_from_algorithm_checkpoint
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

    Yields ``(checkpoint_dir, eval_port)``. Train and eval run against separate
    gym servers on different ports, mirroring real post-hoc eval (e.g. train
    headless, eval with rendering); the train port is irrelevant to eval thanks
    to the CLI-wins env_config rebuild. Uses the session ``ray_cluster`` to avoid
    a double ``ray.init`` under ``ResourceSettings(using_cluster=True)``.
    """
    pytest.importorskip("ray")
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from ray.rllib.policy.policy import PolicySpec
    from schola.rllib.connectors import schola_env_to_module_flatten_connector
    from schola.rllib.env_runner import ScholaEnvRunner
    from schola.rllib.policy_mapping import (
        SCHOLA_POLICY_MAPPING_COMPONENT,
        ScholaPolicyMappingCheckpoint,
    )
    from schola.scripts.rllib.utils import build_env_config, discover_env_metadata
    from schola.scripts.common.settings import (
        EnvironmentSettings,
        GrpcProtocolConfig,
        PortOffsetMode,
    )

    train_port = make_vec_env_server([make_env("CartPole-v1", 0)])

    # Build the baked-in env_config through the same helper the train/eval CLIs
    # use, so the checkpoint matches a real run. ``fixed`` keeps the single
    # local runner on the training server port.
    env_config = build_env_config(
        EnvironmentSettings(
            protocol_settings=GrpcProtocolConfig(
                url="localhost", port=train_port, port_offset_mode=PortOffsetMode.FIXED
            ),
        )
    )

    policy_mapping_fn = lambda agent_id, *args, **kwargs: "shared_policy"
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
            env_to_module_connector=schola_env_to_module_flatten_connector,
        )
        .learners(num_learners=0)
        # Periodic RLlib eval during ``train()`` (not used by ``eval_main``).
        .evaluation(
            evaluation_num_env_runners=0,
            evaluation_interval=1,
            evaluation_duration=2,
            evaluation_duration_unit="episodes",
        )
        .multi_agent(
            policies={"shared_policy": PolicySpec()},
            policy_mapping_fn=policy_mapping_fn,
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
        train_env_settings = EnvironmentSettings(
            protocol_settings=GrpcProtocolConfig(
                url="localhost", port=train_port, port_offset_mode=PortOffsetMode.FIXED
            ),
        )
        agent_ids, _, _ = discover_env_metadata(train_env_settings)
        # Mirror the on-disk layout a ScholaAlgorithm checkpoint produces: the
        # frozen mapping lives in a ``schola_policy_mapping`` Checkpointable
        # subcomponent under the algorithm checkpoint dir.
        agent_to_policy = {agent_id: "shared_policy" for agent_id in agent_ids}
        ScholaPolicyMappingCheckpoint(agent_to_policy).save_to_path(
            ckpt / SCHOLA_POLICY_MAPPING_COMPONENT
        )
        # Separate gRPC port for post-hoc ``eval_main`` (CLI), not the train port.
        eval_port = make_vec_env_server([make_env("CartPole-v1", 0)])
        yield ckpt, eval_port
    finally:
        algo.stop()


# ---- eval_main on a real checkpoint ----------------------------------------
#
# ``dummy_rllib_checkpoint_dir`` trains a small run, then ``eval_main`` / CLI tests
# restore ``MultiRLModule``, build an eval ``EnvRunnerGroup`` from the CLI, and sample.


@pytest.mark.xdist_group(name="ray-cluster")
@pytest.mark.timeout(180)
def test_rllib_eval_main_on_real_checkpoint(dummy_rllib_checkpoint_dir):
    """``eval_main`` restores the checkpoint, rebuilds the env from the CLI
    protocol args (pointed at the dedicated eval server) and evaluates for real."""
    pytest.importorskip("ray")
    from schola.scripts.common.settings import EnvironmentSettings, GrpcProtocolConfig

    ckpt, eval_port = dummy_rllib_checkpoint_dir
    args = RllibEvalScriptSettings(
        checkpoint=ckpt,
        n_eval_episodes=2,
        environment_settings=EnvironmentSettings(
            protocol_settings=GrpcProtocolConfig(url="localhost", port=eval_port),
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
    checkpoint, with the protocol port pointed at the dedicated eval server."""
    pytest.importorskip("ray")
    ckpt, eval_port = dummy_rllib_checkpoint_dir
    results = rllib_eval_meta_app.meta(
        [
            "--checkpoint",
            str(ckpt),
            "--n-eval-episodes",
            "2",
            "--port",
            str(eval_port),
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


# ---- eval helper unit tests -------------------------------------------------


@pytest.fixture
def make_eval_args(tmp_path: Path):
    """Factory for ``RllibEvalScriptSettings`` over an existing (empty) checkpoint
    dir. ``num_simulators`` and ``env_options`` are the per-call knobs used by the
    settings-derived helper tests."""
    from schola.scripts.common.settings import (
        EnvironmentSettings,
        ExternalSimulatorConfig,
        GrpcProtocolConfig,
    )

    def _make(
        *, num_simulators: int = 1, env_options: dict | None = None
    ) -> RllibEvalScriptSettings:
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir(exist_ok=True)
        return RllibEvalScriptSettings(
            checkpoint=ckpt,
            n_eval_episodes=2,
            environment_settings=EnvironmentSettings(
                simulator_settings=ExternalSimulatorConfig(
                    num_simulators=num_simulators
                ),
                protocol_settings=GrpcProtocolConfig(url="localhost", port=1),
                env_options=env_options or {},
            ),
        )

    return _make


# ---- rl_module_dir_from_algorithm_checkpoint ---------------------------------


def test_rl_module_dir_prefers_new_api_stack_layout(tmp_path: Path):
    """The new-API-stack ``learner_group/learner/rl_module`` layout wins."""
    pytest.importorskip("ray")
    from ray.rllib.core import (
        COMPONENT_LEARNER,
        COMPONENT_LEARNER_GROUP,
        COMPONENT_RL_MODULE,
    )

    primary = (
        tmp_path / COMPONENT_LEARNER_GROUP / COMPONENT_LEARNER / COMPONENT_RL_MODULE
    )
    primary.mkdir(parents=True)
    # A legacy dir also present must not shadow the primary one.
    (tmp_path / "learner" / COMPONENT_RL_MODULE).mkdir(parents=True)

    assert rl_module_dir_from_algorithm_checkpoint(tmp_path) == primary


def test_rl_module_dir_falls_back_to_legacy_layout(tmp_path: Path):
    """When only the legacy ``learner/rl_module`` layout exists, it is returned."""
    pytest.importorskip("ray")
    from ray.rllib.core import COMPONENT_RL_MODULE

    legacy = tmp_path / "learner" / COMPONENT_RL_MODULE
    legacy.mkdir(parents=True)

    assert rl_module_dir_from_algorithm_checkpoint(tmp_path) == legacy


def test_rl_module_dir_raises_when_missing(tmp_path: Path):
    """A checkpoint with no RLModule dir raises a descriptive error."""
    pytest.importorskip("ray")
    with pytest.raises(FileNotFoundError, match="No RLModule checkpoint directory"):
        rl_module_dir_from_algorithm_checkpoint(tmp_path)


# ---- _shape_env_runner_metrics ---------------------------------------------
# Aggregates per-episode returns/lengths into the ``env_runners`` metrics shape
# applied in ``eval_main`` after episode sampling.


def test_shape_env_runner_metrics_aggregates_returns_and_lengths():
    """Means, episode count, and ``hist_stats`` are computed from the raw lists."""
    metrics = _shape_env_runner_metrics([5.0, 7.0, 9.0], [10, 20, 30])["env_runners"]

    assert metrics["num_episodes"] == 3.0
    assert metrics["episode_reward_mean"] == 7.0
    assert metrics["episode_len_mean"] == 20.0
    assert metrics["hist_stats"]["episode_reward"] == [5.0, 7.0, 9.0]
    assert metrics["hist_stats"]["episode_lengths"] == [10, 20, 30]


def test_shape_env_runner_metrics_raises_when_no_episodes():
    """No episodes must raise rather than return zeroed means."""
    with pytest.raises(RuntimeError, match="no episodes were collected"):
        _shape_env_runner_metrics([], [])


# ---- _build_eval_config ----------------------------------------------------


def test_build_eval_config_wires_schola_env_runner_and_spec():
    """``_build_eval_config`` sets Schola runner, connector, runner count, and module spec."""
    pytest.importorskip("ray")
    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from ray.rllib.policy.policy import PolicySpec
    from schola.rllib.connectors import schola_env_to_module_flatten_connector
    from schola.rllib.env_runner import ScholaEnvRunner

    spec = MultiRLModuleSpec(
        rl_module_specs={
            "shared_policy": RLModuleSpec(
                model_config=DefaultModelConfig(fcnet_hiddens=[8])
            )
        }
    )
    config = _build_eval_config(
        {"options": {}},
        num_env_runners=3,
        spec=spec,
        policies={"shared_policy": PolicySpec()},
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        rllib_log_level="WARN",
    )

    assert config.num_env_runners == 3
    assert config.env_runner_cls is ScholaEnvRunner
    assert config._env_to_module_connector is schola_env_to_module_flatten_connector
    # The restored module spec drives the eval architecture (RLlib stores a copy).
    assert set(config.rl_module_spec.rl_module_specs.keys()) == {"shared_policy"}
