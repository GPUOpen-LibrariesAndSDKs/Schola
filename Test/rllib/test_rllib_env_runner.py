# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Tests for ``ScholaEnvRunner.make_env``.

The runner is built through its real ``__init__`` (see ``_build_runner``)
rather than a bypass, so ``make_env`` runs the same way it does under RLlib's
``EnvRunnerGroup``.
"""

from schola.rllib.env import RayVecEnv
from schola.rllib.env_runner import ScholaEnvRunner


def _build_runner(
    make_schola_rllib_config,
    *,
    protocol_args=None,
    simulator_args=None,
    port_offset_mode: str = "per_worker",
    worker_index: int = 0,
) -> ScholaEnvRunner:
    """Construct a ``ScholaEnvRunner`` the same way RLlib's ``EnvRunnerGroup``
    would: build a fully-flagged ``AlgorithmConfig`` (via the shared
    ``make_schola_rllib_config`` fixture) and call the runner's real ``__init__``.

    ``num_env_runners=0`` (set in the shared config) puts env creation on the
    local worker, which makes ``MultiAgentEnvRunner.__init__`` invoke
    ``self.make_env()`` -- the behavior under test.
    """
    config = make_schola_rllib_config(
        protocol_args=protocol_args,
        simulator_args=simulator_args,
        port_offset_mode=port_offset_mode,
    )
    return ScholaEnvRunner(config=config, worker_index=worker_index)


def test_env_runner_forwards_protocol_and_simulator_args(
    make_schola_rllib_config, stub_protocol_class, stub_simulator_class
):
    """``make_env`` instantiates ``protocol(**protocol_args)`` and
    ``simulator(**simulator_args)`` from the ``env_config``, after applying
    ``resolve_protocol_args`` (which expands ``{worker_index}`` and adds
    ``worker_index`` to ``port`` under ``port_offset_mode='per_worker'``)."""
    runner = _build_runner(
        make_schola_rllib_config,
        protocol_args={"url": "ue-{worker_index}", "port": 50051},
        simulator_args={"foo": "bar"},
        worker_index=2,
    )

    [protocol_inst] = stub_protocol_class.instances
    [simulator_inst] = stub_simulator_class.instances

    assert protocol_inst.init_kwargs == {"url": "ue-2", "port": 50053}
    assert simulator_inst.init_kwargs == {"foo": "bar"}
    assert isinstance(runner.env, RayVecEnv)
    assert runner.num_envs == stub_protocol_class.NUM_ENVS
