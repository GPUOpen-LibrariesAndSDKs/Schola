# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Tests for ``ScholaEnvRunner.make_env`` forwarding of ``env_config``.

The env runner is the bridge between the training driver's ``env_config``
(populated from CLI ``--env-options.k=v``) and ``RayVecEnv``. This module
pins the only contract ``make_env`` has around env_options: whatever the
driver puts into ``env_ctx`` (with or without an ``options`` key) is what
``RayVecEnv`` receives as ``env_config``, unchanged.

``MultiAgentEnvRunner.__init__`` requires a real RLlib algorithm config, so
we bypass it with ``__new__`` + attribute injection and patch out
``RayVecEnv`` to inspect the kwargs it would have been constructed with.
"""

from unittest.mock import MagicMock, patch

import pytest
from ray.rllib.env.env_context import EnvContext

from schola.core.protocols.base_protocol import BaseRLProtocol
from schola.core.simulators.base_simulator import BaseSimulator
from schola.rllib.env_runner import ScholaEnvRunner


class _FakeProtocol(BaseRLProtocol):
    """Empty subclass that satisfies the ``issubclass(..., BaseRLProtocol)``
    assertion in ``make_env``. ``BaseRLProtocol`` has no abstract methods
    and no ``__init__``, so an empty body is enough."""


class _FakeSimulator(BaseSimulator):
    """Counterpart of ``_FakeProtocol`` for the ``BaseSimulator`` check."""


def _build_runner(env_ctx: dict) -> ScholaEnvRunner:
    """Build a ``ScholaEnvRunner`` with the minimum attributes ``make_env``
    reads, sidestepping ``MultiAgentEnvRunner.__init__``."""
    runner = ScholaEnvRunner.__new__(ScholaEnvRunner)
    runner.env = None
    runner.worker_index = 0
    runner._callbacks = []
    runner.metrics = MagicMock()
    runner.config = MagicMock(
        env_config=EnvContext(
            env_ctx, worker_index=0, num_workers=0, remote=False
        ),
        num_envs_per_env_runner=1,
        disable_env_checking=True,
        callbacks_on_environment_created=[],
    )
    return runner


@pytest.mark.parametrize(
    "options",
    [
        {"level": "1", "curriculum": "easy"},
        None,
    ],
    ids=["with_options", "without_options"],
)
def test_make_env_round_trips_env_ctx_into_env_config(options):
    """``env_config`` handed to ``RayVecEnv`` must equal the input
    ``env_ctx`` key-for-key.

    This single assertion subsumes both "options forwarded when present"
    and "no options key invented when absent", and additionally catches
    any accidental added/dropped/mutated keys (which the old per-key
    assertions would have missed).
    """
    env_ctx = {
        "protocol": _FakeProtocol,
        "protocol_args": {},
        "simulator": _FakeSimulator,
        "simulator_args": {},
    }
    if options is not None:
        env_ctx["options"] = options

    runner = _build_runner(env_ctx)

    with patch("schola.rllib.env_runner.RayVecEnv") as mock_vec:
        mock_vec.return_value = MagicMock(num_envs=1)
        runner.make_env()

    mock_vec.assert_called_once()
    forwarded = dict(mock_vec.call_args.kwargs["env_config"])
    assert forwarded == env_ctx
