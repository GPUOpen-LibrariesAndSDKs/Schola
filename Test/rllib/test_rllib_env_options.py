# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Tests for env_options / ``env_config["options"]`` on the RLlib env.

Behavior under test (on ``BaseRayEnv``, with ``RayVecEnv.reset`` override):

* ``env_config["options"]`` seeds a one-shot cache on construction.
* ``set_options(opts)`` overwrites the cache; ``set_options(None)`` clears it.
* The first ``reset()`` without explicit ``options=`` broadcasts the cache to
  every sub-env and clears it; explicit ``reset(options=dict)`` broadcasts for
  that call only; ``reset(options=list)`` is forwarded element-wise and must
  match ``num_envs``.
* All entry points deepcopy so caller-side mutation cannot leak in.

Only ``RayVecEnv`` is covered since the ``--env-options`` flag always builds
it (never ``RayEnv``). It is constructed directly against the
``stub_protocol_class`` / ``stub_simulator_class`` fixtures, which pre-arm the
wire-level shapes so the real ``__init__`` / ``reset`` run end-to-end -- no
subclassing needed to bypass ``_define_environment`` (see ``make_env``).
"""

import pytest

from schola.rllib.env import RayVecEnv


@pytest.fixture
def make_env(stub_protocol_class, stub_simulator_class):
    """Build a real ``RayVecEnv`` against stub protocol/simulator instances.

    The stubs are framework-agnostic real subclasses of ``BaseRLProtocol`` /
    ``BaseSimulator`` (see ``Test/conftest.py``) whose ``get_definition`` /
    ``send_reset_msg`` are pre-armed for ``stub_protocol_class.NUM_ENVS``
    sub-envs, which is what lets ``RayVecEnv.__init__`` and ``.reset()`` run
    against them without a live gRPC server or hand-crafted post-reset state.
    """

    def _factory(*, env_config=None):
        protocol = stub_protocol_class()
        simulator = stub_simulator_class()
        return RayVecEnv(protocol, simulator, env_config=env_config)

    return _factory


# ---- Cache state (no reset; no protocol mock interaction) -------------------


def test_env_config_options_seeds_cache(make_env):
    """``env_config["options"]`` populates ``_options`` on construction."""
    opts = {"level": "67", "curriculum": "sorta_hard"}
    env = make_env(env_config={"options": opts})
    assert env._options == opts


def test_set_options_overwrites_cache(make_env):
    """``set_options(opts)`` replaces whatever was in the cache."""
    env = make_env(env_config={"options": {"level": "old"}})
    env.set_options({"level": "new"})
    assert env._options == {"level": "new"}


def test_set_options_none_clears_cache(make_env):
    """``set_options(None)`` clears any pending options."""
    env = make_env(env_config={"options": {"level": "67"}})
    env.set_options(None)
    assert env._options == {}


def test_cache_is_deepcopied_from_env_config(make_env):
    """Mutating the source dict after construction must not leak into the
    cache (``BaseRayEnv.__init__`` deepcopies on capture)."""
    src = {"level": "67"}
    env = make_env(env_config={"options": src})
    src["level"] = "MUTATED"
    assert env._options == {"level": "67"}


# ---- Reset consumption ------------------------------------------------------


def test_first_reset_broadcasts_cache_and_clears_it(make_env):
    """A cached options dict is broadcast to every sub-env (as a per-env
    list) and the cache is cleared in the same call -- the one-shot pattern."""
    opts = {"level": "67"}
    env = make_env(env_config={"options": opts})

    env.reset()

    env.protocol.send_reset_msg.assert_called_once_with(
        seeds=None, options=[opts] * env.num_envs
    )
    assert env._options == {}


def test_reset_without_cached_options_forwards_none(make_env):
    """No cache and no explicit ``options=`` → protocol gets ``options=None``,
    preserving the pre-feature behavior for unconfigured envs."""
    env = make_env()
    env.reset()
    env.protocol.send_reset_msg.assert_called_once_with(seeds=None, options=None)


def test_explicit_dict_does_not_consume_cache(make_env):
    """An explicit ``reset(options=dict)`` is broadcast for that reset only
    and leaves the cached value armed for a later ``reset()``."""
    cached = {"level": "cached"}
    env = make_env(env_config={"options": cached})

    env.reset(options={"level": "override"})

    env.protocol.send_reset_msg.assert_called_once_with(
        seeds=None, options=[{"level": "override"}] * env.num_envs
    )
    assert env._options == cached


def test_explicit_list_options_per_env(make_env):
    """A list-of-dicts of length ``num_envs`` is forwarded element-wise; a
    list whose length doesn't match must raise (documented contract)."""
    env = make_env()

    per_env = [{"level": str(i)} for i in range(env.num_envs)]
    env.reset(options=per_env)
    env.protocol.send_reset_msg.assert_called_once_with(seeds=None, options=per_env)

    with pytest.raises(AssertionError):
        env.reset(options=per_env[:-1])
