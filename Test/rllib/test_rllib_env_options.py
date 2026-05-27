# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Tests for env_options / ``env_config["options"]`` on the RLlib env.

The behavior under test, defined on ``BaseRayEnv`` and overridden on
``RayVecEnv.reset``:

* ``env_config["options"]`` seeds a one-shot cache on construction.
* ``set_options(opts)`` overwrites the cache; ``set_options(None)`` clears it.
* The first ``reset()`` without an explicit ``options=`` broadcasts the cache
  to every sub-env and clears it (SB3-style one-shot).
* An explicit ``reset(options=dict)`` is broadcast for that reset only and
  does not consume the cache.
* An explicit ``reset(options=list)`` is forwarded element-wise and its
  length must match ``num_envs``.
* All entry points deepcopy so caller-side mutation cannot leak in.

Only ``RayVecEnv`` is covered: the user-facing ``--env-options`` flag flows
through ``ScholaEnvRunner.make_env``/``Algorithm.from_checkpoint``, both of
which build ``RayVecEnv`` (never ``RayEnv``). The cache itself lives on
``BaseRayEnv``, so single-env coverage would only re-prove the shared layer.
"""

import gymnasium as gym
import numpy as np
import pytest

from schola.rllib.env import RayVecEnv


class _StubRayVecEnv(RayVecEnv):
    """Stub out ``_define_environment`` so ``__init__`` can run without a
    live Unreal protocol.

    Only sets the four attributes ``RayVecEnv._init_agent_tracking`` actually
    reads when building its ``_SingleEnvWrapper`` list (``num_envs``,
    ``possible_agents``, ``_single_observation_spaces``,
    ``_single_action_spaces``). Everything else -- including the one-shot
    ``_options`` cache -- is set by the real ``BaseRayEnv.__init__``, which
    is the whole point of running construction for real.
    """

    def __init__(self, protocol, simulator, num_envs=2, *, env_config=None):
        self._stub_num_envs = num_envs
        super().__init__(protocol, simulator, env_config=env_config)

    def _define_environment(self):
        self.num_envs = self._stub_num_envs
        self.possible_agents = ["agent_0"]
        self._single_observation_spaces = {
            "agent_0": gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32)
        }
        self._single_action_spaces = {"agent_0": gym.spaces.Discrete(2)}


@pytest.fixture
def make_env(mock_protocol_and_simulator):
    """Build a real ``RayVecEnv`` against mocked protocol/simulator.

    ``protocol.send_reset_msg`` is pre-armed with an ``(obs, infos)`` tuple
    shaped to ``num_envs`` so the post-reset wrapper-state update path in
    ``RayVecEnv.reset`` does not ``IndexError``.
    """
    protocol, simulator = mock_protocol_and_simulator

    def _factory(num_envs=2, *, env_config=None):
        obs = [{"agent_0": np.zeros((1,), dtype=np.float32)} for _ in range(num_envs)]
        infos = [{"agent_0": {}} for _ in range(num_envs)]
        protocol.send_reset_msg.return_value = (obs, infos)
        return _StubRayVecEnv(protocol, simulator, num_envs, env_config=env_config)

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
    env = make_env(num_envs=2, env_config={"options": opts})

    env.reset()

    env.protocol.send_reset_msg.assert_called_once_with(
        seeds=None, options=[opts, opts]
    )
    assert env._options == {}


def test_reset_without_cached_options_forwards_none(make_env):
    """No cache and no explicit ``options=`` → protocol gets ``options=None``,
    preserving the pre-feature behavior for unconfigured envs."""
    env = make_env(num_envs=2)
    env.reset()
    env.protocol.send_reset_msg.assert_called_once_with(seeds=None, options=None)


def test_explicit_dict_does_not_consume_cache(make_env):
    """An explicit ``reset(options=dict)`` is broadcast for that reset only
    and leaves the cached value armed for a later ``reset()``."""
    cached = {"level": "cached"}
    env = make_env(num_envs=2, env_config={"options": cached})

    env.reset(options={"level": "override"})

    env.protocol.send_reset_msg.assert_called_once_with(
        seeds=None, options=[{"level": "override"}, {"level": "override"}]
    )
    assert env._options == cached


def test_explicit_list_options_per_env(make_env):
    """A list-of-dicts of length ``num_envs`` is forwarded element-wise; a
    list whose length doesn't match must raise (documented contract)."""
    env = make_env(num_envs=2)

    per_env = [{"level": "1"}, {"level": "2"}]
    env.reset(options=per_env)
    env.protocol.send_reset_msg.assert_called_once_with(seeds=None, options=per_env)

    with pytest.raises(AssertionError):
        env.reset(options=[{"level": "1"}])
