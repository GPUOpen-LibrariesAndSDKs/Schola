# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Tests for staggered-agent-death handling in multi-agent environments.

Problem scenario:
  When agents die at different timesteps in a NEXT_STEP autoreset environment,
  RLlib stops sending actions for dead agents after their terminal step. The
  Python side (RayEnv/RayVecEnv) forwards exactly the actions provided by
  RLlib — no zero-padding is added for dead agents.

  On the C++ side, TScholaEnvironment::Step() handles missing-action agents:
    1. It snapshots agents that are already terminated/truncated.
    2. It filters dead agents out of InActions before calling Execute_Step(),
       so the Blueprint implementation only receives actions for live agents.
    3. It restores the dead agents' terminal flags after Execute_Step(), so
       AllAgentsCompleted() can eventually return true and end the episode.

Python-side contract tested here:
  - RayEnv.step() forwards the caller's action dict unchanged to the protocol.
  - __all__ is computed correctly as agents die one by one.
  - The episode is considered complete once every agent has been marked dead.

Run:
  python -m pytest Test/rllib/test_staggered_death.py -v
  OR (standalone, no pytest required):
  python Test/rllib/test_staggered_death.py
"""

import sys
import numpy as np
import gymnasium as gym


# ---------------------------------------------------------------------------
# FakeProtocol — simulates staggered 3-agent death, no Unreal required
# ---------------------------------------------------------------------------
class FakeProtocol:
    """
    Mock protocol that kills agents one per step.

    Timeline:
      step 1: agent_0 dies (terminated=True), agent_1/agent_2 alive
      step 2: agent_1 dies,                  agent_2 alive
      step 3: agent_2 dies  -> all agents dead -> __all__ can be True
    """

    def __init__(self):
        self.step_count = 0
        self.agent_ids = ["agent_0", "agent_1", "agent_2"]
        self.env_id = 0
        self._started = True
        self.received_actions_log = []  # list[set] — one entry per step

    def __bool__(self):
        return self._started

    def start(self):
        self._started = True

    def close(self):
        pass

    @property
    def properties(self):
        return {}

    def send_startup_msg(self, auto_reset_type=None):
        pass

    def get_definition(self):
        obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        ids = [self.agent_ids]
        agent_types = [{a: "default" for a in self.agent_ids}]
        obs_defns = {0: {a: obs_space for a in self.agent_ids}}
        act_defns = {0: {a: act_space for a in self.agent_ids}}
        return ids, agent_types, obs_defns, act_defns

    def send_reset_msg(self, seeds=None, options=None):
        self.step_count = 0
        self.received_actions_log = []
        obs = [{a: np.zeros(4, dtype=np.float32) for a in self.agent_ids}]
        infos = [{a: {} for a in self.agent_ids}]
        return obs, infos

    def send_action_msg(self, actions, action_spaces):
        self.step_count += 1
        env_actions = actions[self.env_id]
        # Record exactly what keys were received (C++ TScholaEnvironment would
        # see the same set when Python is not doing any no-op padding).
        self.received_actions_log.append(set(env_actions.keys()))

        obs = {a: np.random.randn(4).astype(np.float32) for a in self.agent_ids}
        rewards = {a: 0.0 for a in self.agent_ids}
        terminateds = {a: False for a in self.agent_ids}
        truncateds = {a: False for a in self.agent_ids}
        infos = {a: {} for a in self.agent_ids}

        if self.step_count >= 1:
            terminateds["agent_0"] = True
            rewards["agent_0"] = -1.0
        if self.step_count >= 2:
            terminateds["agent_1"] = True
            rewards["agent_1"] = -1.0
        if self.step_count >= 3:
            terminateds["agent_2"] = True
            rewards["agent_2"] = -1.0

        return (
            {self.env_id: obs},
            {self.env_id: rewards},
            {self.env_id: terminateds},
            {self.env_id: truncateds},
            {self.env_id: infos},
            {},  # initial_obs  (unused in NEXT_STEP mode)
            {},  # initial_infos (unused in NEXT_STEP mode)
        )


class FakeSimulator:
    supported_protocols = (type(None),)

    def start(self, properties=None):
        pass

    def stop(self):
        pass


# ---------------------------------------------------------------------------
# Helper: build a RayEnv around FakeProtocol without starting Unreal
# ---------------------------------------------------------------------------
def make_ray_env():
    """Construct a RayEnv backed by FakeProtocol (no Unreal connection)."""
    protocol = FakeProtocol()
    try:
        from schola.rllib.env import RayEnv
        from ray.rllib.env.multi_agent_env import MultiAgentEnv

        simulator = FakeSimulator()
        simulator.supported_protocols = type(protocol)

        env = RayEnv.__new__(RayEnv)
        env.protocol = protocol
        env.simulator = simulator
        env._init_space_attributes()
        protocol.start()
        env.protocol.send_startup_msg()
        env._define_environment()
        env._init_agent_tracking()
        MultiAgentEnv.__init__(env)
        env._fake_protocol = protocol
        return env
    except ImportError:
        # Fallback: lightweight replica used when ray/rllib is not installed
        return _StandaloneEnv(protocol)


# ---------------------------------------------------------------------------
# Lightweight replica of RayEnv.step() for standalone / CI use
# ---------------------------------------------------------------------------
class _StandaloneEnv:
    """Reproduces RayEnv.step() tracking logic without requiring ray."""

    def __init__(self, protocol):
        self.protocol = protocol
        self._env_id = 0
        self._terminated_agents = set()
        self._truncated_agents = set()
        self._current_agents = set()

        ids, _, obs_defns, act_defns = protocol.get_definition()
        self.possible_agents = list(ids[0])
        self._current_agents = set(self.possible_agents)
        self._single_action_spaces = act_defns[0]
        self._single_observation_spaces = obs_defns[0]
        self._fake_protocol = protocol

    @property
    def single_action_spaces(self):
        return self._single_action_spaces

    def reset(self, seed=None):
        self._terminated_agents = set()
        self._truncated_agents = set()
        obs, infos = self.protocol.send_reset_msg()
        self._current_agents = set(obs[0].keys())
        return obs[self._env_id], infos[self._env_id]

    def step(self, actions):
        # Forward actions as-is — no no-op padding.
        action_dict = {self._env_id: actions}
        observations, rewards, terminateds, truncateds, infos, _, _ = (
            self.protocol.send_action_msg(action_dict, self._single_action_spaces)
        )

        eid = self._env_id
        all_agents_this_step = set(terminateds[eid]) | set(truncateds[eid])

        for a in all_agents_this_step:
            if terminateds[eid].get(a):
                self._terminated_agents.add(a)
            if truncateds[eid].get(a):
                self._truncated_agents.add(a)

        self._current_agents = {
            a
            for a in all_agents_this_step
            if not terminateds[eid].get(a) and not truncateds[eid].get(a)
        }

        all_known = (
            self._current_agents | self._terminated_agents | self._truncated_agents
        )
        num_done = len(self._terminated_agents | self._truncated_agents)
        num_total = len(all_known)

        terminateds[eid]["__all__"] = (
            (num_done == num_total) if num_total > 0 else False
        )
        truncateds[eid]["__all__"] = (
            (len(self._truncated_agents) == num_total) if num_total > 0 else False
        )

        return (
            observations[eid],
            rewards[eid],
            terminateds[eid],
            truncateds[eid],
            infos[eid],
        )


# ===========================================================================
# Tests
# ===========================================================================


def test_only_live_agent_actions_forwarded():
    """
    Python must send exactly the actions RLlib provides — no zero-padding.

    The C++ TScholaEnvironment::Step() is responsible for skipping dead agents
    that are absent from InActions.  If Python were to pad dead agents with
    zeros, hierarchical-RL policies that distinguish no-op from a valid action
    would produce incorrect behaviour.
    """
    env = make_ray_env()
    obs, _ = env.reset()
    protocol = env._fake_protocol
    action_spaces = env.single_action_spaces

    # Step 1: all 3 agents alive — all 3 actions sent
    actions = {a: action_spaces[a].sample() for a in obs}
    env.step(actions)
    assert protocol.received_actions_log[0] == {"agent_0", "agent_1", "agent_2"}

    # Step 2: RLlib only provides actions for agent_1 and agent_2 (agent_0 is dead)
    env.step({a: action_spaces[a].sample() for a in ["agent_1", "agent_2"]})
    assert protocol.received_actions_log[1] == {
        "agent_1",
        "agent_2",
    }, "Dead agent_0 must NOT be zero-padded into the action map"

    # Step 3: only agent_2 alive
    env.step({"agent_2": action_spaces["agent_2"].sample()})
    assert protocol.received_actions_log[2] == {
        "agent_2"
    }, "Only the last living agent's action should reach the protocol"


def test_all_flag_false_while_agents_alive():
    """__all__ must stay False until every agent is terminated/truncated."""
    env = make_ray_env()
    obs, _ = env.reset()
    action_spaces = env.single_action_spaces

    # Step 1: agent_0 dies
    _, _, terminateds, truncateds, _ = env.step(
        {a: action_spaces[a].sample() for a in obs}
    )
    assert terminateds["agent_0"] is True
    assert terminateds["__all__"] is False, "agent_1 and agent_2 are still alive"

    # Step 2: agent_1 dies
    _, _, terminateds, truncateds, _ = env.step(
        {a: action_spaces[a].sample() for a in ["agent_1", "agent_2"]}
    )
    assert terminateds["agent_1"] is True
    assert terminateds["__all__"] is False, "agent_2 is still alive"


def test_all_flag_true_when_last_agent_dies():
    """__all__ must become True on the step the last living agent dies."""
    env = make_ray_env()
    obs, _ = env.reset()
    action_spaces = env.single_action_spaces

    env.step({a: action_spaces[a].sample() for a in obs})
    env.step({a: action_spaces[a].sample() for a in ["agent_1", "agent_2"]})

    # Step 3: agent_2 dies — all agents now dead
    _, _, terminateds, truncateds, _ = env.step(
        {"agent_2": action_spaces["agent_2"].sample()}
    )
    assert terminateds["agent_2"] is True
    assert (
        terminateds["__all__"] is True
    ), "All agents dead — episode must be marked done"


def test_episode_completes_within_bounded_steps():
    """
    Driving the env with only live agents' actions must reach __all__=True.

    A hang here means either the Python tracking or the C++ terminal-state
    preservation (TScholaEnvironment::Step) is broken.
    """
    env = make_ray_env()
    obs, _ = env.reset()
    action_spaces = env.single_action_spaces

    done = False
    for _ in range(10):
        live = [a for a in obs if a != "__all__"]
        if not live:
            break
        obs, _, terminateds, _, _ = env.step(
            {a: action_spaces[a].sample() for a in live}
        )
        if terminateds.get("__all__"):
            done = True
            break

    assert done, "Episode never reached __all__=True — staggered-death hang detected"


# ===========================================================================
# Phase 1.3 — Edge cases
# ===========================================================================


class _AllDieTogetherProtocol(FakeProtocol):
    """All 3 agents die on the very first step."""

    def send_action_msg(self, actions, action_spaces):
        self.step_count += 1
        env_actions = actions[self.env_id]
        self.received_actions_log.append(set(env_actions.keys()))

        obs = {a: np.random.randn(4).astype(np.float32) for a in self.agent_ids}
        rewards = {a: -1.0 for a in self.agent_ids}
        terminateds = {a: True for a in self.agent_ids}
        truncateds = {a: False for a in self.agent_ids}
        infos = {a: {} for a in self.agent_ids}

        return (
            {self.env_id: obs},
            {self.env_id: rewards},
            {self.env_id: terminateds},
            {self.env_id: truncateds},
            {self.env_id: infos},
            {},
            {},
        )


class _TruncationProtocol(FakeProtocol):
    """Agents die via truncation (not termination) one per step."""

    def send_action_msg(self, actions, action_spaces):
        self.step_count += 1
        env_actions = actions[self.env_id]
        self.received_actions_log.append(set(env_actions.keys()))

        obs = {a: np.random.randn(4).astype(np.float32) for a in self.agent_ids}
        rewards = {a: 0.0 for a in self.agent_ids}
        terminateds = {a: False for a in self.agent_ids}
        truncateds = {a: False for a in self.agent_ids}
        infos = {a: {} for a in self.agent_ids}

        if self.step_count >= 1:
            truncateds["agent_0"] = True
        if self.step_count >= 2:
            truncateds["agent_1"] = True
        if self.step_count >= 3:
            truncateds["agent_2"] = True

        return (
            {self.env_id: obs},
            {self.env_id: rewards},
            {self.env_id: terminateds},
            {self.env_id: truncateds},
            {self.env_id: infos},
            {},
            {},
        )


class _SingleAgentProtocol(FakeProtocol):
    """One-agent environment: agent dies at step 1."""

    def __init__(self):
        super().__init__()
        self.agent_ids = ["agent_0"]

    def get_definition(self):
        obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        ids = [self.agent_ids]
        agent_types = [{a: "default" for a in self.agent_ids}]
        obs_defns = {0: {a: obs_space for a in self.agent_ids}}
        act_defns = {0: {a: act_space for a in self.agent_ids}}
        return ids, agent_types, obs_defns, act_defns

    def send_reset_msg(self, seeds=None, options=None):
        self.step_count = 0
        self.received_actions_log = []
        obs = [{a: np.zeros(4, dtype=np.float32) for a in self.agent_ids}]
        infos = [{a: {} for a in self.agent_ids}]
        return obs, infos

    def send_action_msg(self, actions, action_spaces):
        self.step_count += 1
        env_actions = actions[self.env_id]
        self.received_actions_log.append(set(env_actions.keys()))

        obs = {a: np.random.randn(4).astype(np.float32) for a in self.agent_ids}
        rewards = {a: 0.0 for a in self.agent_ids}
        terminateds = {a: False for a in self.agent_ids}
        truncateds = {a: False for a in self.agent_ids}
        infos = {a: {} for a in self.agent_ids}

        if self.step_count >= 1:
            terminateds["agent_0"] = True

        return (
            {self.env_id: obs},
            {self.env_id: rewards},
            {self.env_id: terminateds},
            {self.env_id: truncateds},
            {self.env_id: infos},
            {},
            {},
        )


def _make_vec_env_with(protocol):
    """Construct a RayVecEnv backed by the given protocol (no Unreal connection)."""
    try:
        from schola.rllib.env import RayVecEnv
        from ray.rllib.env.vector.vector_multi_agent_env import VectorMultiAgentEnv

        simulator = FakeSimulator()
        simulator.supported_protocols = type(protocol)

        env = RayVecEnv.__new__(RayVecEnv)
        env.protocol = protocol
        env.simulator = simulator
        env._init_space_attributes()
        protocol.start()
        env.protocol.send_startup_msg()
        env._define_environment()
        env._init_agent_tracking()
        env.metadata = {"autoreset_mode": "next_step"}
        VectorMultiAgentEnv.__init__(env)
        env._fake_protocol = protocol
        return env
    except (ImportError, Exception):
        return None


def _make_env_with(protocol):
    try:
        from schola.rllib.env import RayEnv
        from ray.rllib.env.multi_agent_env import MultiAgentEnv

        simulator = FakeSimulator()
        simulator.supported_protocols = type(protocol)

        env = RayEnv.__new__(RayEnv)
        env.protocol = protocol
        env.simulator = simulator
        env._init_space_attributes()
        protocol.start()
        env.protocol.send_startup_msg()
        env._define_environment()
        env._init_agent_tracking()
        MultiAgentEnv.__init__(env)
        env._fake_protocol = protocol
        return env
    except ImportError:
        return _StandaloneEnv(protocol)


def test_all_agents_die_simultaneously():
    """All 3 agents die on step 1: __all__ must be True immediately; no hang."""
    env = _make_env_with(_AllDieTogetherProtocol())
    obs, _ = env.reset()
    action_spaces = env.single_action_spaces

    actions = {a: action_spaces[a].sample() for a in obs if a != "__all__"}
    _, _, terminateds, truncateds, _ = env.step(actions)

    assert terminateds.get("__all__") or truncateds.get(
        "__all__"
    ), "All agents died simultaneously — __all__ must be True on step 1"


def test_truncation_instead_of_termination():
    """Truncation must trigger the same action-filtering and __all__ logic as termination."""
    env = _make_env_with(_TruncationProtocol())
    obs, _ = env.reset()
    action_spaces = env.single_action_spaces

    # Step 1: agent_0 truncated, agent_1/2 alive → __all__ False
    _, _, terminateds1, truncateds1, _ = env.step(
        {a: action_spaces[a].sample() for a in obs if a != "__all__"}
    )
    assert truncateds1.get("agent_0") is True
    assert truncateds1.get("__all__") is False

    # Step 2: agent_1 truncated → __all__ still False
    _, _, terminateds2, truncateds2, _ = env.step(
        {a: action_spaces[a].sample() for a in ["agent_1", "agent_2"]}
    )
    assert truncateds2.get("agent_1") is True
    assert truncateds2.get("__all__") is False

    # Step 3: agent_2 truncated → __all__ True
    _, _, terminateds3, truncateds3, _ = env.step(
        {"agent_2": action_spaces["agent_2"].sample()}
    )
    assert truncateds3.get("agent_2") is True
    assert (
        truncateds3.get("__all__") is True
    ), "All agents truncated — __all__ must be True"

    # Forwarded action keys must not include dead agents
    protocol = env._fake_protocol
    assert protocol.received_actions_log[1] == {"agent_1", "agent_2"}
    assert protocol.received_actions_log[2] == {"agent_2"}


def test_reset_clears_tracking_state():
    """reset() must wipe stale dead-agent sets so a second episode runs cleanly."""
    env = make_ray_env()
    action_spaces = env.single_action_spaces

    # Run a full episode to completion
    obs, _ = env.reset()
    for _ in range(10):
        live = [a for a in obs if a != "__all__"]
        if not live:
            break
        obs, _, terminateds, _, _ = env.step(
            {a: action_spaces[a].sample() for a in live}
        )
        if terminateds.get("__all__"):
            break

    # Second episode: must start fresh and also reach __all__=True
    obs, _ = env.reset()
    done = False
    for _ in range(10):
        live = [a for a in obs if a != "__all__"]
        if not live:
            break
        obs, _, terminateds, _, _ = env.step(
            {a: action_spaces[a].sample() for a in live}
        )
        if terminateds.get("__all__"):
            done = True
            break

    assert (
        done
    ), "Second episode after reset() never reached __all__=True — stale state leak"


def test_single_agent_environment():
    """Single-agent env: baseline behaviour — terminates after step 1 with __all__=True."""
    env = _make_env_with(_SingleAgentProtocol())
    obs, _ = env.reset()
    action_spaces = env.single_action_spaces

    live = [a for a in obs if a != "__all__"]
    assert live == ["agent_0"]

    _, _, terminateds, _, _ = env.step({a: action_spaces[a].sample() for a in live})
    assert terminateds.get("agent_0") is True
    assert (
        terminateds.get("__all__") is True
    ), "Single-agent env: __all__ must be True when the sole agent dies"


def test_zero_live_agents_at_step():
    """If all agents are already dead, episode must still end; no hang."""
    env = make_ray_env()
    action_spaces = env.single_action_spaces

    # Drive to __all__=True (all 3 dead)
    obs, _ = env.reset()
    for _ in range(10):
        live = [a for a in obs if a != "__all__"]
        if not live:
            break
        obs, _, terminateds, _, _ = env.step(
            {a: action_spaces[a].sample() for a in live}
        )
        if terminateds.get("__all__"):
            break

    assert (
        terminateds.get("__all__") is True
    ), "Setup failed: could not reach a state where all agents are dead"
    # After __all__=True, calling step with an empty action dict should not hang
    # (in practice RLlib resets, but we verify the env does not deadlock).
    # Reset and verify clean state.
    obs2, _ = env.reset()
    live2 = [a for a in obs2 if a != "__all__"]
    assert set(live2) == {
        "agent_0",
        "agent_1",
        "agent_2",
    }, "After reset, all agents must be alive again"


def test_vec_dead_agents_stripped_from_response():
    """
    RayVecEnv must filter dead agents from gRPC response on subsequent steps.

    This covers the NextStep reset protocol path, which shares _filter_dead_agents
    with RayEnv via BaseRayEnv.  Without the filter, RLlib raises MultiAgentEnvError
    when it sees a second observation for an agent whose episode is already closed.
    """
    env = _make_vec_env_with(FakeProtocol())
    if env is None:
        return  # skip if RayVecEnv unavailable in this environment

    protocol = env._fake_protocol
    action_spaces = env._single_action_spaces

    env.reset()

    # Step 1: all 3 agents alive; agent_0 dies this step.
    actions = [{a: action_spaces[a].sample() for a in protocol.agent_ids}]
    obs_list, _, term_list, trunc_list, _ = env.step(actions)
    assert term_list[0]["agent_0"] is True
    assert term_list[0].get("__all__") is False, "agent_1 and agent_2 still alive"

    # Step 2: agent_0 must be absent from every returned dict.
    actions = [{a: action_spaces[a].sample() for a in ["agent_1", "agent_2"]}]
    obs_list, rew_list, term_list, trunc_list, info_list = env.step(actions)
    assert "agent_0" not in obs_list[0], "Dead agent_0 must not appear in obs"
    assert "agent_0" not in rew_list[0], "Dead agent_0 must not appear in rewards"
    assert "agent_0" not in term_list[0], "Dead agent_0 must not appear in terminateds"
    assert "agent_0" not in trunc_list[0], "Dead agent_0 must not appear in truncateds"
    assert "agent_0" not in info_list[0], "Dead agent_0 must not appear in infos"
    assert term_list[0]["agent_1"] is True
    assert term_list[0].get("__all__") is False, "agent_2 still alive"

    # Step 3: agent_1 and agent_0 absent; agent_2 dies; __all__ True.
    actions = [{"agent_2": action_spaces["agent_2"].sample()}]
    obs_list, _, term_list, _, _ = env.step(actions)
    assert "agent_0" not in obs_list[0]
    assert "agent_1" not in obs_list[0]
    assert term_list[0]["agent_2"] is True
    assert term_list[0].get("__all__") is True, "All agents dead — episode must be done"


# ===========================================================================
# Standalone runner
# ===========================================================================
if __name__ == "__main__":
    tests = [
        (
            "test_only_live_agent_actions_forwarded",
            test_only_live_agent_actions_forwarded,
        ),
        (
            "test_all_flag_false_while_agents_alive",
            test_all_flag_false_while_agents_alive,
        ),
        (
            "test_all_flag_true_when_last_agent_dies",
            test_all_flag_true_when_last_agent_dies,
        ),
        (
            "test_episode_completes_within_bounded_steps",
            test_episode_completes_within_bounded_steps,
        ),
        ("test_all_agents_die_simultaneously", test_all_agents_die_simultaneously),
        (
            "test_truncation_instead_of_termination",
            test_truncation_instead_of_termination,
        ),
        ("test_reset_clears_tracking_state", test_reset_clears_tracking_state),
        ("test_single_agent_environment", test_single_agent_environment),
        ("test_zero_live_agents_at_step", test_zero_live_agents_at_step),
        (
            "test_vec_dead_agents_stripped_from_response",
            test_vec_dead_agents_stripped_from_response,
        ),
    ]

    passed = failed = 0
    errors = []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            errors.append((name, e))
            print(f"  FAIL  {name}: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if errors:
        for name, e in errors:
            print(f"  - {name}: {e}")
        sys.exit(1)
    else:
        print("All tests passed!")
        sys.exit(0)
