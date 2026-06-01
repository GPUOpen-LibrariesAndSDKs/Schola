# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Direct RayVecEnv stepping test for the staggered-death fix.

Drives RayVecEnv through 3 complete episodes per environment slot (6 episodes
total across 2 slots) without an RLlib training loop.  This exercises the
_filter_dead_agents path in RayVecEnv.step() directly against a live Unreal
session.

Why not use RLlib training (as in train_test.py)?
-------------------------------------------------
RLlib's new API stack wraps every registered env in SyncVectorMultiAgentEnv.
With num_env_runners >= 1, the remote MultiAgentEnvRunner opens its own gRPC
connection, conflicting with the driver's connection (Unreal accepts only one
client).  With num_env_runners=0, SyncVectorMultiAgentEnv calls RayVecEnv.step()
with a single MultiAgentDict instead of List[MultiAgentDict], breaking the
interface.  Direct stepping avoids all of this while still exercising the exact
C++/Python code paths that the fix protects.

Episode-length note (NEXT_STEP protocol)
-----------------------------------------
Episode 1 starts from an explicit reset() call, so its length = exact Blueprint
death schedule (15 steps).  Episodes 2+ begin on the step immediately after
__all__=True; Unreal resets the env on that same gRPC call and returns the
surviving agent's state.  Python counts that transition step as step 1 of the
new episode, making episodes 2+ one step longer (16).  Both 15 and 16 are within
the accepted tolerance window (13-17).

Pass conditions
---------------
1. All 6 episodes complete with ep_len in 13-17 (no hang / freeze).
2. No agent that died in step D appears in any later step D+1, D+2, ...
   of the same episode (verifies _filter_dead_agents in RayVecEnv.step).
3. No Python exceptions during the run.

Prerequisites
-------------
- Unreal Editor running ScholaStaggeredTest with the FIXED plugin variant and
  TWO BP_TestStaggeredEnvActor instances in the level.
  (Run switch_to_fixed.bat, rebuild, then press Play in UE.)
- LogScholaTraining: Running Gym Connector  visible in the UE Output Log.
- Python: pip install "ray[rllib]>=2.40" "gymnasium>=1.0" numpy torch
- Schola Python package installed:  pip install -e Resources/python

Usage
-----
  cd D:/Github/Schola
  python Test/integration/train_test_vec.py [--port 8500]
"""

import sys
import argparse

GRPC_PORT = 8500
TARGET_EPISODES_PER_ENV = 3
EP_LEN_TARGET = 15
EP_LEN_TOLERANCE = 2  # acceptable: 13-17
MAX_STEPS = TARGET_EPISODES_PER_ENV * (EP_LEN_TARGET + EP_LEN_TOLERANCE + 5) * 4


def run(port: int) -> bool:
    from schola.core.protocols.protobuf.gRPC import gRPCProtocol
    from schola.core.simulators.unreal.editor import UnrealEditor
    from schola.rllib.env import RayVecEnv

    print(f"\nConnecting to Unreal at localhost:{port} (RayVecEnv direct stepping) ...")

    simulator = UnrealEditor()
    protocol = gRPCProtocol(url="localhost", port=port)
    env = RayVecEnv(protocol, simulator)

    num_envs = env.num_envs
    num_agents = len(env.possible_agents)
    print(f"Connected.  num_envs={num_envs}  agents_per_env={num_agents}\n")

    obs_list, _ = env.reset()

    # Per-env tracking
    episode_count = [0] * num_envs  # completed episodes
    episode_steps = [0] * num_envs  # steps elapsed in the current episode
    dead_seen = [set() for _ in range(num_envs)]  # agents dead this episode

    all_ep_lengths: list[int] = []
    errors: list[str] = []
    total_steps = 0

    while min(episode_count) < TARGET_EPISODES_PER_ENV and total_steps < MAX_STEPS:

        # ------------------------------------------------------------------
        # Snapshot dead-agent sets BEFORE the step.
        # Reappearance check: an agent dead BEFORE this step must not appear
        # in THIS step's returned observations.
        # (Appearing on the step it first dies is correct; appearing on any
        # subsequent step within the same episode is the bug.)
        # ------------------------------------------------------------------
        dead_before = [s.copy() for s in dead_seen]

        # ------------------------------------------------------------------
        # Build random actions for currently visible agents
        # ------------------------------------------------------------------
        actions = [
            {
                agent_id: env.single_action_space[agent_id].sample()
                for agent_id in obs_list[env_id]
            }
            for env_id in range(num_envs)
        ]

        # ------------------------------------------------------------------
        # Step
        # ------------------------------------------------------------------
        obs_list, rewards, terminateds, truncateds, infos = env.step(actions)
        total_steps += 1

        # ------------------------------------------------------------------
        # Post-step validation and state updates
        # ------------------------------------------------------------------
        for env_id in range(num_envs):
            episode_steps[env_id] += 1

            # Reappearance check (correct): agent was dead BEFORE this step
            # and must NOT appear in THIS step's observations.
            for agent_id in obs_list[env_id]:
                if agent_id in dead_before[env_id]:
                    errors.append(
                        f"Env {env_id} ep {episode_count[env_id] + 1} "
                        f"step {episode_steps[env_id]}: "
                        f"dead agent '{agent_id}' reappeared in observations "
                        f"(_filter_dead_agents not working)"
                    )

            # Accumulate agents that die in THIS step
            for agent_id, flag in terminateds[env_id].items():
                if agent_id != "__all__" and flag:
                    dead_seen[env_id].add(agent_id)
            for agent_id, flag in truncateds[env_id].items():
                if agent_id != "__all__" and flag:
                    dead_seen[env_id].add(agent_id)

            # Episode completion
            ep_done = terminateds[env_id].get("__all__", False) or truncateds[
                env_id
            ].get("__all__", False)
            if ep_done:
                ep_len = episode_steps[env_id]
                ok_str = (
                    "OK" if abs(ep_len - EP_LEN_TARGET) <= EP_LEN_TOLERANCE else "FAIL"
                )
                print(
                    f"  Env {env_id}  Episode {episode_count[env_id] + 1}: "
                    f"ep_len={ep_len} [{ok_str}]"
                )
                all_ep_lengths.append(ep_len)
                episode_count[env_id] += 1
                episode_steps[env_id] = 0
                dead_seen[env_id] = set()

    try:
        env.close_extras()
    except Exception:
        pass

    total_expected = TARGET_EPISODES_PER_ENV * num_envs
    checks_passed = 0
    checks_total = 0

    # ------------------------------------------------------------------
    # Check 1 — all episodes completed with correct length
    # ------------------------------------------------------------------
    checks_total += 1
    bad_lens = [l for l in all_ep_lengths if abs(l - EP_LEN_TARGET) > EP_LEN_TOLERANCE]
    if len(all_ep_lengths) < total_expected:
        print(
            f"\nCHECK 1 FAIL  Only {len(all_ep_lengths)}/{total_expected} episodes "
            f"completed before step limit (possible hang or freeze)."
        )
    elif bad_lens:
        print(f"\nCHECK 1 FAIL  Episodes with out-of-range length: {bad_lens}")
    else:
        print(
            f"\nCHECK 1 PASS  All {len(all_ep_lengths)}/{total_expected} episodes "
            f"completed with ep_len in [{EP_LEN_TARGET - EP_LEN_TOLERANCE}, "
            f"{EP_LEN_TARGET + EP_LEN_TOLERANCE}]."
        )
        checks_passed += 1

    # ------------------------------------------------------------------
    # Check 2 — no dead-agent reappearance (core fix verification)
    # ------------------------------------------------------------------
    checks_total += 1
    reapp_errors = [e for e in errors if "reappeared" in e]
    if reapp_errors:
        print(
            f"CHECK 2 FAIL  Dead-agent reappearance detected ({len(reapp_errors)} instance(s)):"
        )
        for e in reapp_errors:
            print(f"    {e}")
    else:
        print(
            "CHECK 2 PASS  No dead agent appeared in observations after its "
            "death step (_filter_dead_agents working correctly in RayVecEnv)."
        )
        checks_passed += 1

    # ------------------------------------------------------------------
    # Check 3 — no unexpected errors or exceptions
    # ------------------------------------------------------------------
    checks_total += 1
    other_errors = [e for e in errors if e not in reapp_errors]
    if other_errors:
        print(f"CHECK 3 FAIL  Unexpected errors ({len(other_errors)}):")
        for e in other_errors:
            print(f"    {e}")
    else:
        print("CHECK 3 PASS  No unexpected errors or exceptions.")
        checks_passed += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Final result: {checks_passed}/{checks_total} checks passed")
    if checks_passed == checks_total:
        print(
            "*** ALL CHECKS PASSED -- "
            "RayVecEnv staggered-death fix is working correctly. ***"
        )
        return True
    else:
        print("!!! SOME CHECKS FAILED -- see details above. !!!")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Staggered-death integration test (RayVecEnv direct stepping)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=GRPC_PORT,
        help=f"gRPC port Unreal is listening on (default: {GRPC_PORT})",
    )
    args = parser.parse_args()
    ok = run(port=args.port)
    sys.exit(0 if ok else 1)
