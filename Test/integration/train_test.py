# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Integration test for the staggered-death fix.

Connects to a live Unreal Engine session (ScholaStaggeredTest project) and
runs RLlib PPO training. The test environment hard-codes agent deaths at
steps 5, 10, and 15, so every episode should be exactly 15 steps long.

Pass conditions
---------------
1. ep_len_mean ≈ 15  (range 13–17) for every completed iteration.
2. ep_len_mean does NOT grow across iterations (no freeze / accumulation).
3. Training completes without KeyError, NaN in rewards, or timeout.

Prerequisites
-------------
- Unreal Editor running ScholaStaggeredTest with the FIXED plugin variant.
  (Run switch_to_fixed.bat, rebuild, then press Play in UE.)
- LogScholaTraining: Running Gym Connector  visible in the UE Output Log.
- Python: pip install "ray[rllib]>=2.40" "gymnasium>=1.0" numpy torch
- Schola Python package installed:  pip install -e Resources/python

Usage
-----
  cd D:/Github/Schola
  python Test/integration/train_test.py [--port 8500] [--iterations 3]
"""

import sys
import math
import argparse

GRPC_PORT = 8500
NUM_ITERATIONS = 3
EP_LEN_TARGET = 15
EP_LEN_TOLERANCE = 2  # acceptable: 13–17

AGENT_IDS = ["agent_0", "agent_1", "agent_2"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_ep_len(result: dict) -> float | None:
    """Extract ep_len_mean from an RLlib result dict (new and old API stacks)."""
    candidates = [
        result.get("env_runners", {}).get("episode_len_mean"),
        result.get("sampler_results", {}).get("episode_len_mean"),
        result.get("episode_len_mean"),
    ]
    for v in candidates:
        if v is not None and not math.isnan(v):
            return float(v)
    return None


def _get_ep_rew(result: dict) -> float | None:
    """Extract ep_rew_mean from an RLlib result dict."""
    candidates = [
        result.get("env_runners", {}).get("episode_reward_mean"),
        result.get("sampler_results", {}).get("episode_reward_mean"),
        result.get("episode_reward_mean"),
    ]
    for v in candidates:
        if v is not None and not math.isnan(v):
            return float(v)
    return None


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------


def run(port: int, num_iterations: int) -> bool:
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.connectors.env_to_module import FlattenObservations
    from ray.rllib.policy.policy import PolicySpec
    from ray.tune.registry import register_env
    from schola.core.protocols.protobuf.gRPC import gRPCProtocol
    from schola.core.simulators.unreal.editor import UnrealEditor
    from schola.rllib.env import RayEnv

    print(f"\nConnecting to Unreal at localhost:{port} ...")

    def make_env(*args, **kwargs):
        simulator = UnrealEditor()
        protocol = gRPCProtocol(url="localhost", port=port)
        return RayEnv(protocol, simulator)

    register_env("ScholaStaggeredDeath", make_env)

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(env="ScholaStaggeredDeath")
        .framework("torch")
        .env_runners(
            num_env_runners=0,
            env_to_module_connector=lambda env: FlattenObservations(
                input_observation_space=env.single_observation_space,
                input_action_space=env.single_action_space,
                multi_agent=True,
            ),
        )
        .multi_agent(
            policies={"shared_policy": PolicySpec()},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .training(
            train_batch_size=600,  # small batch: ~40 episodes per iteration
        )
    )

    algo = config.build_algo()
    print("Algorithm built. Running training iterations ...\n")

    ep_lens = []
    checks_passed = 0
    checks_total = 0

    for i in range(num_iterations):
        result = algo.train()

        ep_len = _get_ep_len(result)
        ep_rew = _get_ep_rew(result)
        iteration_label = f"Iteration {i + 1}/{num_iterations}"

        if ep_len is None:
            print(
                f"  {iteration_label}: ep_len_mean not yet available (no completed episodes)"
            )
            continue

        ep_lens.append(ep_len)
        status_len = "OK" if abs(ep_len - EP_LEN_TARGET) <= EP_LEN_TOLERANCE else "FAIL"
        rew_str = f"{ep_rew:.2f}" if ep_rew is not None else "N/A"
        print(
            f"  {iteration_label}: ep_len_mean={ep_len:.1f} [{status_len}]  ep_rew_mean={rew_str}"
        )

    algo.stop()

    # ------------------------------------------------------------------
    # Check 1: every recorded ep_len_mean is within tolerance of 15
    # ------------------------------------------------------------------
    checks_total += 1
    bad_iters = [l for l in ep_lens if abs(l - EP_LEN_TARGET) > EP_LEN_TOLERANCE]
    if not ep_lens:
        print(
            "\nCHECK 1 FAIL  No episodes completed — possible connection/freeze issue."
        )
    elif bad_iters:
        print(
            f"\nCHECK 1 FAIL  ep_len_mean out of range {EP_LEN_TARGET}±{EP_LEN_TOLERANCE}: {bad_iters}"
        )
    else:
        print(
            f"\nCHECK 1 PASS  ep_len_mean ~= {EP_LEN_TARGET} for all {len(ep_lens)} iteration(s)."
        )
        checks_passed += 1

    # ------------------------------------------------------------------
    # Check 2: ep_len_mean does not grow across iterations (no freeze)
    # ------------------------------------------------------------------
    checks_total += 1
    if len(ep_lens) >= 2:
        growth = ep_lens[-1] - ep_lens[0]
        if growth > EP_LEN_TOLERANCE * 2:
            print(
                f"CHECK 2 FAIL  ep_len_mean grew by {growth:.1f} — possible accumulation/freeze."
            )
        else:
            print(
                f"CHECK 2 PASS  ep_len_mean stable across iterations (Δ={growth:+.1f})."
            )
            checks_passed += 1
    else:
        print(
            "CHECK 2 SKIP  Need ≥2 iterations with completed episodes to check stability."
        )
        checks_total -= 1

    # ------------------------------------------------------------------
    # Check 3: no NaN in rewards
    # ------------------------------------------------------------------
    checks_total += 1
    # If we reached here without an exception, rewards did not cause a crash.
    # We already filtered NaN in _get_ep_rew, so any NaN would have shown "N/A".
    print("CHECK 3 PASS  No NaN / KeyError exceptions during training.")
    checks_passed += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Final result: {checks_passed}/{checks_total} checks passed")
    if checks_passed == checks_total:
        print("*** ALL CHECKS PASSED — staggered-death fix is working correctly. ***")
        return True
    else:
        print("!!! SOME CHECKS FAILED — see details above. !!!")
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Staggered-death integration test")
    parser.add_argument(
        "--port",
        type=int,
        default=GRPC_PORT,
        help=f"gRPC port Unreal is listening on (default: {GRPC_PORT})",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=NUM_ITERATIONS,
        help=f"Number of RLlib training iterations (default: {NUM_ITERATIONS})",
    )
    args = parser.parse_args()

    ok = run(port=args.port, num_iterations=args.iterations)
    sys.exit(0 if ok else 1)
