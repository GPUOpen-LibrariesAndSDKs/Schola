#!/usr/bin/env python3
"""
Cluster training driver — runs on the Ray head node.

Flow
----
1. Start a *local* mock-UE gRPC server for one-shot space discovery.
2. Create a temporary ``RayVecEnv`` to learn observation / action shapes.
3. ``ray.init(address="auto")`` — connect to the running Ray cluster.
4. Launch a short PPO training run via ``tune.run`` with configurable
   learner processes (``NUM_LEARNERS``, default 1) and env-runner
   workers (``NUM_WORKERS``, default 2), each talking to *its own*
   container's mock-UE server via ``localhost:50051``.
5. Print training metrics and exit.
"""

import os
import sys
from concurrent import futures

import grpc
import gymnasium as gym
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.connectors.env_to_module import FlattenObservations

from schola.rllib.env import RayVecEnv
from schola.rllib.env_runner import ScholaEnvRunner
from schola.core.protocols.protobuf.grpc_protocol import GrpcProtocol
from schola.core.simulators.external_simulator import ExternalSimulator
import schola.generated.GymConnector_pb2_grpc as gym_grpc

from schola.core.simulators.gym.servicer import VecGymToGymServiceServicer

GRPC_PORT = int(os.environ.get("SCHOLA_GRPC_PORT", "50051"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
NUM_LEARNERS = int(os.environ.get("NUM_LEARNERS", "1"))
TIMESTEPS = int(os.environ.get("TRAIN_TIMESTEPS", "2000"))


def _start_local_server(port: int):
    """Start a lightweight gRPC server for space discovery."""
    servicer = VecGymToGymServiceServicer(
        env_id=[lambda: gym.make("CartPole-v1")],
    )
    options = [
        ("grpc.max_send_message_length", 100 * 1024 * 1024),
        ("grpc.max_receive_message_length", 100 * 1024 * 1024),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2), options=options)
    gym_grpc.add_GymServiceServicer_to_server(servicer, server)
    actual_port = server.add_insecure_port(f"[::]:{port}")
    server.start()
    return server, actual_port


def main():
    print("=" * 60)
    print("  Schola  —  Docker Cluster Training Simulation")
    print("=" * 60, flush=True)

    # ── 1. Space discovery (local, ephemeral server) ──────────────────
    print("\n[head] Starting local mock-UE server for space discovery…")
    disc_server, disc_port = _start_local_server(port=0)
    print(f"[head] Discovery server on port {disc_port}")

    protocol = GrpcProtocol(url="localhost", port=disc_port, credential_mode="insecure")
    simulator = ExternalSimulator()
    tmp_env = RayVecEnv(protocol, simulator)
    agent_names = {aid: aid for aid in tmp_env.possible_agents}
    print(f"[head] Discovered agents: {list(agent_names.keys())}")
    print(f"[head] Observation space: {tmp_env.single_observation_space}")
    print(f"[head] Action space:      {tmp_env.single_action_space}")
    tmp_env.close()
    disc_server.stop(grace=0)

    # ── 2. Connect to Ray cluster ─────────────────────────────────────
    print("\n[head] Connecting to Ray cluster…")
    ray.init(address="auto")
    print(f"[head] Cluster resources: {ray.cluster_resources()}")
    nodes = ray.nodes()
    alive = [n for n in nodes if n["Alive"]]
    print(f"[head] Alive nodes: {len(alive)}")
    for n in alive:
        print(
            f"       ↳ {n['NodeManagerAddress']}  cpus={n['Resources'].get('CPU', 0)}"
        )

    # ── 3. Configure PPO ──────────────────────────────────────────────
    print(
        f"\n[head] Configuring PPO  (learners={NUM_LEARNERS}, workers={NUM_WORKERS}, timesteps={TIMESTEPS})"
    )
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(
            env_config={
                "protocol": GrpcProtocol,
                "protocol_args": {
                    "url": "localhost",
                    "port": GRPC_PORT,
                    "credential_mode": "insecure",
                    "environment_start_timeout": 120,
                },
                "port_offset_mode": "fixed",
                "simulator": ExternalSimulator,
                "simulator_args": {},
            },
        )
        .framework("torch")
        .env_runners(
            env_runner_cls=ScholaEnvRunner,
            num_env_runners=NUM_WORKERS,
            custom_resources_per_env_runner={"ue_sidecar": 1},
            env_to_module_connector=lambda env, spaces=None, device=None: FlattenObservations(
                input_observation_space=(
                    env.single_observation_space
                    if env is not None
                    else spaces["__env_single__"][0]
                ),
                input_action_space=(
                    env.single_action_space
                    if env is not None
                    else spaces["__env_single__"][1]
                ),
                multi_agent=True,
            ),
        )
        .multi_agent(
            policies={aid: PolicySpec() for aid in agent_names},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
        )
        .training(
            lr=3e-4,
            train_batch_size=512,
            minibatch_size=128,
            num_epochs=3,
        )
        .learners(num_learners=NUM_LEARNERS, num_gpus_per_learner=0)
    )

    # ── 4. Train ──────────────────────────────────────────────────────
    print("\n[head] Launching training…\n")
    stop = {"num_env_steps_sampled_lifetime": TIMESTEPS}
    results = tune.run(
        PPO,
        config=config,
        stop=stop,
        verbose=2,
        storage_path="/tmp/ray_results",
    )

    # ── 5. Report ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    trial = results.trials[0]
    last = trial.last_result
    print(f"  Episodes total ......... {last.get('episodes_this_iter', '?')}")
    print(
        f"  Timesteps sampled ...... {last.get('num_env_steps_sampled_lifetime', '?')}"
    )
    mean_reward = last.get("env_runners", {}).get("episode_reward_mean", "?")
    print(f"  Mean episode reward .... {mean_reward}")
    print(f"  Trial path ............. {trial.path}")
    print("=" * 60, flush=True)

    ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
