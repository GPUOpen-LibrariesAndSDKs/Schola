#!/usr/bin/env python3
"""
Standalone mock Unreal Engine gRPC server.

Uses the ``VecGymToGymServiceServicer`` from ``schola.core.simulators.gym``,
backing each instance with a single CartPole-v1 gymnasium environment.
Each worker container in the Docker simulation runs one instance of this
server, mimicking the sidecar-UE pattern used in a real KubeRay deployment.
"""

import argparse
import signal
import sys
import threading
from concurrent import futures

import grpc
import gymnasium as gym

from schola.core.simulators.gym.servicer import VecGymToGymServiceServicer

import schola.generated.GymConnector_pb2_grpc as gym_grpc


def serve(port: int):
    servicer = VecGymToGymServiceServicer(
        env_id=[lambda: gym.make("CartPole-v1")],
    )
    options = [
        ("grpc.max_send_message_length", 100 * 1024 * 1024),
        ("grpc.max_receive_message_length", 100 * 1024 * 1024),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=options)
    gym_grpc.add_GymServiceServicer_to_server(servicer, server)
    bound_port = server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"[mock-UE] gRPC server listening on port {bound_port}", flush=True)

    stop_event = threading.Event()

    def _shutdown(signum, frame):
        print("[mock-UE] Shutting down…", flush=True)
        server.stop(grace=2)
        stop_event.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    stop_event.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50051)
    args = parser.parse_args()
    serve(args.port)
