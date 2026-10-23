"""Tabular Q-learning client for the Schola D2 demo.

Talks to Godot's ScholaLocalConnector (the D2 stand-in for US3) with newline-delimited JSON over
localhost TCP. As in Schola's GymService, Python sends every request and Godot answers it:
start, definition, then update (reset, step, close). See Godot/docs/local_protocol.md.

    python trainer.py --port 8000 --episodes 300 --policy-out ../policies/q_policy.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import socket
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

POLICY_FORMAT = "schola-d2-qtable"


class StopRequested(Exception):
    pass


def space_from_dict(definition: dict) -> gym.Space:
    # Unbounded Box sides arrive as null.
    kind = definition["type"]
    if kind == "Box":
        low = np.array([-np.inf if v is None else v for v in definition["low"]], dtype=np.float32)
        high = np.array([np.inf if v is None else v for v in definition["high"]], dtype=np.float32)
        shape = tuple(int(d) for d in definition["shape"])
        return gym.spaces.Box(low.reshape(shape), high.reshape(shape), shape, dtype=np.dtype(definition["dtype"]))
    if kind == "Discrete":
        return gym.spaces.Discrete(int(definition["n"]))
    if kind == "MultiDiscrete":
        return gym.spaces.MultiDiscrete([int(n) for n in definition["nvec"]])
    if kind == "MultiBinary":
        return gym.spaces.MultiBinary(int(definition["n"]))
    if kind == "Dict":
        return gym.spaces.Dict({key: space_from_dict(value) for key, value in definition["spaces"].items()})
    raise ValueError(f"unknown space type {kind!r}")


class GodotConnection:
    def __init__(self, host: str, port: int, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            try:
                self.sock = socket.create_connection((host, port), timeout=2.0)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise SystemExit(f"could not connect to Godot at {host}:{port}; press Start Training first")
                time.sleep(0.25)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(None)
        self.reader = self.sock.makefile("r", encoding="utf-8")

    def notify(self, message: dict) -> None:
        """Sends a one-way demo message (progress, saved, log); Godot does not reply."""
        self.sock.sendall((json.dumps(message) + "\n").encode("utf-8"))

    def request(self, message: dict, expected: str) -> dict:
        """Sends one request and returns Godot's reply, which must have type ``expected``."""
        self.sock.sendall((json.dumps(message) + "\n").encode("utf-8"))
        line = self.reader.readline()
        if not line:
            raise ConnectionError("Godot closed the connection")
        reply = json.loads(line)
        if reply.get("type") == "error":
            if reply.get("code") == "CLOSED":
                raise StopRequested(reply.get("message", ""))
            raise RuntimeError(f"Godot rejected {message['type']} ({reply.get('code')}): {reply.get('message')}")
        if reply.get("type") != expected:
            raise RuntimeError(f"expected {expected} from Godot, got {reply}")
        return reply

    def close_session(self) -> None:
        """Ends the session cleanly; safe to call after Godot has already closed."""
        try:
            self.request({"type": "update", "status": "CLOSED"}, "close_result")
        except (OSError, ConnectionError, RuntimeError, StopRequested, json.JSONDecodeError):
            pass

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


class GodotGridEnv(gym.Env):
    def __init__(self, connection: GodotConnection) -> None:
        self.connection = connection
        connection.request({"type": "start", "autoreset_mode": "DISABLED"}, "start_result")
        environments = connection.request({"type": "definition"}, "definition_result")["environments"]
        if len(environments) != 1 or len(environments[0]) != 1:
            raise RuntimeError(f"this trainer supports one environment with one agent, got {environments}")
        (self.agent_id, self.definition), = environments[0].items()
        self.observation_space = space_from_dict(self.definition["observation_space"])
        self.action_space = space_from_dict(self.definition["action_space"])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        reply = self.connection.request(
            {"type": "update", "reset": {"environments": {"0": {"seed": seed, "options": options or {}}}}},
            "reset_result",
        )
        agent = reply["initial_state"]["0"][self.agent_id]
        return self._observation(agent), agent.get("info", {})

    def step(self, action):
        # The client validates actions before sending, as Schola's Python package does.
        if not self.action_space.contains(action):
            raise ValueError(f"action {action!r} is outside {self.action_space}")
        reply = self.connection.request(
            {"type": "update", "step": {"actions": [{self.agent_id: int(action)}]}},
            "step_result",
        )
        agent = reply["state"][0][self.agent_id]
        return (
            self._observation(agent),
            float(agent["reward"]),
            bool(agent["terminated"]),
            bool(agent["truncated"]),
            agent.get("info", {}),
        )

    def _observation(self, reply: dict) -> np.ndarray:
        observation = np.asarray(reply["observation"], dtype=self.observation_space.dtype)
        if not self.observation_space.contains(observation):
            raise RuntimeError(f"observation {observation} is outside {self.observation_space}")
        return observation


def state_key(observation) -> str:
    """Must match ScholaSavedPolicy.state_key() in Godot."""
    return ",".join(str(int(round(float(v)))) for v in observation)


def save_policy(path: Path, env: GodotGridEnv, q_table: dict, episodes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "format": POLICY_FORMAT,
        "observation_space": env.definition["observation_space"],
        "action_space": env.definition["action_space"],
        "episodes": episodes,
        "q": {key: [float(v) for v in values] for key, values in q_table.items()},
    }
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def export_onnx_policy(policy_path: Path, onnx_path: Path) -> None:
    try:
        from export_onnx import export
    except ImportError as error:
        print(f"skipped ONNX export: {error}", file=sys.stderr)
        return
    checked = export(policy_path, onnx_path)
    print(f"exported ONNX policy to {onnx_path} ({checked} states checked)")


def train(env: GodotGridEnv, args: argparse.Namespace) -> tuple[dict, int]:
    n_actions = int(env.action_space.n)
    q_table: dict[str, list[float]] = {}
    rng = random.Random(args.seed)
    completed = 0
    try:
        for episode in range(1, args.episodes + 1):
            epsilon = max(args.epsilon_min, math.exp(-4.0 * episode / args.episodes))
            observation, _ = env.reset(seed=rng.randrange(1 << 30))
            key = state_key(observation)
            total = 0.0
            outcome = "running"
            while True:
                values = q_table.setdefault(key, [0.0] * n_actions)
                if rng.random() < epsilon:
                    action = rng.randrange(n_actions)
                else:
                    action = int(np.argmax(values))
                observation, reward, terminated, truncated, info = env.step(action)
                next_key = state_key(observation)
                next_values = q_table.setdefault(next_key, [0.0] * n_actions)
                target = reward if terminated else reward + args.gamma * max(next_values)
                values[action] += args.alpha * (target - values[action])
                total += reward
                key = next_key
                if terminated or truncated:
                    outcome = info.get("outcome", "done")
                    break
            completed = episode
            env.connection.notify({
                "type": "progress",
                "episode": episode,
                "episodes": args.episodes,
                "epsilon": epsilon,
                "states": len(q_table),
            })
            if episode % 10 == 0 or episode == 1:
                print(f"episode {episode:4d}  reward {total:7.2f}  {outcome:8s}  epsilon {epsilon:.2f}  states {len(q_table)}")
    except StopRequested as stop:
        print(f"Godot asked to stop after {completed} episodes: {stop}")
    return q_table, completed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--policy-out", type=Path, default=Path(__file__).resolve().parent.parent / "policies" / "q_policy.json")
    parser.add_argument("--onnx-out", type=Path, help="Also export the policy to ONNX (needs the onnx package)")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    args = parser.parse_args()

    print(f"connecting to Godot at {args.host}:{args.port} ...")
    connection = GodotConnection(args.host, args.port, args.connect_timeout)
    try:
        env = GodotGridEnv(connection)
        print(f"observation space: {env.observation_space}")
        print(f"action space:      {env.action_space}")
        print(f"agent:             {env.agent_id}")
        q_table, completed = train(env, args)
        save_policy(args.policy_out, env, q_table, completed)
        print(f"saved policy with {len(q_table)} states to {args.policy_out}")
        if args.onnx_out is not None:
            export_onnx_policy(args.policy_out, args.onnx_out)
        try:
            connection.notify({"type": "saved", "path": str(args.policy_out)})
        except OSError:
            pass
        connection.close_session()
    except StopRequested as stop:
        print(f"Godot ended the session: {stop}", file=sys.stderr)
        return 1
    except (ConnectionError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
