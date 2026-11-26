# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Train a tiny 1D "line world" agent with SB3 (via Schola) and export it to IREE.

Run from ``Plugins/Schola``::

    python -m Test.sb3.line_world_iree_demo [--timesteps 20000] [--out ./line_world_out]

Environment
-----------
* The agent starts at position 0.
* Discrete actions: 0 = do nothing, 1 = forward (+100), 2 = backward (-100).
* Reaching +500 gives reward 1.00 and ends the episode.
* Reaching -500 gives reward 0.10 and ends the episode.
* Every other step gives reward -0.01.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete

STEP_SIZE = 100
GOAL_POSITION = 500
LOSE_POSITION = -500
GOAL_REWARD = 1.00
LOSE_REWARD = 0.10
STEP_PENALTY = -0.01

ACTION_NOOP, ACTION_FORWARD, ACTION_BACKWARD = 0, 1, 2


class LineWorldEnv(gym.Env):
    """1D walk from 0 towards +500 (see module docstring)."""

    def __init__(self, max_steps: int = 50):
        super().__init__()
        self.max_steps = max_steps
        # Observation is the position normalized to [-1, 1].
        self.observation_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = Discrete(3)
        self.position = 0
        self.step_count = 0

    def _obs(self) -> np.ndarray:
        return np.array([self.position / GOAL_POSITION], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.position = 0
        self.step_count = 0
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        if action == ACTION_FORWARD:
            self.position += STEP_SIZE
        elif action == ACTION_BACKWARD:
            self.position -= STEP_SIZE
        self.step_count += 1

        terminated = False
        reward = STEP_PENALTY
        if self.position >= GOAL_POSITION:
            reward, terminated = GOAL_REWARD, True
        elif self.position <= LOSE_POSITION:
            reward, terminated = LOSE_REWARD, True
        truncated = not terminated and self.step_count >= self.max_steps
        return self._obs(), reward, terminated, truncated, {}


def train(timesteps: int, out_dir: Path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    env = make_vec_env(LineWorldEnv, n_envs=4, seed=0)
    model = PPO("MlpPolicy", env, verbose=1, seed=0)
    model.learn(total_timesteps=timesteps)
    ckpt = out_dir / "line_world_ppo.zip"
    model.save(ckpt)
    return model, ckpt


def evaluate_sb3(model, episodes: int = 5) -> float:
    env = LineWorldEnv()
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, total = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
        returns.append(total)
    return float(np.mean(returns))


def evaluate_iree(vmfb_path: Path, episodes: int = 5) -> float:
    from schola.core.export.iree_export import load_iree_module

    vm_module, meta = load_iree_module(vmfb_path)
    entry = getattr(vm_module, meta.entry_function)
    env = LineWorldEnv()
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, total = False, 0.0
        while not done:
            (inp,) = meta.call_inputs
            arg = np.ascontiguousarray(obs[None, :], dtype=np.dtype(inp.dtype))
            outputs = entry(arg)
            if not isinstance(outputs, (tuple, list)):
                outputs = (outputs,)
            out = outputs[0]
            action = int(np.asarray(out.to_host() if hasattr(out, "to_host") else out).reshape(-1)[0])
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
        returns.append(total)
    return float(np.mean(returns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--out", type=Path, default=Path("line_world_out"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model, ckpt = train(args.timesteps, args.out)
    print(f"Saved checkpoint: {ckpt}")

    from schola.sb3.export import save_model_as_iree

    vmfb = args.out / "line_world.vmfb"
    save_model_as_iree(model, str(vmfb))
    print(f"Exported IREE module: {vmfb}")

    print(f"SB3 mean return:  {evaluate_sb3(model):.3f}")
    print(f"IREE mean return: {evaluate_iree(vmfb):.3f}")


if __name__ == "__main__":
    main()
