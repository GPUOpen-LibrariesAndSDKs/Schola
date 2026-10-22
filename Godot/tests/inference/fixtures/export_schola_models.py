#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Exports small untrained Stable-Baselines3 PPO policies with Schola's own ONNX exporter, to check that
the Godot inference reads what Python actually produces.

Next to each model, <name>.expected.txt holds one observation and the action ONNX Runtime returns for it:
    observation <values...>
    action <values...>

Regenerate with the Schola Python package installed (pip install -e "./Resources/python[sb3]"):
    python Godot/tests/inference/fixtures/export_schola_models.py
"""

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import onnxruntime
import torch as th
from schola.sb3.export import SB3PPOModel
from stable_baselines3.common.policies import (
    ActorCriticPolicy,
    MultiInputActorCriticPolicy,
)

FIXTURES_DIRECTORY = Path(__file__).resolve().parent

MODELS = {
    "schola_ppo_discrete": (
        gym.spaces.Box(-1.0, 1.0, (4,), np.float32),
        gym.spaces.Discrete(3),
        {"obs": np.array([[0.5, -0.25, 0.75, -1.0]], np.float32)},
    ),
    "schola_ppo_box": (
        gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
        gym.spaces.Box(-1.0, 1.0, (2,), np.float32),
        {"obs": np.array([[0.1, 0.2, -0.3]], np.float32)},
    ),
    "schola_ppo_dict": (
        gym.spaces.Dict(
            {
                "position": gym.spaces.Box(-1.0, 1.0, (2,), np.float32),
                "target": gym.spaces.Box(-1.0, 1.0, (2,), np.float32),
            }
        ),
        gym.spaces.MultiDiscrete([3, 2]),
        {
            "position": np.array([[0.3, -0.6]], np.float32),
            "target": np.array([[-0.2, 0.9]], np.float32),
        },
    ),
}


def export(name, observation_space, action_space):
    th.manual_seed(0)
    policy_class = (
        MultiInputActorCriticPolicy
        if isinstance(observation_space, gym.spaces.Dict)
        else ActorCriticPolicy
    )
    policy = policy_class(observation_space, action_space, lambda _: 0.0)
    SB3PPOModel(policy).save_as_onnx(FIXTURES_DIRECTORY / f"{name}.onnx")


def write_expected(name, feeds):
    session = onnxruntime.InferenceSession(str(FIXTURES_DIRECTORY / f"{name}.onnx"))
    outputs = session.run(None, feeds)
    assert len(outputs) == 1, f"{name}: expected a single action output"
    observation = np.concatenate([value.reshape(-1) for value in feeds.values()])
    action = outputs[0].reshape(-1)
    lines = [
        "observation " + " ".join(repr(float(v)) for v in observation),
        "action " + " ".join(repr(float(v)) for v in action),
    ]
    (FIXTURES_DIRECTORY / f"{name}.expected.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    for name, (observation_space, action_space, feeds) in MODELS.items():
        export(name, observation_space, action_space)
        write_expected(name, feeds)
        print(f"Exported {name}.onnx")
    print(
        f"onnxruntime {onnxruntime.__version__}, torch {th.__version__}, python {sys.version.split()[0]}"
    )


if __name__ == "__main__":
    main()
