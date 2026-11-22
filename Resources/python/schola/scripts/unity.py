# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Commands for connecting and training a running Unity environment."""

from pathlib import Path

from cyclopts import App

app = App(name="unity", help="Connect and train Unity environments through Schola.")

_SHARED_POLICY_RULES = (
    "schola unity train gives every agent its own slot of one shared PPO policy, "
    "so every agent needs the same observation and action spaces and all agents "
    "must finish an episode on the same step. For agents that differ or finish at "
    "different times, run schola unity bridge and train with RLlib."
)


@app.command
def bridge(unity_port: int = 8001, port: int = 8000, timeout: float = 30):
    """Expose Unity to existing Schola trainers; stop with Ctrl+C."""
    from schola.unity.bridge import UnityBridgeServer

    with UnityBridgeServer(unity_port, port, timeout) as adapter:
        print(f"Schola Unity adapter listening on 127.0.0.1:{adapter.port}", flush=True)
        try:
            adapter.server.wait_for_termination()
        except KeyboardInterrupt:
            pass


@app.command
def train(
    unity_port: int = 8001,
    timesteps: int = 10000,
    checkpoint: Path = Path("ckpt/unity_ppo"),
    seed: int = 0,
    timeout: float = 30,
    export_onnx: bool = False,
    onnx_path: Path | None = None,
):
    """Train PPO using Schola's existing SB3 environment and save a checkpoint.

    Start Unity Play Mode with a TrainingServer first. Continuous action spaces
    must have finite bounds (set TranslateActuator.MaxDisplacement above zero).

    A multi-agent Unity environment trains one shared policy: each agent is one
    slot of the vectorized environment, so every agent needs the same observation
    and action spaces and all agents must finish an episode on the same step.
    One rollout collects 128 steps per agent, and timesteps counts agent steps.

    With --export-onnx (or --onnx-path), the final policy is also exported with
    Schola's ONNX exporter so Unity's SentisPolicy can run it.
    """
    if timesteps < 1:
        raise ValueError("timesteps must be positive")
    import gymnasium as gym
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecMonitor

    from schola.core.error_manager import EnvironmentException
    from schola.core.protocols.protobuf.grpc_protocol import GrpcProtocol
    from schola.core.simulators.external_simulator import ExternalSimulator
    from schola.sb3.env import VecEnv
    from schola.sb3.utils import VecMergeDictActionWrapper
    from schola.unity.bridge import UnityBridgeServer

    with UnityBridgeServer(unity_port, port=0, timeout=timeout) as adapter:
        try:
            env = VecEnv(ExternalSimulator(), GrpcProtocol("127.0.0.1", adapter.port))
        except AssertionError as error:
            # Schola's VecEnv asserts that every agent shares one pair of spaces.
            raise ValueError(f"{_SHARED_POLICY_RULES}\n{error}") from error
        merge_wrapper = None
        try:
            if isinstance(env.action_space, gym.spaces.Dict):
                merge_wrapper = VecMergeDictActionWrapper(env)
                env = merge_wrapper
            if isinstance(env.action_space, gym.spaces.Box) and (
                not np.isfinite(env.action_space.low).all()
                or not np.isfinite(env.action_space.high).all()
            ):
                raise ValueError(
                    "Set finite action bounds, e.g. TranslateActuator.MaxDisplacement = 0.1"
                )
            env = VecMonitor(env)
            policy = (
                "MultiInputPolicy"
                if isinstance(env.observation_space, gym.spaces.Dict)
                else "MlpPolicy"
            )
            model = PPO(policy, env, n_steps=128, batch_size=64, seed=seed, verbose=1)
            try:
                model.learn(total_timesteps=timesteps)
            except EnvironmentException as error:
                # Raised when agents of one environment finish on different steps.
                raise RuntimeError(f"{_SHARED_POLICY_RULES}\n{error}") from error

            if merge_wrapper is not None:
                action_space = merge_wrapper.unmerged_action_space
                setattr(model, "__unmerged_action_space", action_space)
            else:
                action_space = model.action_space

            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            model.save(checkpoint)
            print(f"Saved Unity policy to {checkpoint}", flush=True)

            if export_onnx or onnx_path is not None:
                from schola.sb3.export import save_model_as_onnx

                if onnx_path is None:
                    onnx_path = checkpoint.with_suffix(".onnx")
                onnx_path.parent.mkdir(parents=True, exist_ok=True)
                save_model_as_onnx(model, str(onnx_path), action_space=action_space)
                print(f"Exported ONNX policy to {onnx_path}", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    app()
