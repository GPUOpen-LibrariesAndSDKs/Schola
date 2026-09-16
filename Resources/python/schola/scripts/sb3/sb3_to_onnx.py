# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Script to convert a Stable Baselines3 model to an ONNX model for use in Unreal Engine.
"""

import enum
import logging
from cyclopts import App, Parameter, types
from typing import Annotated, Literal

from schola.scripts.common.console import configure_logging

export_onnx_app = App(
    name="sb3-to-onnx",
    help="Convert a StableBaselines 3 policy to ONNX for Unreal Engine",
)


def _get_algorithm_class(algo: str):

    import stable_baselines3 as sb3

    _ALGORITHM_CLASSES = {
        "PPO": sb3.PPO,
        "A2C": sb3.A2C,
        "SAC": sb3.SAC,
        "TD3": sb3.TD3,
        "DDPG": sb3.DDPG,
        "DQN": sb3.DQN,
    }
    if algo not in _ALGORITHM_CLASSES:
        raise ValueError(f"Invalid algorithm: {algo}")
    return _ALGORITHM_CLASSES[algo.upper()]


AlgorithmIdentifier = Literal["PPO", "A2C", "SAC", "TD3", "DDPG", "DQN"]


@export_onnx_app.default
def export(
    policy_checkpoint_path: types.ExistingFile,
    output_path: types.File,
    algorithm: AlgorithmIdentifier,
):
    """
    Export a SB3 policy to ONNX format.
    Parameters
    ----------
    policy_checkpoint_path : types.ExistingFile
        The path to the SB3 policy checkpoint .
    output_path : types.File
        The path to save the ONNX model to.
    algorithm : Algorithm, default="PPO"
        The SB3 algorithm used to train the policy being loaded.
    """
    configure_logging(logging.INFO)
    from schola.sb3.export import convert_ckpt_to_onnx_for_unreal

    trainer = _get_algorithm_class(algorithm)
    convert_ckpt_to_onnx_for_unreal(
        trainer=trainer,
        model_path=str(policy_checkpoint_path),
        export_path=str(output_path),
    )


if __name__ == "__main__":
    export_onnx_app()
