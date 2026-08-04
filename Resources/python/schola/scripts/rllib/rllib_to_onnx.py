# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Script to convert a Ray model to an ONNX model for use in Unreal Engine.
"""

from typing import Annotated
from cyclopts import App, Parameter, types

export_onnx_app = App(
    name="rllib-to-onnx",
    help="Convert a Ray RLlib Algorithm Checkpoint to ONNX for Unreal Engine",
)

@export_onnx_app.default
def export(
    policy_checkpoint_path: types.ExistingDirectory,
    output_path: types.Directory | None = None,
):
    """
    Export an RLlib Algorithm Checkpoint to Schola ONNX format.

    Parameters
    ----------
    policy_checkpoint_path : types.ExistingDirectory
        The path to the RLlib Algorithm Checkpoint.
    output_path : types.ExistingDirectory | None
        The path to save the ONNX model to. If not provided, the model will be saved in the same directory as the checkpoint.
    """
    if output_path is None:
        output_path = policy_checkpoint_path
    from ray.rllib.algorithms.algorithm import Algorithm
    from schola.rllib.export import export_onnx_from_policy

    algo = Algorithm.from_checkpoint(str(policy_checkpoint_path))

    export_onnx_from_policy(algo, output_path)

if __name__ == "__main__":
    export_onnx_app()
