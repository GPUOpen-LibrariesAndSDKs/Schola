# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Script to convert a Ray model to an ONNX model for use in Unreal Engine.
"""

import logging
from cyclopts import App, types

from schola.scripts.common.console import configure_logging

logger = logging.getLogger(__name__)

export_onnx_app = App(
    name="rllib-to-onnx",
    help="Convert a Ray RLlib Algorithm Checkpoint to ONNX for Unreal Engine",
)


@export_onnx_app.default
def export(
    policy_checkpoint_path: types.ExistingDirectory,
    output_path: types.Directory | None = None,
    spaces_path: types.ExistingFile | None = None,
):
    """
    Export an RLlib Algorithm Checkpoint to Schola ONNX format.

    Parameters
    ----------
    policy_checkpoint_path : types.ExistingDirectory
        The path to the RLlib Algorithm Checkpoint.
    output_path : types.Directory | None
        The path to save the ONNX model to. If not provided, the model will be saved in the same directory as the checkpoint.
    spaces_path : types.ExistingFile, optional
        YAML model-space configuration used instead of checkpoint spaces,
        either shared by every policy or keyed by policy ID (only listed
        policies are exported).
    """
    configure_logging(logging.INFO)
    if output_path is None:
        output_path = policy_checkpoint_path
    from schola.core.model_space_config import load_policy_model_spaces
    from schola.rllib.export import export_onnx_from_policy

    if spaces_path is None:
        from ray.rllib.algorithms.algorithm import Algorithm

        logger.info(
            "No spaces YAML provided; using observation and action spaces "
            "stored in the RLlib checkpoint."
        )
        algo = Algorithm.from_checkpoint(str(policy_checkpoint_path))
        export_onnx_from_policy(algo, output_path)
    else:
        observation_space, action_space = load_policy_model_spaces(spaces_path)
        export_onnx_from_policy(
            policy_checkpoint_path, output_path, observation_space, action_space
        )


if __name__ == "__main__":
    export_onnx_app()
