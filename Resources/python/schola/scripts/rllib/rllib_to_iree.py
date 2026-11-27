# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Script to convert a Ray model to an IREE ``.vmfb`` for use in Unreal Engine.
"""

import logging
from cyclopts import App, types

from schola.scripts.common.console import configure_logging

export_iree_app = App(
    name="rllib-to-iree",
    help="Convert a Ray RLlib Algorithm Checkpoint to IREE (.vmfb) for Unreal Engine",
)


@export_iree_app.default
def export(
    policy_checkpoint_path: types.ExistingDirectory,
    output_path: types.Directory | None = None,
):
    """
    Export an RLlib Algorithm Checkpoint to Schola IREE format.

    Parameters
    ----------
    policy_checkpoint_path : types.ExistingDirectory
        The path to the RLlib Algorithm Checkpoint.
    output_path : types.Directory | None
        The path to save the IREE model to. If not provided, the model will be
        saved in the same directory as the checkpoint.
    """
    configure_logging(logging.INFO)
    if output_path is None:
        output_path = policy_checkpoint_path
    from ray.rllib.algorithms.algorithm import Algorithm
    from schola.rllib.export import export_iree_from_policy

    algo = Algorithm.from_checkpoint(str(policy_checkpoint_path))

    export_iree_from_policy(algo, output_path)


if __name__ == "__main__":
    export_iree_app()
