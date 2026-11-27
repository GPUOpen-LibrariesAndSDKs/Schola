# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Scripts for training with Ray and RLlib.
"""

from cyclopts import App, Parameter, validators, group_extractors, Group

rllib_app = App(name="schola-rllib", help="Use Ray-Rllib with Schola!")

from .train.train import app as train_app
from .eval.eval import app as eval_app

from .rllib_to_onnx import export_onnx_app
from .rllib_to_iree import export_iree_app

rllib_app.command(train_app.meta, name="train")
rllib_app.command(eval_app.meta, name="eval")

rllib_app.command(export_onnx_app, name="export")
rllib_app.command(export_iree_app, name="export-iree")


if __name__ == "__main__":
    rllib_app()
