# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Scripts for training with Ray and RLlib.
"""

from cyclopts import App, Parameter, validators, group_extractors, Group

rllib_app = App(name="schola-rllib", help="Use Ray-Rllib with Schola!")

from .train.train import app as train_app
from .eval.eval import app as eval_app
from .collect.collect import app as collect_app
from .offline_train.offline_train import app as offline_train_app

from .rllib_to_onnx import export_onnx_app

rllib_app.command(train_app.meta, name="train")
rllib_app.command(eval_app.meta, name="eval")
rllib_app.command(collect_app.meta, name="collect")
rllib_app.command(offline_train_app.meta, name="offline-train")

rllib_app.command(export_onnx_app, name="export")


if __name__ == "__main__":
    rllib_app()
