# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Scripts for training with Stable Baselines 3.
"""

from cyclopts import App

sb3_app = App(name="schola-sb3", help="Use StableBaselines3 with Schola!")

from .train.train import app as train_app
from .eval.eval import app as eval_app

from .sb3_to_onnx import export_onnx_app

sb3_app.command(train_app.meta, name="train")
sb3_app.command(eval_app.meta, name="eval")

sb3_app.command(export_onnx_app, name="export")

if __name__ == "__main__":
    sb3_app()
