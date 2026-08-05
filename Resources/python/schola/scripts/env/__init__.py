# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Cyclopts CLI for Schola environment utilities (`schola env`)."""

from cyclopts import App

env_utils_app = App(
    name="schola-env",
    help="Inspect, check, and debug Schola environments.",
)

from .check.check import app as check_app
from .inspect.inspect import app as inspect_app

env_utils_app.command(inspect_app.meta, name="inspect")
env_utils_app.command(check_app.meta, name="check")

if __name__ == "__main__":
    env_utils_app()
