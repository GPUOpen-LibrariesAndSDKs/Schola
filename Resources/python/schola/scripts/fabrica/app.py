# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Cyclopts CLI for Schola Fabrica."""

from __future__ import annotations

import logging

from cyclopts import App

from schola.scripts.fabrica.run_cli import fabrica_run_cli

logger = logging.getLogger(__name__)

fabrica_app = App(
    name="fabrica",
    help="Schola Fabrica: Deep-Agent C++ reward shaping for AFabricaEnvironment + SB3 scoring.",
)

fabrica_app.command(fabrica_run_cli.meta, name="run")


if __name__ == "__main__":
    fabrica_app()
