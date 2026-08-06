# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Fabrica ``run`` CLI built with ``MetaAlgCommand`` (same structure as ``schola sb3 train``)."""

from __future__ import annotations

import logging

from cyclopts import App

from schola.scripts.common.console import configure_logging
from schola.scripts.common.settings import UnrealProjectSimulatorConfig
from schola.scripts.common.command_template import (
    ScholaCommandTemplate,
    SimulatorArgsType,
)
from schola.scripts.fabrica.settings import (
    FabricaScriptSettings,
)
from schola.scripts.sb3.train.settings import (
    PPOTrainSettings,
    SACTrainSettings,
)

logger = logging.getLogger(__name__)


class MetaFabricaRunCommand(ScholaCommandTemplate[FabricaScriptSettings]):
    """``MetaAlgCommand`` for Fabrica: ``ppo`` | ``sac`` × simulator layout like SB3 train."""

    @property
    def algorithm_table(self):
        return {
            "sac": SACTrainSettings,
            "ppo": PPOTrainSettings,
        }

    @property
    def algorithm_help(self):
        return {
            "sac": "Train with Soft Actor-Critic (SAC) and Stable-Baselines3.",
            "ppo": "Train with Proximal Policy Optimization (PPO) and Stable-Baselines3.",
        }

    @property
    def simulator_table(self) -> dict[str, type[SimulatorArgsType]]:
        return {
            "project": UnrealProjectSimulatorConfig,
        }


def fabrica_run_main(settings: FabricaScriptSettings) -> None:
    """Entrypoint invoked by Cyclopts after ``MetaFabricaRunCommand`` parsing."""
    from schola.scripts.fabrica.runner import run_fabrica_loop

    configure_logging()

    for label, p in (
        ("env_header", settings.paths_settings.env_header),
        ("task_description", settings.paths_settings.task_description),
    ):
        if not p.exists():
            raise FileNotFoundError(
                f"Fabrica {label} must be an existing path (got {p.resolve()!s})."
            )

    run_fabrica_loop(settings)


_inner_run_app = App(
    name="run",
    help="Run Fabrica (Deep Agent codegen + optional SB3 scoring).",
)
fabrica_run_cli = MetaFabricaRunCommand(
    _inner_run_app,
    FabricaScriptSettings,
    fabrica_run_main,
    logger,
).make()
