# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Inspect a Schola-backed environment: report agent definitions and validate one reset.
"""

from __future__ import annotations

import logging
import signal
from typing import Callable, Type

from cyclopts import App

from schola.scripts.common.command_template import ScholaCommandTemplate
from schola.scripts.env.settings import EnvInspectScriptSettings
from schola.scripts.env.utils import (
    inspect_agents,
)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

logger = logging.getLogger(__name__)


def main(args: EnvInspectScriptSettings) -> None:
    """
    Start an environment, log its definition, and inspect a single reset.

    Parameters
    ----------
    args : EnvInspectScriptSettings
        CLI / script configuration.
    """
    from schola.core.error_manager import ScholaErrorContextManager
    from schola.gym.env import GymVectorEnv

    env = None
    try:
        with ScholaErrorContextManager():
            n_sim = args.environment_settings.simulator_settings.num_simulators
            if n_sim > 1:
                logger.warning(
                    "Multiple simulators (-n=%d) is not supported for inspect; "
                    "using a single simulator.",
                    n_sim,
                )

            sim_args = args.environment_settings.simulator_settings
            protocol_args = args.environment_settings.protocol_settings
            env = GymVectorEnv(
                sim_args.make(),
                protocol_args.make(),
                verbosity=args.logging_settings.schola_verbosity,
            )
            
            logger.info("Environment definitions:")
            logger.info("  Sub-environments: %d", env.id_manager.num_envs)
            logger.info("  Total agents: %d", env.id_manager.num_ids)

            inspect_agents(
                env,
                logger=logger,
                seed=args.environment_settings.seed,
                options=args.environment_settings.env_options or None,
            )
    except (KeyboardInterrupt, Exception) as exc:
        if isinstance(exc, KeyboardInterrupt):
            logger.info("Ctrl-C received. Shutting down gracefully;")
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise
    finally:
        if env is not None:
            env.close()


app = App(
    name="inspect",
    help="Start a Schola environment and report agent definitions plus one reset.",
)


class EnvInspectCommand(ScholaCommandTemplate[EnvInspectScriptSettings]):
    @property
    def algorithm_table(self):
        return {}

    @property
    def script_args_type(self) -> Type[EnvInspectScriptSettings]:
        return EnvInspectScriptSettings

    @property
    def main_func(self) -> Callable[[EnvInspectScriptSettings], None]:
        return main


app = EnvInspectCommand(app, logger).make()

if __name__ == "__main__":
    app.meta()
