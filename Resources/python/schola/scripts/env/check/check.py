# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Check a Schola-backed environment with Gymnasium's environment checker.
"""

from __future__ import annotations

import logging
import signal
from dataclasses import replace
from typing import Any, Callable

from cyclopts import App

from schola.core.error_manager import (
    MultipleAgentsException,
    MultipleEnvironmentsException,
)
from schola.scripts.common.command_template import ScholaCommandTemplate
from schola.scripts.common.settings import (
    AllSingularSimulatorConfigs,
    BaseSimulatorConfig,
    SingularExecutableSimulatorConfig,
    SingularExternalSimulatorConfig,
    SingularGymSimulatorConfig,
    SingularProjectSimulatorConfig,
)
from schola.scripts.env.check.settings import EnvCheckScriptSettings
from schola.scripts.env.utils import run_gym_env_checker

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

logger = logging.getLogger(__name__)


def main(args: EnvCheckScriptSettings) -> None:
    """
    Start an environment and run Gymnasium's environment checker on it.

    Parameters
    ----------
    args : EnvCheckScriptSettings
        CLI / script configuration.
    """
    from schola.core.error_manager import ScholaErrorContextManager
    from schola.gym.env import GymEnv

    env = None
    try:
        with ScholaErrorContextManager():
            sim_args = args.environment_settings.simulator_settings

            protocol_args = args.environment_settings.protocol_settings

            try:
                env = GymEnv(
                    sim_args.make(),
                    protocol_args.make(),
                    verbosity=args.logging_settings.schola_verbosity,
                )
            except MultipleEnvironmentsException:
                logger.error(
                    "schola env check is not supported for vectorized environments. "
                    "Please retry with a non-vectorized environment."
                )
                return
            except MultipleAgentsException:
                logger.error(
                    "schola env check is not supported for multi-agent environments. "
                    "Please retry with a single-agent environment."
                )
                return

            logger.info("Checking environment:")
            logger.info("  Action space: %s", env.action_space)
            logger.info("  Observation space: %s", env.observation_space)

            run_gym_env_checker(env, logger=logger)
    except (KeyboardInterrupt, Exception) as exc:
        if isinstance(exc, KeyboardInterrupt):
            logger.info("Ctrl-C received. Shutting down gracefully;")
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise
    finally:
        if env is not None:
            env.close()


app = App(
    name="check",
    help="Start a Schola environment and run Gymnasium's environment checker on it.",
)


class EnvCheckCommand(ScholaCommandTemplate[EnvCheckScriptSettings]):
    @property
    def algorithm_table(self):
        return {}

    @property
    def script_args_type(self) -> type[EnvCheckScriptSettings]:
        return EnvCheckScriptSettings

    @property
    def main_func(self) -> Callable[[EnvCheckScriptSettings], None]:
        return main

    @property
    def simulator_table(self) -> dict[str, type[BaseSimulatorConfig[Any]]]:

        return {
            "gym": SingularGymSimulatorConfig,
            "executable": SingularExecutableSimulatorConfig,
            "project": SingularProjectSimulatorConfig,
            "external": SingularExternalSimulatorConfig,
        }


app = EnvCheckCommand(app, logger).make()

if __name__ == "__main__":
    app.meta()
