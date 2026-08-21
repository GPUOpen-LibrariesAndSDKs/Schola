# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Helpers for configuring the Stable Baselines3 :class:`~stable_baselines3.common.logger.Logger`.

These replace :func:`stable_baselines3.common.utils.configure_logger` so that the
human-readable tabular output is routed through the shared Schola Rich console
(which writes to ``stderr``) instead of writing directly to ``stdout``. This keeps
SB3's progress tables consistent with the rest of the CLI output and leaves
``stdout`` free for piping.

Only the output formats reachable from the Schola CLI are handled here: the
human-readable console output and (optionally) tensorboard.
"""

from __future__ import annotations

import os
from typing import TextIO, cast

import stable_baselines3.common.logger as sb3_logger
from stable_baselines3.common.logger import (
    HumanOutputFormat,
    KVWriter,
    Logger,
    TensorBoardOutputFormat,
)
from stable_baselines3.common.utils import get_latest_run_id

from rich.file_proxy import FileProxy

from schola.scripts.common.console import console

# ``SummaryWriter`` is ``None`` on this attribute when tensorboard is not installed.
SummaryWriter = getattr(sb3_logger, "SummaryWriter", None)

__all__ = ["configure_sb3_logger"]


def configure_sb3_logger(
    verbose: int = 0,
    tensorboard_log: str | None = None,
    tb_log_name: str = "",
    reset_num_timesteps: bool = True,
) -> Logger:
    """Configure an SB3 :class:`Logger` that logs human-readable output via the shared console.

    This mirrors :func:`stable_baselines3.common.utils.configure_logger` but replaces
    the ``stdout`` output format with one that writes to the shared Schola Rich console.

    Parameters
    ----------
    verbose : int, default=0
        Verbosity level: 0 for no console output, 1 (or higher) to include the
        standard human-readable output in the logger outputs.
    tensorboard_log : str or None, default=None
        The log location for tensorboard. If ``None``, no tensorboard logging is done.
    tb_log_name : str, default=""
        The tensorboard run name used to build the run subdirectory.
    reset_num_timesteps : bool, default=True
        Whether the ``num_timesteps`` attribute is reset. When ``False`` the run
        continues in the previous tensorboard directory instead of creating a new one.

    Returns
    -------
    Logger
        The configured SB3 logger.

    Raises
    ------
    ImportError
        If ``tensorboard_log`` is provided but tensorboard is not installed.
    """
    if tensorboard_log is not None and SummaryWriter is None:
        raise ImportError(
            "Trying to log data to tensorboard but tensorboard is not installed."
        )

    output_formats: list[KVWriter] = []
    save_path: str | None = None

    if verbose >= 1:
        fake_stdout = FileProxy(console, console.file)
        output_formats.append(HumanOutputFormat(cast(TextIO, fake_stdout)))

    if tensorboard_log is not None:
        latest_run_id = get_latest_run_id(tensorboard_log, tb_log_name)
        if not reset_num_timesteps:
            # Continue training in the same directory.
            latest_run_id -= 1
        save_path = os.path.join(tensorboard_log, f"{tb_log_name}_{latest_run_id + 1}")
        os.makedirs(save_path, exist_ok=True)
        output_formats.append(TensorBoardOutputFormat(save_path))

    logger = Logger(folder=save_path, output_formats=output_formats)
    if save_path is not None:
        logger.log(f"Logging to {save_path}")
    return logger
