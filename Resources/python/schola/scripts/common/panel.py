# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Lightweight panel printing utilities for CLI scripts.

Provides simple helpers to present messages (info / warning / error) using
Cyclopts' rich panel integration. Panels are printed on the shared
``console`` from ``schola.scripts.common.console``, which is also passed to
the top-level Cyclopts ``App`` as both ``console`` and ``error_console`` so
Cyclopts does not allocate a separate stderr console.
"""

from __future__ import annotations

from typing import Iterable
import sys
from cyclopts import CycloptsPanel

from schola.scripts.common.console import console

__all__ = [
    "print_panel",
    "print_error",
    "print_warning",
    "print_info",
]

STYLE_ERROR = "red"
STYLE_WARNING = "yellow"
STYLE_INFO = "cyan"


def print_panel(
    message: str | Iterable[str], *, title: str = "", style: str = STYLE_INFO
) -> None:
    """
    Print a panel with the given message and style.

    Parameters
    ----------
    message : str | Iterable[str]
        The message to print.
    title : str, optional
        The title of the panel, by default ""
    style : str, optional
        The style of the panel, by default STYLE_INFO
    """
    if not isinstance(message, str):
        message = "\n".join(str(m) for m in message)
    console.print(CycloptsPanel(message=message, title=title or "Message", style=style))


def print_error(message: str | Iterable[str]) -> None:  # noqa: D401
    """
    Print an error panel and terminate with exit code 1.

    Parameters
    ----------
    message : str | Iterable[str]
        The message to print.
    """
    print_panel(message, title="Error", style=STYLE_ERROR)
    sys.exit(1)


def print_warning(message: str | Iterable[str]) -> None:  # noqa: D401
    """
    Print a non-fatal warning using a Rich panel (yellow styling).

    Parameters
    ----------
    message : str or iterable of str
        Body text; iterables are joined with newlines.
    """
    print_panel(message, title="Warning", style=STYLE_WARNING)


def print_info(message: str | Iterable[str]) -> None:  # noqa: D401
    """
    Print an informational message using a Rich panel (cyan styling).

    Parameters
    ----------
    message : str or iterable of str
        Body text; iterables are joined with newlines.
    """
    print_panel(message, title="Info", style=STYLE_INFO)
