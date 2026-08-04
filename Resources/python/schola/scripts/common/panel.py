# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""Lightweight panel printing utilities for CLI scripts.

Provides simple helpers to present messages (info / warning / error) using
Cyclopts' rich panel integration. This module intentionally excludes any
exception hooking or context manager capture logic; scripts should decide how
exceptions are handled explicitly.
"""

from __future__ import annotations

from typing import Iterable
import sys
from rich.console import Console
from cyclopts import CycloptsPanel

__all__ = [
    "print_panel",
    "print_error",
    "print_warning",
    "print_info",
]

_console = Console()

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
    _console.print(
        CycloptsPanel(message=message, title=title or "Message", style=style)
    )

def print_error(message: str | Iterable[str]) -> None:  # noqa: D401
    """
    Print an error panel and terminate with exit code 1.

    Parameters
    ----------
    message : str | Iterable[str]
        The message to print.
    """
    print_panel(message, title="Error", style=STYLE_ERROR)
    try:
        sys.exit(1)
    except SystemExit:
        raise

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
