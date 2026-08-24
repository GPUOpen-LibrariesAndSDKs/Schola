# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Lightweight panel printing utilities for CLI scripts.

Provides simple helpers to present messages (info / warning / error) using
Cyclopts' rich panel integration. Error panels are rendered through a dedicated
``error_console`` so that they match the width and styling of the panels
Cyclopts emits for its own runtime errors (Cyclopts' default ``error_formatter``
is :func:`~cyclopts.CycloptsPanel`, which we reuse here).

The consoles live in ``schola.scripts.common.console`` and are shared with the
top-level Cyclopts ``App`` (see ``schola.scripts.launch``) so that every panel
— whether emitted by Cyclopts or by these helpers — is printed to the same
console object, guaranteeing consistent widths.
"""

from __future__ import annotations

from typing import Iterable
import sys
from cyclopts import CycloptsPanel
from rich.console import Console

from schola.scripts.common.console import console, error_console

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
    message: str | Iterable[str],
    *,
    title: str = "",
    style: str = STYLE_INFO,
    console_: Console | None = None,
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
    console_ : Console, optional
        The console to print to. Defaults to the shared ``console``.
    """
    if not isinstance(message, str):
        message = "\n".join(str(m) for m in message)
    (console_ or console).print(
        CycloptsPanel(message=message, title=title or "Message", style=style)
    )


def print_error(message: str | Iterable[str]) -> None:  # noqa: D401
    """
    Print an error panel and terminate with exit code 1.

    The panel is rendered with :func:`~cyclopts.CycloptsPanel` on the shared
    ``error_console`` (stderr), matching the panels Cyclopts prints for its own
    runtime errors.

    Parameters
    ----------
    message : str | Iterable[str]
        The message to print.
    """
    print_panel(message, title="Error", style=STYLE_ERROR, console_=error_console)
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
