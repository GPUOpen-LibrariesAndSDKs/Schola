# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Shared Rich console, logging setup, and tqdm helpers for Schola CLI scripts."""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generator, Iterable

from rich.console import Console
from rich.logging import RichHandler

# stderr keeps stdout pipeable for subprocess / piping workflows
console = Console(stderr=True)

_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger with RichHandler on the shared console (idempotent)."""
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED:
        root.setLevel(level)
        return

    for handler in list(root.handlers):
        if isinstance(handler, logging.StreamHandler) and handler.stream in (
            sys.stdout,
            sys.stderr,
        ):
            root.removeHandler(handler)

    rich_handler = RichHandler(
        console=console,
        show_path=True,
        rich_tracebacks=True,
        show_time=False,
    )
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(rich_handler)
    root.setLevel(level)
    _CONFIGURED = True


@contextmanager
def maybe_tqdm(
    enabled: bool,
) -> Generator[Callable[..., Iterable[Any]], None, None]:
    """Yield a tqdm wrapper for progress bars, or a no-op when disabled."""
    if not enabled:
        yield lambda iterable, **kwargs: iterable
        return

    try:
        import tqdm
    except ImportError:
        logging.getLogger(__name__).warning(
            "tqdm not installed. Disabling progress bar."
        )
        yield lambda iterable, **kwargs: iterable
        return

    from tqdm.rich import tqdm as rich_tqdm

    def tqdm_with_console(iterable: Iterable[Any], **kwargs: Any) -> Iterable[Any]:
        opts = kwargs.pop("options", {})
        opts.setdefault("console", console)
        return rich_tqdm(iterable, options=opts, **kwargs)

    yield tqdm_with_console


@contextmanager
def supplemental_file_logging(
    log_path: Path,
    level: int = logging.INFO,
    formatter: (
        str | logging.Formatter
    ) = "%(asctime)s %(levelname)s [%(name)s] %(message)s",
) -> Generator[None, None, None]:
    """Attach a file handler to the root logger for the duration of the context."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(level)
    if isinstance(formatter, str):
        formatter = logging.Formatter(formatter)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()
