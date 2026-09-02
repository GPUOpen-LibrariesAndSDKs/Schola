# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Shared Rich console, logging setup, and tqdm helpers for Schola CLI scripts."""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import IO, Any, Callable, Generator, Iterable, TextIO, cast

from rich.console import Console
from rich.file_proxy import FileProxy
from rich.highlighter import ReprHighlighter
from rich.logging import RichHandler


class ScholaReprHighlighter(ReprHighlighter):
    """``ReprHighlighter`` that also styles the float specials NumPy prints.

    Group names map to ``repr.<name>`` styles, so reusing ``number`` gives
    ``inf``/``-inf``/``nan`` the same colour as ordinary numbers in space
    bounds such as ``Box(-inf, inf, (4,), float32)``.
    """

    highlights = [
        *ReprHighlighter.highlights,
        r"(?<![\w.])(?P<number>[-+]?(?:inf|infinity|nan))\b",
    ]


# stderr keeps stdout pipeable for subprocess / piping workflows
console = Console(stderr=True, highlighter=ScholaReprHighlighter())

SCHOLA_LOGGER_NAME = "schola"
# Root stays quiet so third-party libraries do not inherit Schola verbosity.
ROOT_LOG_LEVEL = logging.WARNING


def _strip_console_stream_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if isinstance(handler, logging.StreamHandler) and handler.stream in (
            sys.stdout,
            sys.stderr,
        ):
            logger.removeHandler(handler)


def _has_rich_handler(logger: logging.Logger) -> bool:
    return any(isinstance(handler, RichHandler) for handler in logger.handlers)


def configure_logging(level: int = logging.WARNING) -> None:
    """Install a shared Rich handler and set the ``schola`` logger level.

    The root logger is kept at ``WARNING`` so third-party libraries stay quiet
    unless they are configured separately (for example
    :func:`schola.scripts.rllib.utils.configure_ray_logging` or SB3's own
    verbosity). Schola modules use ``logging.getLogger(__name__)`` and inherit
    from the ``schola`` logger.

    Callers that expose ``--schola-verbosity`` should pass
    :attr:`schola.scripts.common.settings.BaseLoggingSettings.log_level`.
    The Rich handler is installed once; the
    ``schola`` level is applied on every call.

    Parameters
    ----------
    level : int, default=logging.WARNING
        Level for the ``schola`` logger.
    """
    root = logging.getLogger()
    _strip_console_stream_handlers(root)
    if not _has_rich_handler(root):
        rich_handler = RichHandler(
            console=console,
            show_path=True,
            rich_tracebacks=True,
            show_time=False,
            highlighter=ScholaReprHighlighter(),
        )
        rich_handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(rich_handler)
    root.setLevel(ROOT_LOG_LEVEL)
    logging.getLogger(SCHOLA_LOGGER_NAME).setLevel(level)


def redirect_logger_to_console(name: str, level: int | str | None = None) -> None:
    """Route a third-party logger through the handler installed by ``configure_logging``.

    Libraries such as Ray attach their own stderr handler to their top-level
    logger on import and set ``propagate = False``, which bypasses the shared
    Rich handler. Dropping those handlers and re-enabling propagation puts their
    records back on the shared console.

    Parameters
    ----------
    name : str
        Name of the logger to redirect, e.g. ``"ray"``.
    level : int or str, optional
        Level to apply to the logger. When ``None`` the existing level is kept.
    """
    lib_logger = logging.getLogger(name)
    for handler in list(lib_logger.handlers):
        lib_logger.removeHandler(handler)
    lib_logger.propagate = True
    if level is not None:
        lib_logger.setLevel(level)


class ConsolePrintSettingsProxy(Console):
    """
    Proxy for the Rich Console that allows for injecting settings into print calls that are forwarded to the console.
    such as soft_wrap, highlight, and markup.
    """

    def __init__(
        self,
        console: Console,
        soft_wrap: bool = True,
        highlight: bool = True,
        markup: bool = True,
    ):
        self.__console = console
        self.soft_wrap = soft_wrap
        self.highlight = highlight
        self.markup = markup

    def print(self, *args, **kwargs) -> None:
        kwargs.setdefault("soft_wrap", self.soft_wrap)
        kwargs.setdefault("highlight", self.highlight)
        kwargs.setdefault("markup", self.markup)
        self.__console.print(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__console, name)


@contextmanager
def redirect_stdout_to_console(target_console: Console) -> Generator[None, None, None]:
    """Send everything written to ``stdout`` to the shared console instead.

    Used around third-party code that reports progress with bare ``print``
    (Ray Tune's progress reporters, Ray's worker log forwarding) so its output
    shares the console with the rest of the CLI.
    """
    writer = FileProxy(target_console, sys.stdout)
    with redirect_stdout(cast(TextIO, writer)):
        try:
            yield
        finally:
            writer.flush()


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
