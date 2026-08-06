# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Sandboxed filesystem helpers for Fabrica Deep Agent tools."""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterable
from pathlib import Path

MAX_READ_BYTES = 512_000


def format_sandbox_tool_error(
    *,
    operation: str,
    target: str,
    exc: BaseException,
) -> str:
    """Return an agent-facing error string for sandboxed file-tool failures."""
    if isinstance(exc, PermissionError):
        return (
            f"Access denied while trying to {operation} {target!r}: {exc}. "
            "Paths must stay under --code-roots and outside ignored folders "
            "(e.g. Binaries, Intermediate, DerivedDataCache)."
        )
    elif isinstance(exc, FileNotFoundError):
        return f"{operation.title()} target not found: {target!r}"
    elif isinstance(exc, IsADirectoryError):
        return f"Expected a file but found a directory: {target!r}"
    elif isinstance(exc, NotADirectoryError):
        return f"Expected a directory but found a file: {target!r}"
    elif isinstance(exc, re.error):
        return f"Invalid regex pattern {target!r}: {exc}"
    else:
        return f"Failed to {operation} {target!r}: {exc}"


def _is_under_root(path: Path, roots: Iterable[Path]) -> bool:
    rp = path.resolve()
    for root in roots:
        try:
            rp.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def _matches_ignore(rel: str, globs: list[str]) -> bool:
    for g in globs:
        if fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(rel.replace("\\", "/"), g):
            return True
    return False


def assert_allowed_path(path: Path, roots: list[Path], ignore_globs: list[str]) -> Path:
    rp = path.resolve()
    if not rp.is_file() and not rp.is_dir():
        raise FileNotFoundError(str(rp))
    if not _is_under_root(rp, roots):
        raise PermissionError(f"Path outside code roots: {rp}")
    for root in roots:
        try:
            rel = str(rp.relative_to(root.resolve()))
            if _matches_ignore(rel, ignore_globs):
                raise PermissionError(f"Path matches ignore glob: {rp}")
        except ValueError:
            continue
    return rp


def _normalize_rel_path(rel_path: str, roots: list[Path]) -> str:
    """Strip a leading code-root directory name when the agent repeats it (e.g. ``Source/foo``)."""
    rel_path = rel_path.replace("\\", "/").lstrip("/")
    for root in roots:
        root_name = root.name
        if rel_path == root_name:
            return ""
        prefix = f"{root_name}/"
        if rel_path.startswith(prefix):
            return rel_path[len(prefix) :]
    return rel_path


def _resolve_by_basename(name: str, roots: list[Path]) -> Path | None:
    """When the agent guesses a wrong directory, fall back to a unique basename match."""
    if not name or name in (".", ".."):
        return None
    matches: list[Path] = []
    for root in roots:
        rr = root.resolve()
        for path in rr.rglob(name):
            if not path.is_file():
                continue
            try:
                path.resolve().relative_to(rr)
            except ValueError:
                continue
            matches.append(path.resolve())
    if len(matches) == 1:
        return matches[0]
    return None


def _resolve_under_roots(rel_path: str, roots: list[Path]) -> Path:
    if not roots:
        raise ValueError("code_roots is empty")
    rel_path = _normalize_rel_path(rel_path, roots)
    for r in roots:
        cand = (r / rel_path).resolve()
        if cand.is_file() or cand.is_dir():
            if _is_under_root(cand, roots):
                return cand
    fallback = _resolve_by_basename(Path(rel_path).name, roots)
    if fallback is not None:
        return fallback
    raise FileNotFoundError(rel_path)


def ue_list_dir(rel_path: str, roots: list[Path], ignore_globs: list[str]) -> str:
    target = assert_allowed_path(
        _resolve_under_roots(rel_path, roots), roots, ignore_globs
    )
    if not target.is_dir():
        raise NotADirectoryError(str(target))
    lines = sorted(p.name for p in target.iterdir())
    return "\n".join(lines)


def ue_read_file(rel_path: str, roots: list[Path], ignore_globs: list[str]) -> str:
    target = assert_allowed_path(
        _resolve_under_roots(rel_path, roots), roots, ignore_globs
    )
    if not target.is_file():
        raise IsADirectoryError(str(target))
    data = target.read_bytes()
    if len(data) > MAX_READ_BYTES:
        data = data[:MAX_READ_BYTES]
        return data.decode("utf-8", errors="replace") + "\n\n<<truncated>>"
    return data.decode("utf-8", errors="replace")


def ue_grep(
    pattern: str, roots: list[Path], ignore_globs: list[str], glob: str = "*.h"
) -> str:
    rx = re.compile(pattern)
    hits: list[str] = []
    for root in roots:
        rr = root.resolve()
        for path in rr.rglob(glob):
            if not path.is_file():
                continue
            try:
                rel = str(path.relative_to(rr))
            except ValueError:
                continue
            if _matches_ignore(rel, ignore_globs):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{path}:{i}:{line}")
                    if len(hits) >= 200:
                        return "\n".join(hits) + "\n<<truncated>>"
    return "\n".join(hits) if hits else "(no matches)"
