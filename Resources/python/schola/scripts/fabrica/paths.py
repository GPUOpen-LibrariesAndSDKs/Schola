# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Path and header helpers for Fabrica codegen."""

from __future__ import annotations

import re
from pathlib import Path

_FABRICA_ENV_CLASS_RE = re.compile(
    r"class\s+(?:\w+_API\s+)?(\w+)\s*:\s*public\s+AFabricaEnvironment",
    re.MULTILINE,
)


def include_path_from_header(header: Path) -> str:
    """Return the UE-style quoted include path (e.g. ``Environment/Foo.h``)."""
    header = header.resolve()
    parts = header.parts
    for anchor in ("Public", "Private"):
        if anchor in parts:
            idx = parts.index(anchor)
            return "/".join(parts[idx + 1 :])
    return header.name


def parse_class_name_from_header(header: Path) -> str:
    """Return the concrete ``AFabricaEnvironment`` subclass name from a user header."""
    text = header.resolve().read_text(encoding="utf-8")
    match = _FABRICA_ENV_CLASS_RE.search(text)
    if not match:
        raise ValueError(
            f"No AFabricaEnvironment subclass found in {header.resolve()!s}. "
            "Expected e.g. ``class AYourEnv : public AFabricaEnvironment``."
        )
    return match.group(1)
