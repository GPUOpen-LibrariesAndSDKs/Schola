# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Schola Fabrica: LLM-driven C++ reward authoring and SB3 scoring loop."""

from __future__ import annotations

__all__ = ["fabrica_app"]


def __getattr__(name: str):
    if name == "fabrica_app":
        from schola.scripts.fabrica.app import fabrica_app

        return fabrica_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
