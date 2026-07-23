# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Typing protocols for spawning duplicate simulator instances.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable


S = TypeVar("S")

@runtime_checkable
class SupportsSpawn(Protocol[S]):
    """Protocol for types that can spawn additional copies of themselves."""

    def spawn(self, count: int = 1) -> list[S]:
        """Return a list of new instances with the same configuration as ``self``."""
        ...
