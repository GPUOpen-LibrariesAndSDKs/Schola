# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Helpers for working with entry_point plugins for Schola
"""

from typing import Any, List, cast


def get_plugins(group_name: str) -> List[Any]:
    """
    Returns a list of plugins for a given group name.

    Parameters
    ----------
    group_name : str
        The name of the plugin group to search for.

    Returns
    -------
    List
        A list of loaded plugin objects for the specified group name.
    """
    from importlib.metadata import entry_points

    eps = entry_points()
    if hasattr(eps, "select"):
        discovered_plugins = eps.select(group=group_name)
    else:
        discovered_plugins = cast(Any, eps).get(group_name, [])
    return [x.load() for x in discovered_plugins]
