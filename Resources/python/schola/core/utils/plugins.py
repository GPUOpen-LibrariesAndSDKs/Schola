# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Helpers for working with entry_point plugins for Schola
"""


def get_plugins(group_name: str) -> list[object]:
    """
    Returns a list of plugins for a given group name.

    Parameters
    ----------
    group_name : str
        The name of the plugin group to search for.

    Returns
    -------
    list[object]
        A list of loaded plugin objects for the specified group name.
    """
    from importlib.metadata import entry_points

    eps = entry_points()
    if hasattr(eps, "select"):
        discovered_plugins = eps.select(group=group_name)
    elif isinstance(eps, dict):
        # Python 3.9: entry_points() returned dict[str, list[EntryPoint]]
        discovered_plugins = eps.get(group_name, [])
    else:
        discovered_plugins = []
    return [plugin.load() for plugin in discovered_plugins]
