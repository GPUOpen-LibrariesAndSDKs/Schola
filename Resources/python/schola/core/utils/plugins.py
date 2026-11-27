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

    return [plugin.load() for plugin in entry_points(group=group_name)]
