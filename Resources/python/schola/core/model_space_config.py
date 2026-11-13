# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Load model observation and action space definitions from YAML."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete


def load_model_spaces(config_path: str | Path) -> tuple[Dict, Dict]:
    """
    Build named observation and action spaces from a YAML configuration.

    Nested keys are flattened into dotted model input and output names.
    ``observation`` and ``action`` are configuration section names, not parts
    of nested tensor names.
    """
    return _model_spaces(_read_config(config_path))


def load_policy_model_spaces(
    config_path: str | Path,
) -> tuple[Dict | dict[str, Dict], Dict | dict[str, Dict]]:
    """
    Build model spaces shared by every policy, or defined per policy ID.

    A file with top-level ``observation`` and ``action`` sections applies to
    every policy. Otherwise each top-level key is a policy ID with its own
    ``observation`` and ``action`` sections, and dicts mapping policy ID to
    space are returned.
    """
    raw = _read_config(config_path)
    if "observation" in raw:
        return _model_spaces(raw)
    spaces_by_policy = {
        policy_id: _model_spaces(sections) for policy_id, sections in raw.items()
    }
    return (
        {policy_id: obs for policy_id, (obs, _) in spaces_by_policy.items()},
        {policy_id: act for policy_id, (_, act) in spaces_by_policy.items()},
    )


def _read_config(config_path: str | Path) -> Mapping[str, Any]:
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def _model_spaces(sections: Mapping[str, Any]) -> tuple[Dict, Dict]:
    if missing := {"observation", "action"} - sections.keys():
        raise ValueError(
            f"Spaces config is missing top-level {sorted(missing)} sections; "
            f"found {sorted(sections)}"
        )
    return (
        Dict(dict(_iter_leaf_spaces(sections["observation"], "", default_name="obs"))),
        Dict(dict(_iter_leaf_spaces(sections["action"], "", default_name="action"))),
    )


def _iter_leaf_spaces(
    node: Mapping[str, Any], path: str, default_name: str
) -> Iterator[tuple[str, gym.Space]]:
    """Yield ``(dotted path, space)`` for every leaf below ``node``."""
    space_type = node.get("type")
    if isinstance(space_type, str):
        leaf_name = path or default_name
        yield leaf_name, _leaf_to_space(node, leaf_name)
        return

    node_name = path or default_name
    scalar_leaf_fields = {
        key
        for key in ("type", "shape", "dtype")
        if key in node and not isinstance(node[key], Mapping)
    }
    if scalar_leaf_fields:
        raise ValueError(
            f"{node_name}: expected a leaf node with a string 'type' and its "
            f"required fields, or an intermediate node containing only child "
            f"mappings; got fields {sorted(node)}"
        )

    for key, child in node.items():
        child_path = f"{path}.{key}" if path else key
        if not isinstance(child, Mapping):
            raise TypeError(
                f"{child_path}: expected a child mapping; got "
                f"{type(child).__name__}"
            )
        yield from _iter_leaf_spaces(child, child_path, default_name)


def _leaf_to_space(leaf: Mapping[str, Any], path: str) -> gym.Space:
    """Build one Gym space from a ``type``/``shape``(/``dtype``) leaf."""
    space_type = leaf["type"]
    required_fields = {
        "box": {"type", "shape", "dtype"},
        "discrete": {"type", "shape"},
        "multi_discrete": {"type", "shape"},
        "multi_binary": {"type", "shape"},
    }
    if space_type not in required_fields:
        raise ValueError(f"{path}: unsupported space type {space_type!r}")

    expected = required_fields[space_type]
    actual = set(leaf)
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unexpected:
            details.append(f"unexpected {sorted(unexpected)}")
        raise ValueError(
            f"{path}: expected {space_type!r} leaf fields {sorted(expected)}; "
            f"got {sorted(actual)} ({'; '.join(details)})"
        )

    shape = tuple(leaf["shape"])
    match space_type:
        case "box":
            dtype = np.dtype(leaf["dtype"])
            # Bounds only need to keep sampling valid; ONNX ignores them.
            low, high = (
                (np.iinfo(dtype).min, np.iinfo(dtype).max)
                if np.issubdtype(dtype, np.integer)
                else (-np.inf, np.inf)
            )
            return Box(low=low, high=high, shape=shape, dtype=dtype.type)
        case "discrete":
            return Discrete(shape[0])
        case "multi_discrete":
            return MultiDiscrete(list(shape))
        case "multi_binary":
            return MultiBinary(list(shape))
        case _:
            raise AssertionError(f"unhandled validated space type {space_type!r}")
