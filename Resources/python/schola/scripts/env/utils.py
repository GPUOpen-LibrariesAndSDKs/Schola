# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Shared helpers for Schola environment utility commands.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree
import numpy as np
from schola.core.utils.id_manager import IdManager
from schola.gym.env import GymVectorEnv


def _format_value_for_log(value: Any) -> str:
    
    if isinstance(value, np.ndarray):
        flat = value.ravel()
        if flat.size <= 16:
            return np.array2string(value, threshold=16, max_line_width=120)
        preview = np.array2string(
            flat[:8], threshold=8, max_line_width=120, separator=", "
        )
        return (
            f"ndarray shape={value.shape} dtype={value.dtype} "
            f"preview(first 8)={preview} ..."
        )
    if isinstance(value, Mapping):
        parts = [
            f"{key}={_format_value_for_log(nested)}"
            for key, nested in list(value.items())[:8]
        ]
        suffix = " ..." if len(value) > 8 else ""
        return "{" + ", ".join(parts) + suffix + "}"
    return repr(value)


def _iter_single_observations(
    env: GymVectorEnv, observations: Any
) -> list[tuple[int, str, Any]]:
    slots: list[tuple[int, str, Any]] = []
    for flat_id, single_obs in enumerate(
        gym.vector.utils.iterate(env.observation_space, observations)
    ):
        env_id, agent_id = env.id_manager.get_nested_id(flat_id)
        slots.append((env_id, agent_id, single_obs))
    return slots


def inspect_agents(
    env: GymVectorEnv,
    *,
    seed: int | None = None,
    options: dict[str, str] | None = None,
) -> RenderableType:
    """Call ``reset`` once and print per-agent spaces and initial observations."""
    observations, _infos = env.reset(
        seed=seed,
        options=dict(options) if options else None,
    )

    id_manager = env.id_manager
    agent_types = id_manager.agent_types
    obs_by_agent = {
        (env_id, agent_id): single_obs
        for env_id, agent_id, single_obs in _iter_single_observations(env, observations)
    }
    tree = Tree(label="[bold green]Environment definitions:[/bold green]", highlight=True)
    for env_id, agent_ids in enumerate(id_manager.ids):
        env_node = tree.add(f"[bold]Environment {env_id}[/bold]:")
        env_agent_types = agent_types.get(env_id, {})
        for agent_id in agent_ids:
            agent_type = env_agent_types.get(agent_id, "")
            if agent_type:
                agent_node = env_node.add(f"[bold]{agent_id}[/bold] (type: [bold]{agent_type}[/bold]):")
            else:
                agent_node = env_node.add(f"[bold]{agent_id}[/bold]:")

            single_obs = obs_by_agent.get((env_id, agent_id))
            agent_node.add(f"[bold]Observation Space:[/bold] {env.single_observation_space}")
            agent_node.add(f"[bold]Action Space:[/bold] {env.single_action_space}")
            agent_node.add(f"[bold]Initial Obs:[/bold] {_format_value_for_log(single_obs)}")
            in_space = (
                env.single_observation_space.contains(single_obs)
                if single_obs is not None
                else False
            )
            agent_node.add(f"[bold]Initial Obs in Space:[/bold] {in_space}")
    return tree


def run_gym_env_checker(env: gym.Env, *, console: Console) -> None:
    """Run Gymnasium's ``check_env`` on a single-agent ``gym.Env``."""
    console.print("Running Gymnasium environment checker...")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=r".*[Ii]nfinit.*",
        )
        check_env(env, skip_render_check=True)
    console.print("Environment checker passed.")
    