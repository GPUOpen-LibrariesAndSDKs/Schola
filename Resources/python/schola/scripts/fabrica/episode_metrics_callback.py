# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Accumulate per-episode returns, lengths, reward-component Info keys, and one task-success scalar."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, DefaultDict, Dict, List, Mapping, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def _safe_float(value: Any) -> float:
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return 0.0
    return 0.0


@dataclass
class FabricaEpisodeRow:
    """One completed episode (one vectorized env index)."""

    env_index: int
    episode_return: float
    episode_length: int
    task_success: float
    reward_components: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Shape aligned with aggregate ``episode_metrics.json`` keys."""
        return {
            "env_index": self.env_index,
            "episode_return": self.episode_return,
            "episode_length": self.episode_length,
            "reward_components": dict(self.reward_components),
            "task_success": self.task_success,
        }


@dataclass
class FabricaEpisodeAggregate:
    """Aggregate over all completed episodes in a run."""

    episode_return: Optional[float]
    episode_length: Optional[float]
    task_success: Optional[float]
    num_episodes: Optional[int]
    reward_components: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_episodes": self.num_episodes,
            "episode_return": self.episode_return,
            "episode_length": self.episode_length,
            "reward_components": dict(self.reward_components),
            "task_success": self.task_success,
        }


@dataclass
class FabricaEpisodeMetrics:
    """
    All episode rows collected during training.

    :class:`EpisodeMetricsCallback` appends a :class:`FabricaEpisodeRow` each time an
    environment signals ``done``.
    """

    episodes: List[FabricaEpisodeRow] = field(default_factory=list)

    def append_row(self, row: FabricaEpisodeRow) -> None:
        self.episodes.append(row)

    def mean(self):
        return self.aggregate(lambda x: float(np.mean(x)))

    def max(self):
        return self.aggregate(lambda x: float(np.max(x)))

    def min(self):
        return self.aggregate(lambda x: float(np.min(x)))

    def aggregate(
        self, agg_func: Callable[[List[float]], float]
    ) -> FabricaEpisodeAggregate:
        if not self.episodes:
            return FabricaEpisodeAggregate(
                num_episodes=0,
                episode_return=None,
                episode_length=None,
                reward_components={},
                task_success=None,
            )

        returns = [float(ep.episode_return) for ep in self.episodes]
        lengths = [float(ep.episode_length) for ep in self.episodes]
        ts = [float(ep.task_success) for ep in self.episodes]

        comp_keys: set[str] = set()
        for ep in self.episodes:
            comp_keys.update(ep.reward_components.keys())

        agg_components: Dict[str, float] = {}
        for key in sorted(comp_keys):
            vals = [float(ep.reward_components.get(key, 0.0)) for ep in self.episodes]
            agg_components[key] = agg_func(vals) if vals else 0.0

        return FabricaEpisodeAggregate(
            num_episodes=len(self.episodes),
            episode_return=agg_func(returns),
            episode_length=agg_func(lengths),
            reward_components=agg_components,
            task_success=agg_func(ts),
        )

    def format_row(
        self,
        metric_key: str,
        raw_value: List[float],
        max_value: float,
        mean_value: float,
        min_value: float,
    ) -> str:
        # format so max of 2 decimal places
        raw_value_strs = [f"{x:.2f}" for x in raw_value]
        return f"{metric_key}: {raw_value_strs}, Max: {max_value:.2f}, Mean: {mean_value:.2f}, Min: {min_value:.2f}"

    def make_str_row(
        self,
        label: str,
        metric_key: str,
        mean_agg: FabricaEpisodeAggregate,
        max_agg: FabricaEpisodeAggregate,
        min_agg: FabricaEpisodeAggregate,
        episode_freq: int = 1,
    ) -> str:
        raw_value = [
            float(x.to_dict()[metric_key]) for x in self.episodes[::episode_freq]
        ]
        max_value = max_agg.to_dict()[metric_key]
        mean_value = mean_agg.to_dict()[metric_key]
        min_value = min_agg.to_dict()[metric_key]
        return self.format_row(label, raw_value, max_value, mean_value, min_value)

    def make_str_reward_components(
        self,
        metric_key: str,
        mean_agg: FabricaEpisodeAggregate,
        max_agg: FabricaEpisodeAggregate,
        min_agg: FabricaEpisodeAggregate,
        episode_freq: int = 1,
    ) -> str:
        raw_value = [
            float(x.to_dict()["reward_components"].get(metric_key, 0.0))
            for x in self.episodes[::episode_freq]
        ]
        max_value = max_agg.to_dict()["reward_components"].get(metric_key, 0.0)
        mean_value = mean_agg.to_dict()["reward_components"].get(metric_key, 0.0)
        min_value = min_agg.to_dict()["reward_components"].get(metric_key, 0.0)

        return self.format_row(metric_key, raw_value, max_value, mean_value, min_value)

    def to_string(self, episode_freq: int = 1) -> str:
        mean_agg = self.mean()
        max_agg = self.max()
        min_agg = self.min()
        str_rows = [
            self.make_str_row(
                "Task Success", "task_success", mean_agg, max_agg, min_agg, episode_freq
            ),
            self.make_str_row(
                "Episode Return",
                "episode_return",
                mean_agg,
                max_agg,
                min_agg,
                episode_freq,
            ),
            self.make_str_row(
                "Episode Length",
                "episode_length",
                mean_agg,
                max_agg,
                min_agg,
                episode_freq,
            ),
        ] + [
            self.make_str_reward_components(
                key, mean_agg, max_agg, min_agg, episode_freq
            )
            for key in mean_agg.reward_components.keys()
        ]

        return "\n".join(str_rows)

    def __str__(self) -> str:
        return self.to_string()


class EpisodeMetricsCallback(BaseCallback):
    """
    Track episodic return and length from step rewards and ``dones``, mirroring Monitor's
    ``r`` / ``l`` while summing reward-component Info keys by prefix and reading a single
    task-success scalar from a dedicated Info key on episode end.

    SB3 passes ``rewards``, ``dones``, and ``infos`` into callback locals each rollout step.
    On each done, one :class:`FabricaEpisodeRow` is appended to ``metrics.episodes``.
    """

    def __init__(
        self,
        reward_component_prefix: str = "fabrica_r:",
        task_success_key: str = "fabrica_ts",
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.reward_component_prefix = reward_component_prefix
        self.task_success_key = task_success_key
        self.metrics = FabricaEpisodeMetrics()
        self._ep_ret: Optional[np.ndarray] = None
        self._ep_len: Optional[np.ndarray] = None
        self._component_acc: Optional[List[DefaultDict[str, float]]] = None

    def _ensure_buffers(self, n_envs: int) -> None:
        if self._ep_ret is not None and self._ep_ret.shape[0] == n_envs:
            return
        self._ep_ret = np.zeros((n_envs,), dtype=np.float64)
        self._ep_len = np.zeros((n_envs,), dtype=np.int64)
        self._component_acc = [defaultdict(float) for _ in range(n_envs)]

    def _task_success_from_info(self, info: Mapping[Any, Any]) -> float:
        if not self.task_success_key:
            return 0.0
        raw = info.get(self.task_success_key)
        if raw is None:
            return 0.0
        return _safe_float(raw)

    def format_reward_comp_key(self, key: str) -> str:
        return key.removeprefix(self.reward_component_prefix).replace("_", " ").title()

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if rewards is None or dones is None or infos is None:
            return True

        n_envs = self.training_env.num_envs
        self._ensure_buffers(n_envs)
        assert (
            self._ep_ret is not None
            and self._ep_len is not None
            and self._component_acc is not None
        )

        rewards_arr = np.asarray(rewards).reshape((n_envs,))
        dones_arr = np.asarray(dones).reshape((n_envs,))

        for i in range(n_envs):
            self._ep_ret[i] += float(rewards_arr[i])
            self._ep_len[i] += 1

            info = infos[i] if i < len(infos) else {}
            if isinstance(info, Mapping):
                for key, raw in info.items():
                    if not isinstance(key, str):
                        continue
                    if key.startswith(self.reward_component_prefix):
                        self._component_acc[i][key] += _safe_float(raw)

            if bool(dones_arr[i]):
                task_success = 0.0
                if isinstance(info, Mapping):
                    task_success = self._task_success_from_info(info)

                components = {
                    self.format_reward_comp_key(k): float(v)
                    for k, v in self._component_acc[i].items()
                }
                self.metrics.append_row(
                    FabricaEpisodeRow(
                        env_index=int(i),
                        episode_return=float(self._ep_ret[i]),
                        episode_length=int(self._ep_len[i]),
                        reward_components=components,
                        task_success=task_success,
                    )
                )
                self._ep_ret[i] = 0.0
                self._ep_len[i] = 0
                self._component_acc[i].clear()

        return True

    def summary(self) -> Dict[str, Any]:
        """Aggregate metrics as a plain dict (e.g. for JSON ``metric.json``)."""
        return self.metrics.mean().to_dict()

    def to_jsonable(self) -> Dict[str, Any]:
        """Full training record: per-episode rows plus aggregate summary."""
        return {
            "episodes": [row.to_dict() for row in self.metrics.episodes],
            "aggregate": self.summary(),
        }
