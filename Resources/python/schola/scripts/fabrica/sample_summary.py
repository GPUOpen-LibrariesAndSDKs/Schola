# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Fabrica per-sample aggregate passed between iterations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from langchain_core.messages import BaseMessage

from schola.scripts.fabrica.episode_metrics_callback import FabricaEpisodeMetrics
from schola.scripts.fabrica.codegen import FabricaCodegenData


@dataclass
class FabricaSampleSummary:
    sample_index: int
    iteration_index: int

    response: Optional[FabricaCodegenData]
    metrics: Optional[FabricaEpisodeMetrics]
    messages: Optional[List[BaseMessage]]

    def to_string(self, episode_freq: int = 1) -> str:
        """Policy feedback text passed to the next Fabrica iteration."""
        if self.metrics is None:
            return "(No SB3 training metrics; policy feedback unavailable.)"
        if not self.metrics.episodes:
            return "(No completed episodes; policy feedback unavailable.)"
        return self.metrics.to_string(episode_freq)
