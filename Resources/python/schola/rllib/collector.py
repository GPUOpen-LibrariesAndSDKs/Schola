# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Record a Schola imitation session as RLlib ``SingleAgentEpisode`` objects."""

from __future__ import annotations

import logging
from typing import Any, SupportsFloat

import gymnasium as gym

from schola.core.error_manager import (
    NoServerError,
    ScholaErrorContextManager,
    UnrealCrashedError,
)
from schola.core.protocols.base_protocol import BaseImitationProtocol
from schola.core.simulators.base_simulator import BaseSimulator
from schola.core.utils.id_manager import IdManager
from schola.rllib.offline import build_rllib_episode

from ray.rllib.env.single_agent_episode import SingleAgentEpisode

logger = logging.getLogger(__name__)


class _OpenEpisode:
    """Buffers one unfinished episode as per-timestep samples."""

    def __init__(self, initial_observation: Any) -> None:
        self.observations: list[Any] = [initial_observation]
        self.actions: list[Any] = []
        self.rewards: list[SupportsFloat] = []
        self.terminated = False
        self.truncated = False

    def add_step(
        self,
        observation: Any,
        action: Any,
        reward: SupportsFloat,
        *,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminated = terminated
        self.truncated = truncated

    @property
    def has_steps(self) -> bool:
        return len(self.actions) > 0


class RllibImitationCollector:
    """
    Record a single-environment, single-agent imitation session for RLlib.

    Notes
    -----
    Requires exactly one environment and one agent in the Schola definition.
    """

    def __init__(
        self,
        protocol: BaseImitationProtocol,
        simulator: BaseSimulator,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.protocol = protocol
        self.simulator = simulator
        self.episodes: list[SingleAgentEpisode] = []
        self._open_episode: _OpenEpisode | None = None
        self._saw_step = False

        self.protocol.start()
        self.simulator.start(self.protocol.properties)

        ids, _, observation_spaces, action_spaces = self.protocol.get_definition()
        id_manager = IdManager(ids)
        self._env_id, self._agent_id = id_manager[0]

        if id_manager.num_ids != 1:
            raise ValueError(
                "RLlib imitation collection only supports one environment and one agent"
            )

        self.protocol.send_startup_msg(seeds=[seed], options=[options])

        self.observation_space: gym.Space[Any] = observation_spaces[self._env_id][
            self._agent_id
        ]
        self.action_space: gym.Space[Any] = action_spaces[self._env_id][self._agent_id]

    def collect_until_closed(self, max_steps: int | None = None) -> list[SingleAgentEpisode]:
        """
        Read imitation steps until the simulator or gRPC session ends.

        Parameters
        ----------
        max_steps : int or None
            Optional safety cap on recorded environment steps. ``None`` means
            wait for the process or stream to close.

        Returns
        -------
        list of SingleAgentEpisode
            Completed episodes, including a truncated tail if the session
            ended mid-episode after at least one action.
        """
        steps = 0
        try:
            with ScholaErrorContextManager():
                while True:
                    if max_steps is not None and steps >= max_steps:
                        logger.info(
                            "Reached max-steps cap (%s); ending collection.", max_steps
                        )
                        break
                    if not self.simulator:
                        logger.info("Simulator process ended; finishing collection.")
                        break
                    self._step()
                    steps += 1
        except UnrealCrashedError:
            logger.info("Simulator session ended; finishing collection.")
        except NoServerError:
            if not self._saw_step:
                raise
            logger.info("Simulator connection closed; finishing collection.")
        self._finalize_open_episode()
        logger.info(
            "Collected %s episodes (%s steps).",
            len(self.episodes),
            sum(len(episode) for episode in self.episodes),
        )
        return self.episodes

    def _lookup(self, table: Any) -> Any:
        return table[self._env_id][self._agent_id]

    def _step(self) -> None:
        (
            observations,
            rewards,
            terminations,
            truncations,
            _infos,
            initial_obs,
            _initial_infos,
            actions,
        ) = self.protocol.get_data()
        self._saw_step = True

        if self._open_episode is None:
            if self._env_id in initial_obs and self._agent_id in initial_obs[self._env_id]:
                self._open_episode = _OpenEpisode(
                    self._lookup(initial_obs)
                )
            else:
                raise RuntimeError(
                    "Imitation session sent a step before an initial observation."
                )

        terminated = bool(self._lookup(terminations))
        truncated = bool(self._lookup(truncations))
        self._open_episode.add_step(
            self._lookup(observations),
            self._lookup(actions),
            float(self._lookup(rewards)),
            terminated=terminated,
            truncated=truncated,
        )
        if terminated or truncated:
            self._commit_open_episode()
            if self._env_id in initial_obs and self._agent_id in initial_obs[self._env_id]:
                self._open_episode = _OpenEpisode(self._lookup(initial_obs))

    def _commit_open_episode(self) -> None:
        episode_buffer = self._open_episode
        self._open_episode = None
        if episode_buffer is None or not episode_buffer.has_steps:
            return
        self.episodes.append(
            build_rllib_episode(
                episode_buffer.observations,
                episode_buffer.actions,
                episode_buffer.rewards,
                self.observation_space,
                self.action_space,
                terminated=episode_buffer.terminated,
                truncated=episode_buffer.truncated,
            )
        )

    def _finalize_open_episode(self) -> None:
        if self._open_episode is None or not self._open_episode.has_steps:
            self._open_episode = None
            return
        self._open_episode.truncated = True
        self._open_episode.terminated = False
        self._commit_open_episode()

    def close(self) -> None:
        self.protocol.close()
        self.simulator.stop()
