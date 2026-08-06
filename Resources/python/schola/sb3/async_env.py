# Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Async VecEnv implementation using AsyncBaseRLProtocol and a dedicated event-loop thread.
Supports multiple (simulator, protocol) pairs on one event loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Mapping, Sequence
import logging
import sys
import time
from collections import defaultdict
from concurrent.futures import Future
from copy import deepcopy
from threading import Thread
from typing import Any, TypeVar, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs

from schola.core.simulators.base_simulator import (
    BaseSimulator,
    UnsupportedProtocolException,
)
from schola.core.protocols.base_protocol import AutoResetType
from schola.core.utils.id_manager import IdManager, NestedDict

from .env import BaseVecEnv, _validate_definition
from .utils import split_value

from schola.core.protocols.async_base_protocol import AsyncBaseRLProtocol

logger = logging.getLogger(__name__)
T = TypeVar("T")


def _merge_async_definitions(
    results: list[
        tuple[
            list[list[str]],
            list[dict[str, str]],
            dict[int, dict[str, gym.Space]],
            dict[int, dict[str, gym.Space]],
        ]
    ],
) -> tuple[
    IdManager,
    list[dict[str, str]],
    gym.Space,
    gym.Space,
    list[IdManager],
    list[int],
    list[int],
]:
    """
    Merge per-protocol definitions into global env indices.

    Returns
    -------
    id_manager, agent_types, obs_space, action_space,
    segment_id_managers, segment_flat_sizes, segment_env_bases
    """
    merged_ids: list[list[str]] = []
    merged_agent_types: list[dict[str, str]] = []
    merged_obs: dict[int, dict[str, gym.Space]] = {}
    merged_act: dict[int, dict[str, gym.Space]] = {}
    segment_id_managers: list[IdManager] = []
    segment_flat_sizes: list[int] = []
    segment_env_bases: list[int] = []
    env_offset = 0
    for ids, agent_types, obs_defns, action_defns in results:
        segment_env_bases.append(env_offset)
        merged_ids.extend(ids)
        merged_agent_types.extend(agent_types)
        for eid, od in obs_defns.items():
            merged_obs[env_offset + eid] = od
        for eid, ad in action_defns.items():
            merged_act[env_offset + eid] = ad
        seg_im = IdManager(ids)
        segment_id_managers.append(seg_im)
        segment_flat_sizes.append(seg_im.num_ids)
        env_offset += len(ids)
    id_manager = IdManager(merged_ids)
    obs_space, action_space = _validate_definition(id_manager, merged_obs, merged_act)
    return (
        id_manager,
        merged_agent_types,
        obs_space,
        action_space,
        segment_id_managers,
        segment_flat_sizes,
        segment_env_bases,
    )


def _merge_step_results(
    segment_results: list[
        tuple[
            list[dict[str, Any]],
            list[dict[str, Any]],
            list[dict[str, bool]],
            list[dict[str, bool]],
            list[dict[str, dict[str, str]]],
            dict[int, dict[str, Any]],
            dict[int, dict[str, dict[str, str]]],
        ]
    ],
    segment_env_bases: list[int],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, bool]],
    list[dict[str, bool]],
    list[dict[str, dict[str, str]]],
    dict[int, dict[str, Any]],
    dict[int, dict[str, dict[str, str]]],
]:
    merged_obs: list[dict[str, Any]] = []
    merged_rew: list[dict[str, Any]] = []
    merged_term: list[dict[str, bool]] = []
    merged_trunc: list[dict[str, bool]] = []
    merged_infos: list[dict[str, dict[str, str]]] = []
    merged_init_obs: dict[int, dict[str, Any]] = {}
    merged_init_infos: dict[int, dict[str, dict[str, str]]] = {}
    for i, res in enumerate(segment_results):
        obs, rew, term, trunc, infos, init_o, init_i = res
        merged_obs.extend(obs)
        merged_rew.extend(rew)
        merged_term.extend(term)
        merged_trunc.extend(trunc)
        merged_infos.extend(infos)
        base = segment_env_bases[i]
        for local_eid, agents in init_o.items():
            merged_init_obs[base + local_eid] = agents
        for local_eid, agents in init_i.items():
            merged_init_infos[base + local_eid] = agents
    return (
        merged_obs,
        merged_rew,
        merged_term,
        merged_trunc,
        merged_infos,
        merged_init_obs,
        merged_init_infos,
    )


def is_iterable(obj: Any) -> bool:
    """
    Return whether ``iter(obj)`` succeeds.

    Parameters
    ----------
    obj : object
        Candidate value.

    Returns
    -------
    bool
        ``True`` if ``iter`` does not raise ``TypeError``; otherwise ``False``.
    """
    try:
        iter(obj)
        return True
    except TypeError:
        return False


class AsyncVecEnv(BaseVecEnv):
    """
    Stable-Baselines3 vectorized environment using async protocols (AsyncBaseRLProtocol).

    Uses a dedicated background thread with a long-lived event loop. Multiple
    (simulator, protocol) pairs share that loop; ``step_async`` schedules all
    ``send_action_msg`` calls concurrently and ``step_wait`` blocks until all complete.

    Pass either a single ``(simulator, protocol)`` pair or equal-length sequences
    of simulators and protocols.
    """

    def __init__(
        self,
        simulator: BaseSimulator | Sequence[BaseSimulator],
        protocol: AsyncBaseRLProtocol | Sequence[AsyncBaseRLProtocol],
        verbosity: int = 0,
    ):
        if sys.version_info < (3, 11):
            raise RuntimeError(
                "AsyncVecEnv requires Python 3.11 or later (uses asyncio.TaskGroup)."
            )
        _simulator = simulator if isinstance(simulator, Sequence) else [simulator]
        _protocol = protocol if isinstance(protocol, Sequence) else [protocol]

        self.simulators = list(_simulator)
        self.protocols = list(_protocol)

        if len(self.simulators) != len(self.protocols):
            raise ValueError(
                "simulators and protocols must have the same length "
                f"({len(self.simulators)} vs {len(self.protocols)})."
            )
        if not self.simulators:
            raise ValueError("At least one (simulator, protocol) pair is required.")

        for sim, proto in zip(self.simulators, self.protocols):
            if not isinstance(proto, sim.supported_async_protocols):
                raise UnsupportedProtocolException(
                    f"Protocol {proto} is not supported by the simulator {sim}."
                )

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: Thread | None = None
        self._step_future: Future[Any] | None = None
        self._segment_id_managers: list[IdManager] = []
        self._segment_flat_sizes: list[int] = []
        self._segment_env_bases: list[int] = []

        def _run_loop() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._thread = Thread(target=_run_loop, daemon=True)
        self._thread.start()
        while self._loop is None:
            time.sleep(0)

        raw = self._run(self._async_init())
        started_sims = []
        started_protocols = []
        for i, value in enumerate(raw):
            if isinstance(value, Exception):
                logger.error("... Error starting simulator %d: %s", i, value)
            else:
                started_sims.append(self.simulators[i])
                started_protocols.append(self.protocols[i])
        self.simulators = started_sims
        self.protocols = started_protocols
        no_exceptions_raw = filter(lambda x: not isinstance(x, Exception), raw)
        (
            id_manager,
            agent_types,
            obs_space,
            action_space,
            self._segment_id_managers,
            self._segment_flat_sizes,
            self._segment_env_bases,
        ) = _merge_async_definitions(list(no_exceptions_raw))

        super().__init__(id_manager, agent_types, obs_space, action_space)

    async def _async_init(self):
        async with asyncio.TaskGroup() as tg:
            logger.info("...Starting Protocols and Simulators")
            tasks = []
            for sim, proto in zip(self.simulators, self.protocols):
                tasks.append(
                    tg.create_task(self._start_protocol_and_simulator(sim, proto))
                )

        return [task.result() for task in tasks]

    async def _start_protocol_and_simulator(
        self, sim: BaseSimulator, protocol: AsyncBaseRLProtocol
    ) -> Any | Exception:
        try:
            await protocol.start()
            sim.start(protocol.properties)
            await protocol.send_startup_msg(auto_reset_type=AutoResetType.SAME_STEP)
            definition = await protocol.get_definition()
            return definition
        except Exception as e:
            await protocol.close()
            sim.stop()
            return e

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        assert self._loop is not None
        return self._loop

    def _run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run a coroutine on the dedicated event loop and return the result."""
        future: Future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

    def close(self) -> None:
        logger.info("... closing environment")

        async def _close_all() -> None:
            await asyncio.gather(
                *(p.close() for p in self.protocols), return_exceptions=True
            )

        self._run(_close_all())
        if self._loop is not None:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        for sim in self.simulators:
            sim.stop()

    def seed(self, seed: int | None = None) -> list[int]:
        if seed is None:
            seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        seeds = [int(seed + i) for i in range(self.num_envs)]
        self._seeds = seeds  # type: ignore
        return seeds

    def set_options(
        self, options: list[dict[str, Any]] | dict[str, Any] | None = None
    ) -> None:
        if options is None:
            options = {}
        if isinstance(options, dict):
            self._options = deepcopy([options] * self.num_envs)
            return
        if len(options) != self.num_envs:
            raise ValueError(
                f"Expected options list length {self.num_envs}, got {len(options)}."
            )
        self._options = deepcopy(options)

    def reset(self) -> VecEnvObs:
        async def _do_reset() -> (
            tuple[list[dict[str, Any]], list[dict[str, dict[str, str]]]]
        ):
            obs_all: list[Any] = []
            info_all: list[Any] = []
            off = 0
            for proto, nflat in zip(self.protocols, self._segment_flat_sizes):
                seeds = self._seeds[off : off + nflat] if self._seeds else None
                opts = self._options[off : off + nflat] if self._options else None
                o, inf = await proto.send_reset_msg(seeds=seeds, options=opts)
                obs_all.extend(o)
                info_all.extend(inf)
                off += nflat
            return obs_all, info_all

        obs, nested_infos = self._run(_do_reset())
        self._reset_seeds()
        self._reset_options()
        return self._process_reset(obs, nested_infos)

    def step_async(self, actions: np.ndarray) -> None:
        off = 0
        coros = []
        for proto, seg_im in zip(self.protocols, self._segment_id_managers):
            n = seg_im.num_ids
            seg_actions = actions[off : off + n]
            next_actions = seg_im.nest_list_to_dict_of_dicts(seg_actions)  # type: ignore[arg-type]
            if isinstance(self.action_space, gym.spaces.Dict):
                for env_id, agent_id_list in enumerate(seg_im.ids):
                    for agent_id in agent_id_list:
                        next_actions[env_id][agent_id] = split_value(
                            next_actions[env_id][agent_id], self.action_space
                        )  # type: ignore[arg-type]

            next_actions = cast(dict[int, dict[str, Any]], next_actions)
            coros.append(
                proto.send_action_msg(
                    next_actions, defaultdict(lambda: self.action_space)
                )
            )
            off += n

        async def _all_steps():
            return await asyncio.gather(*coros)

        self._step_future = asyncio.run_coroutine_threadsafe(_all_steps(), self.loop)

    def step_wait(
        self,
    ) -> tuple[VecEnvObs, np.ndarray, np.ndarray, list[dict[str, str]]]:
        assert self._step_future is not None
        segment_results = self._step_future.result()
        merged = _merge_step_results(segment_results, self._segment_env_bases)
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            nested_infos,
            initial_obs,
            initial_infos,
        ) = merged
        return self._process_step_wait(
            observations,
            rewards,
            terminateds,
            truncateds,
            nested_infos,
            initial_obs,
            initial_infos,
        )
