# Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Implementation of stable_baselines3.common.vec_env.VecEnv backed by a Schola Environment.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv as Sb3VecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import _stack_obs

from schola.core.error_manager import (
    EnvironmentException,
    NoAgentsException,
    NoEnvironmentsException,
)
from schola.core.protocols.base_protocol import BaseRLProtocol
from schola.core.simulators.base_simulator import (
    BaseSimulator,
    UnsupportedProtocolException,
)
from schola.core.utils.id_manager import IdManager
from schola.generated.GymConnector_pb2 import AutoResetType
from schola.sb3.utils import split_value

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _validate_definition(
    id_manager: IdManager,
    obs_defns: Dict[int, Dict[str, gym.Space]],
    action_defns: Dict[int, Dict[str, gym.Space]],
) -> Tuple[gym.Space, gym.Space]:
    """
    Validate environment definition and return unified observation and action spaces.

    Parameters
    ----------
    id_manager : IdManager
        Manager built from ``protocol.get_definition()`` agent id layout.
    obs_defns : Dict[int, Dict[str, gym.Space]]
        Observation spaces keyed by environment id and agent id.
    action_defns : Dict[int, Dict[str, gym.Space]]
        Action spaces keyed by environment id and agent id.

    Returns
    -------
    Tuple[gym.Space, gym.Space]
        ``(obs_space, action_space)`` shared by every agent slot.

    Raises
    ------
    NoEnvironmentsException
        If there are no environments.
    NoAgentsException
        If any environment lists zero agents.
    AssertionError
        If any agent's spaces differ from the reference agent.
    """
    ids = id_manager.ids
    if len(ids) == 0:
        raise NoEnvironmentsException()

    first_env_id, first_agent_id = id_manager[0]
    action_space = action_defns[first_env_id][first_agent_id]
    obs_space = obs_defns[first_env_id][first_agent_id]

    for env_id, agent_id_list in enumerate(id_manager.ids):
        if len(agent_id_list) == 0:
            raise NoAgentsException(env_id)

    for env_id, agent_id in id_manager.id_list:
        assert action_defns[env_id][agent_id] == action_space, (
            f"Action Space Mismatch on Agent:{agent_id} in Env {env_id}.\n"
            f"Got: {action_defns[env_id][agent_id]}\nExpected:{action_space}"
        )
        assert obs_defns[env_id][agent_id] == obs_space, (
            f"Observation Space Mismatch on Agent:{agent_id} in Env {env_id}.\n"
            f"Got: {obs_defns[env_id][agent_id]}\nExpected:{obs_space}"
        )
    return obs_space, action_space


class BaseVecEnv(Sb3VecEnv):
    """
    Base vectorized environment for Schola, shared by sync and async implementations.

    Holds common state (id_manager, agent_types), step/reset processing, and VecEnv
    methods that do not depend on protocol sync vs async. Subclasses own simulator
    lifecycle and must implement ``close()``.

    Parameters
    ----------
    id_manager : IdManager
        Flattened agent indexing for every (environment, agent) pair.
    agent_types : List[Dict[str, str]]
        Per-environment agent type metadata from Unreal.
    obs_space : gym.Space
        Stacked observation space passed to SB3's ``VecEnv`` constructor.
    action_space : gym.Space
        Stacked action space passed to SB3's ``VecEnv`` constructor.
    """

    def __init__(
        self,
        id_manager: IdManager,
        agent_types: List[Dict[str, str]],
        obs_space: gym.Space,
        action_space: gym.Space,
    ):
        self.id_manager = id_manager
        self.agent_types = agent_types
        self.next_actions: Optional[Dict[int, Dict[str, Any]]] = None
        super().__init__(id_manager.num_ids, obs_space, action_space)

    def _process_reset(
        self,
        obs: List[Dict[str, Any]],
        nested_infos: List[Dict[str, Dict[str, str]]],
    ) -> Dict[str, np.ndarray]:
        """
        Stack reset observations and copy per-agent info into ``reset_infos``.

        Parameters
        ----------
        obs : list of dict
            One observation dict per environment from the protocol.
        nested_infos : list of dict
            Nested ``info`` payloads per environment and agent.

        Returns
        -------
        Dict[str, numpy.ndarray]
            Stacked observation batch in SB3 dict-of-arrays layout.
        """
        for env_id, info_dict in enumerate(nested_infos):
            for agent_id in nested_infos[env_id]:
                uid = self.id_manager[env_id, agent_id]
                self.reset_infos[uid] = nested_infos[env_id][agent_id]
        flat_obs = self.id_manager.flatten_list_of_dicts(obs)
        return _stack_obs(flat_obs, self.observation_space)

    def _process_step_wait(
        self,
        observations: List[Dict[str, Any]],
        rewards: List[Dict[str, Any]],
        terminateds: List[Dict[str, bool]],
        truncateds: List[Dict[str, bool]],
        nested_infos: List[Dict[str, Dict[str, str]]],
        initial_obs: Dict[int, Dict[str, Any]],
        initial_infos: Dict[int, Dict[str, str]],
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[Dict[str, str]]]:
        """
        Convert a protocol step payload into SB3 ``step_wait`` return values.

        Parameters
        ----------
        observations : list of dict
            Per-environment, per-agent observations after the step.
        rewards : list of dict
            Per-environment, per-agent rewards.
        terminateds : list of dict
            Per-environment, per-agent episode termination flags.
        truncateds : list of dict
            Per-environment, per-agent truncation flags.
        nested_infos : list of dict
            Per-environment, per-agent info dicts.
        initial_obs : dict
            Auto-reset initial observations keyed by environment id.
        initial_infos : dict
            Auto-reset info dicts keyed by environment id.

        Returns
        -------
        Tuple[Dict[str, numpy.ndarray], numpy.ndarray, numpy.ndarray, list of dict]
            ``(stacked_obs, rewards, dones, infos)`` where ``dones`` combines
            termination and truncation per flattened agent index.

        Raises
        ------
        EnvironmentException
            If agents in one Unreal environment finish on different timesteps.
        """
        array_dones = np.empty((self.id_manager.num_ids,), dtype=np.bool_)
        array_rewards = np.asarray(self.id_manager.flatten_list_of_dicts(rewards))
        array_observations = self.id_manager.flatten_list_of_dicts(observations)

        infos = [{} for _ in range(self.num_envs)]
        for env_id, single_env_info in enumerate(nested_infos):
            for agent_id in single_env_info:
                uid = self.id_manager[env_id, agent_id]
                infos[uid] = single_env_info[agent_id]

        for env_id, agent_id_list in enumerate(self.id_manager.ids):
            any_done = False
            all_done = True
            for agent_id in agent_id_list:
                uid = self.id_manager[env_id, agent_id]
                array_dones[uid] = terminateds[env_id].get(
                    agent_id, False
                ) or truncateds[env_id].get(agent_id, False)
                any_done = any_done or array_dones[uid]
                all_done = all_done and array_dones[uid]
            if any_done and not all_done:
                raise EnvironmentException(
                    f"SB3 with multi-agent environments does not support agents completing at different steps. "
                    f"Env {env_id} had agents in different completion states."
                )

        for env_id in initial_infos:
            for agent_id in initial_infos[env_id]:
                uid = self.id_manager[env_id, agent_id]
                self.reset_infos[uid] = initial_infos[env_id][agent_id]

        for env_id in initial_obs:
            for agent_id in self.id_manager.partial_get(env_id):
                uid = self.id_manager[env_id, agent_id]
                infos[uid]["terminal_observation"] = observations[env_id][agent_id]
                infos[uid]["TimeLimit.truncated"] = (
                    truncateds[env_id][agent_id] and not terminateds[env_id][agent_id]
                )
                array_observations[uid] = initial_obs[env_id][agent_id]

        return (
            _stack_obs(array_observations, self.observation_space),
            array_rewards,
            array_dones,
            infos,
        )

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError(
            "env_method is not implemented for Schola environments, as sub-environments are not individually accessible."
        )

    def get_attr(self, attr_name, indices=None):
        return [None for _ in range(self.id_manager.num_ids)]

    def env_is_wrapped(
        self, wrapper_class, indices: Optional[Iterable[int]] = None
    ) -> List[bool]:
        if indices is None:
            indices = range(self.id_manager.num_ids)
        return [False for _ in indices]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError(
            "set_attr is not implemented for Schola environments, as sub-environments are not individually accessible."
        )


class VecEnv(BaseVecEnv):
    """
    Stable-Baselines3 vectorized environment implementation for Schola (synchronous protocol).

    This class wraps Schola environments to be compatible with Stable-Baselines3's
    VecEnv interface, enabling use with SB3 algorithms.

    Parameters
    ----------
    simulator : BaseSimulator
        The simulator instance managing the simulator lifecycle.
    protocol : BaseRLProtocol
        The protocol instance for communication with the Simulator.
    verbosity : int, default=0
        The verbosity level for logging.
    """

    def __init__(
        self, simulator: BaseSimulator, protocol: BaseRLProtocol, verbosity: int = 0
    ):
        self.simulator = simulator
        self.protocol = protocol

        if not isinstance(protocol, simulator.supported_protocols):
            raise UnsupportedProtocolException(
                f"Protocol {protocol} is not supported by the simulator {simulator}."
            )

        logger.info("...Starting Protocol and Simulator")
        protocol.start()
        simulator.start(protocol.properties)

        logger.info("...Sending Startup Message")
        try:
            protocol.send_startup_msg(auto_reset_type=AutoResetType.SAME_STEP)
        except Exception:
            protocol.close()
            simulator.stop()
            raise

        logger.info("...Requesting environment definition")
        id_manager, agent_types, obs_space, action_space = self._define_environment(
            protocol
        )

        super().__init__(id_manager, agent_types, obs_space, action_space)

    def _define_environment(
        self, protocol: BaseRLProtocol
    ) -> Tuple[IdManager, List[Dict[str, str]], gym.Space, gym.Space]:
        """
        Fetch and validate the Schola definition for SB3 stacking.

        Parameters
        ----------
        protocol : BaseRLProtocol
            Active protocol used for ``get_definition``.

        Returns
        -------
        Tuple[IdManager, List[Dict[str, str]], gym.Space, gym.Space]
            ``(id_manager, agent_types, obs_space, action_space)`` after validation.

        Raises
        ------
        NoEnvironmentsException, NoAgentsException, AssertionError
            Propagated from :func:`_validate_definition` or ``IdManager`` construction.
        """
        ids, agent_types, obs_defns, action_defns = protocol.get_definition()
        id_manager = IdManager(ids)
        try:
            obs_space, action_space = _validate_definition(
                id_manager, obs_defns, action_defns
            )
        except Exception as e:
            protocol.close()
            self.simulator.stop()
            raise e
        return id_manager, agent_types, obs_space, action_space

    def _close_protocol(self) -> None:
        """
        Close only the protocol handle.
        """
        self.protocol.close()

    def close(self) -> None:
        logger.info("... closing environment")
        self._close_protocol()
        self.simulator.stop()

    def reset(self) -> Dict[str, np.ndarray]:
        obs, nested_infos = self.protocol.send_reset_msg(
            seeds=self._seeds, options=self._options
        )
        self._reset_seeds()
        self._reset_options()
        return self._process_reset(obs, nested_infos)

    def step_async(self, actions: np.ndarray) -> None:
        """
        Buffer flattened actions for the next :meth:`step_wait` call.

        Parameters
        ----------
        actions : numpy.ndarray
            SB3-flattened actions reshaped into nested env/agent dicts; dict action
            spaces are split with :func:`~schola.sb3.utils.split_value`.

        Returns
        -------
        None
            Sets :attr:`next_actions` in place.
        """
        self.next_actions = self.id_manager.nest_list_to_dict_of_dicts(actions)
        if isinstance(self.action_space, gym.spaces.Dict):
            for env_id, agent_id_list in enumerate(self.id_manager.ids):
                for agent_id in agent_id_list:
                    self.next_actions[env_id][agent_id] = split_value(
                        self.next_actions[env_id][agent_id], self.action_space
                    )

    def step_wait(
        self,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[Dict[str, str]]]:
        assert (
            self.next_actions is not None
        ), "step_async must be called before step_wait"
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            nested_infos,
            initial_obs,
            initial_infos,
        ) = self.protocol.send_action_msg(
            self.next_actions, defaultdict(lambda: self.action_space)
        )
        return self._process_step_wait(
            observations,
            rewards,
            terminateds,
            truncateds,
            nested_infos,
            initial_obs,
            initial_infos,
        )
