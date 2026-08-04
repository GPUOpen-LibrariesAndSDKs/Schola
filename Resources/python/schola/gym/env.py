# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Implementation of gym.vector.VectorEnv backed by a Schola Environment.
"""

from collections import defaultdict
from typing import Any, Dict, List, Mapping, SupportsFloat, Tuple, TypeVar, cast

from schola.core.protocols.base_protocol import AutoResetType, BaseRLProtocol
from schola.core.protocols.protobuf.grpc_protocol import coerce_auto_reset_type
from schola.core.simulators.base_simulator import (
    BaseSimulator,
    UnsupportedProtocolException,
)
from schola.core.error_manager import (
    EnvironmentException,
    MultipleAgentsException,
    MultipleEnvironmentsException,
    NoAgentsException,
    NoEnvironmentsException,
)
import numpy as np
import gymnasium as gym
from schola.core.utils.id_manager import IdManager
from gymnasium.vector.vector_env import AutoresetMode, VectorEnv

T = TypeVar("T")


class GymEnv(gym.Env):
    """
    A Gym Environment that wraps a Schola Environment.

    Parameters
    ----------
    simulator : BaseSimulator
        Simulator managing the Unreal (or other) process lifecycle.
    protocol : BaseRLProtocol
        gRPC (or other) protocol used to talk to the running simulation.
    verbosity : int, default=0
        Reserved verbosity level for future logging hooks.
    """

    def __init__(
        self, simulator: BaseSimulator, protocol: BaseRLProtocol, verbosity: int = 0
    ):

        self.protocol = protocol
        self.simulator = simulator
        if not isinstance(self.protocol, self.simulator.supported_protocols):
            raise UnsupportedProtocolException(
                f"Protocol {self.protocol} is not supported by the simulator {self.simulator}."
            )

        self.protocol.start()
        self.simulator.start(self.protocol.properties)

        self.protocol.send_startup_msg(auto_reset_type=AutoResetType.DISABLED)

        self._agent_id, self.action_space, self.observation_space = (
            self._define_environment()
        )
        super().__init__()

    def _define_environment(self):
        """
        Returns
        -------
        tuple
            ``(agent_id, action_space, observation_space)`` for the single allowed agent.

        Raises
        ------
        MultipleEnvironmentsException
            If the definition contains more than one environment.
        MultipleAgentsException
            If the single environment contains more than one agent.
        """

        ids, self.agent_types, obs_defns, action_defns = self.protocol.get_definition()

        id_manager = IdManager(ids)
        env_id, agent_id = id_manager[0]

        action_space = action_defns[env_id][agent_id]
        obs_space = obs_defns[env_id][agent_id]

        if id_manager.num_envs > 1:
            self.protocol.close()
            self.simulator.stop()
            raise MultipleEnvironmentsException(id_manager.num_envs, GymEnv)

        if id_manager.num_ids > 1:
            self.protocol.close()
            self.simulator.stop()
            raise MultipleAgentsException(id_manager.num_ids, GymEnv)

        return agent_id, action_space, obs_space

    def close(self, **kwargs):
        self.protocol.close()
        self.simulator.stop()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        # info values are ``str`` in practice; typed ``Any`` to match Gymnasium
        # and stay forward-compatible with non-string payloads.
        super().reset(seed=seed, options=options)
        # wrap options in a list if provided
        _options : list[dict[str,Any]] | None = [options] if options is not None else None
        seeds = [seed] if seed is not None else None
        obs, nested_infos = self.protocol.send_reset_msg(seeds=seeds, options=_options)
        return obs[0][self._agent_id], nested_infos[0][self._agent_id]

    def step(
        self, action: Any
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Take one environment step for the single agent.

        Parameters
        ----------
        action : Any
            Action for the agent, matching :attr:`action_space`.

        Returns
        -------
        tuple
            ``(observation, reward, terminated, truncated, info)`` as in Gymnasium.

        Notes
        -----
        Auto-reset is disabled on the protocol, so bootstrap observations from a
        fresh episode after a terminal step are not merged into this return value.
        """
        # reset mode is disabled so we can ignore the observations/infos coresponding to the first step of a new episode
        observations, rewards, terminateds, truncateds, nested_infos, _, _ = (
            self.protocol.send_action_msg(
                {0: {self._agent_id: action}}, {self._agent_id: self.action_space}
            )
        )
        observations, rewards, terminated, truncated, infos = (
            observations[0][self._agent_id],
            rewards[0][self._agent_id],
            terminateds[0][self._agent_id],
            truncateds[0][self._agent_id],
            cast(Dict[str, Any], nested_infos[0][self._agent_id]),
        )
        return observations, rewards, terminated, truncated, infos


class GymVectorEnv(VectorEnv):
    """
    A Gym Vector Environment that wraps a Schola Environment.

    Parameters
    ----------
    simulator : BaseSimulator
        Simulator managing the Unreal (or other) process lifecycle.
    protocol : BaseRLProtocol
        Protocol used to talk to the running simulation.
    verbosity : int, default=0
        Reserved verbosity level for future logging hooks.
    autoreset_mode : str or gymnasium.vector.vector_env.AutoresetMode, default=AutoresetMode.SAME_STEP
        Passed to the protocol as auto-reset behavior when episodes end.

    Notes
    -----
    :meth:`get_attr` follows Gymnasium's vector API but always returns ``None`` per
    slot because individual sub-environments are not exposed.

    Attributes
    ----------
    reset_infos : List[Dict[str, Any]]
        One ``info`` dict per flattened agent slot (length ``num_envs``), updated on each
        :meth:`reset`. Values are strings from the simulator in typical setups.
        For Gymnasium's batched layout (arrays plus ``_<key>`` masks), use the ``infos``
        value returned from :meth:`reset`; that dict is built with ``VectorEnv._add_info``.
    """

    def __init__(
        self,
        simulator: BaseSimulator,
        protocol: BaseRLProtocol,
        verbosity: int = 0,
        autoreset_mode: str | AutoresetMode = AutoresetMode.SAME_STEP,
    ):

        self.protocol = protocol
        self.simulator = simulator
        if not isinstance(self.protocol, self.simulator.supported_protocols):
            raise UnsupportedProtocolException(
                f"Protocol {self.protocol} is not supported by the simulator {self.simulator}."
            )

        self.protocol.start()
        self.simulator.start(self.protocol.properties)

        self.autoreset_mode = (
            autoreset_mode
            if isinstance(autoreset_mode, AutoresetMode)
            else AutoresetMode(autoreset_mode)
        )
        self.metadata["autoreset_mode"] = self.autoreset_mode

        self.protocol.send_startup_msg(
            auto_reset_type=coerce_auto_reset_type(self.autoreset_mode)
        )

        self._define_environment()

        self.reset_infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]

        super().__init__()

    def _define_environment(self):
        """
        Populates ``id_manager``, ``num_envs``, spaces, and validates homogeneity.

        Raises
        ------
        NoEnvironmentsException
            If Unreal reports no environments.
        NoAgentsException
            If any environment has zero agents.
        AssertionError
            If observation or action spaces differ across agents.
        """

        ids, agent_types, obs_defns, action_defns = self.protocol.get_definition()

        self.id_manager = IdManager(ids)
        self.num_envs = self.id_manager.num_ids

        self.action_space = gym.vector.utils.batch_differing_spaces(
            self.id_manager.flatten_dict_of_dicts(action_defns)
        )

        self.observation_space = gym.vector.utils.batch_differing_spaces(
            self.id_manager.flatten_dict_of_dicts(obs_defns)
        )

        first_env_id, first_agent_id = self.id_manager[0]

        self.single_action_space = action_defns[first_env_id][first_agent_id]
        self.single_observation_space = obs_defns[first_env_id][first_agent_id]
        try:
            if len(ids) == 0:
                raise NoEnvironmentsException()

            for env_id, agent_id_list in enumerate(self.id_manager.ids):
                if len(agent_id_list) == 0:
                    raise NoAgentsException(env_id)

            for env_id, agent_id in self.id_manager.id_list:
                assert (
                    action_defns[env_id][agent_id] == self.single_action_space
                ), f"Action Space Mismatch on Agent:{agent_id} in Env {env_id}.\nGot: {action_defns[env_id][agent_id]}\nExpected:{self.single_action_space}"
                assert (
                    obs_defns[env_id][agent_id] == self.single_observation_space
                ), f"Observation Space Mismatch on Agent:{agent_id} in Env {env_id}.\nGot: {obs_defns[env_id][agent_id]}\nExpected:{self.single_observation_space}"
        except Exception as e:
            self.protocol.close()
            self.simulator.stop()
            raise e

    def close(self, **kwargs):
        self.protocol.close()
        self.simulator.stop()

    def get_attr(self, name: str) -> list[Any]:
        """
        Get an attribute from the environment.

        Parameters
        ----------
        name: str
            The name of the attribute to get.

        Notes
        -----
        This method is not implemented and will always return a list of None values, as sub-environments are not individually accessible.
        """
        return [None for x in range(0, self.num_envs)]

    def reset(
        self,
        *,
        seed: List[int] | int | None = None,
        options: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        # info values are ``str`` in practice; typed ``Any`` to match Gymnasium
        # and stay forward-compatible with non-string payloads.
        num_unreal_envs = self.id_manager.num_envs
        if seed is not None:
            if isinstance(seed, int):
                self.seed_sequence = np.random.SeedSequence(entropy=seed)
                self._np_random = np.random.default_rng(self.seed_sequence.spawn(1)[0])
                self._np_random_seed = seed
                seed = [
                    np.int32(x.generate_state(1)).item()
                    for x in self.seed_sequence.spawn(num_unreal_envs)
                ]
            if isinstance(seed, list):
                if len(seed) != num_unreal_envs:
                    if len(seed) == self.num_envs:
                        raise ValueError(
                            "Number of seeds must match the number of Unreal sub-environments "
                            f"({num_unreal_envs}), not the Gym vector size ({self.num_envs}), "
                            "when passed as a list."
                        )
                    raise ValueError(
                        "Number of seeds must match the number of Unreal sub-environments "
                        f"({num_unreal_envs}) when passed as a list. Got {len(seed)}."
                    )

        processed_options = None
        if options is not None:
            if isinstance(options, dict) and "reset_mask" in options:
                raise NotImplementedError(
                    "reset_mask option is not currently supported in Schola Vector Environments."
                )
            if isinstance(options, dict):
                processed_options = [options.copy() for _ in range(num_unreal_envs)]
            else:
                if len(options) != num_unreal_envs:
                    raise ValueError(
                        f"Expected options list length {num_unreal_envs} (Unreal "
                        f"sub-environments), got {len(options)}. GymVectorEnv.num_envs "
                        f"is {self.num_envs} (flattened agent slots); per-env options "
                        f"must align with Unreal environment count, not num_envs."
                    )
                for option_dict in options:
                    if "reset_mask" in option_dict:
                        raise NotImplementedError(
                            "reset_mask option is not currently supported in Schola Vector Environments."
                        )
                processed_options = [option_dict.copy() for option_dict in options]

        obs, nested_infos = self.protocol.send_reset_msg(
            seeds=seed, options=processed_options
        )

        self.reset_infos = [{} for _ in range(self.num_envs)]
        infos = {}
        for env_id, info_dict in enumerate(nested_infos):
            for agent_id in info_dict:
                uid = self.id_manager[env_id, agent_id]
                slot_info = nested_infos[env_id][agent_id]
                self.reset_infos[uid] = dict(slot_info)
                infos = self._add_info(infos, slot_info, uid)

        # flatten the observations, converting from dict to list using key as indices
        batched_observations = gym.vector.utils.create_empty_array(
            self.single_observation_space, n=self.num_envs
        )
        flattened_observations = self.id_manager.flatten_list_of_dicts(obs)
        gym.vector.utils.concatenate(
            self.single_observation_space, flattened_observations, batched_observations
        )

        return batched_observations, infos

    def unbatch_actions(
        self, actions: Mapping[int, np.ndarray]
    ) -> Mapping[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        Unbatch actions from Dict[ObsID,Batched] to a nested Dict[EnvId,Dict[AgentId,Dict[ObsId,Value]]], effectively moving the env, and agent dimensions into Dictionaries.

        Parameters
        ----------
        actions : Dict[int,np.ndarray]
            The batched actions to unbatch.

        Returns
        -------
        Dict[int,Dict[int,Dict[str,np.ndarray]]]
            The unbatched actions.
        """
        # To prevent issues with non-iterable spaces we use the regular action space if num_envs ==1
        it = gym.vector.utils.iterate(self.action_space, actions)
        return self.id_manager.nest_list_to_dict_of_dicts([value for value in it])

    def step(
        self, actions: Dict[int, np.ndarray]
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        unbatched_actions = self.unbatch_actions(actions)
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            nested_infos,
            initial_obs,
            initial_infos,
        ) = self.protocol.send_action_msg(
            unbatched_actions, defaultdict(lambda: self.single_action_space)
        )

        array_rewards = np.asarray(self.id_manager.flatten_list_of_dicts(rewards))
        array_terminateds = np.asarray(
            self.id_manager.flatten_list_of_dicts(terminateds)
        )
        array_truncateds = np.asarray(self.id_manager.flatten_list_of_dicts(truncateds))

        infos = {}

        for env_id, agent_id_list in enumerate(self.id_manager.ids):
            any_done = False
            all_done = True
            for agent_id in agent_id_list:
                uid = self.id_manager[env_id, agent_id]
                is_done = terminateds[env_id][agent_id] or truncateds[env_id][agent_id]
                any_done = any_done or is_done
                all_done = all_done and is_done

            # We don't handle the case where 1 agent ends early currently.
            if any_done and not all_done:
                raise EnvironmentException(
                    f"SB3 with multi-agent environments does not support agents completing at different steps. Env {env_id} had agents in different completion states."
                )

        # if we get these then it means we are in self reset mode

        for env_id, agent_id_list in enumerate(self.id_manager.ids):
            for agent_id in agent_id_list:
                uid = self.id_manager[env_id, agent_id]
                if env_id in initial_obs:
                    infos = self._add_info(
                        infos,
                        {
                            "final_info": nested_infos[env_id][agent_id],
                            "final_obs": observations[env_id][agent_id],
                            **initial_infos[env_id][agent_id],
                        },
                        uid,
                    )
                    observations[env_id][agent_id] = initial_obs[env_id][agent_id]
                else:
                    infos = self._add_info(infos, nested_infos[env_id][agent_id], uid)

        # flatten the observations, converting to one bit numpy ndarray

        array_observations = gym.vector.utils.create_empty_array(
            self.single_observation_space, n=self.num_envs
        )
        flattened_observations = self.id_manager.flatten_list_of_dicts(observations)
        gym.vector.utils.concatenate(
            self.single_observation_space, flattened_observations, array_observations
        )

        return (
            array_observations,
            array_rewards,
            array_terminateds,
            array_truncateds,
            infos,
        )
