# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""gRPC servicers exposing Gymnasium environments for Schola."""

import functools
from typing import Callable, List, Optional
import numpy as np
from schola.core.protocols.protobuf.deserialize import from_proto
from schola.core.protocols.protobuf.serialize import (
    space_to_proto,
    to_proto,
    make_generic,
    fill_generic,
)
from schola.generated.GymConnector_pb2_grpc import GymServiceServicer
from schola.generated.GymConnector_pb2 import *
from schola.generated.Definitions_pb2 import *
from schola.generated.StateUpdates_pb2 import *
from schola.generated.State_pb2 import *
from schola.generated.Spaces_pb2 import *
from schola.generated.Points_pb2 import *
import grpc
import gymnasium as gym
import traceback


def capture_traceback(func):
    """
    Capture traceback and move it to the body of the message so that it is forwarded to the client.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            raise Exception(traceback.format_exc())

    return wrapper


def _info_as_str_map(info: Optional[dict]) -> dict[str, str]:
    return {k: str(v) for k, v in (info or {}).items()}


def _seed_from_proto(env_settings: EnvironmentSettings) -> int | None:
    """
    Return a Gymnasium reset seed from protobuf environment settings.

    Proto3 oneof ``optional_seed`` defaults to ``0`` when unset; treat ``0`` as
    no seed so ``env.reset(seed=None)`` is used instead of forcing seed 0.
    """
    if not env_settings.HasField("seed") or env_settings.seed == 0:
        return None
    return env_settings.seed


def wrap(env, wrappers: Optional[list] = None):
    if not isinstance(env, gym.Env):
        env = env()
    if wrappers:
        for wrapper in wrappers:
            env = wrapper(env)
    return env

class GymToGymServiceServicer(GymServiceServicer):

    def __init__(
        self, env_id: str | Callable[..., gym.Env], wrappers: Optional[list] = None
    ):
        if isinstance(env_id, str):
            self._env_factory = lambda: gym.make(env_id)
        else:
            self._env_factory = env_id
        self._agent_id = "single_agent"
        self._last_reset_obs = None
        self._last_reset_info = None
        self._env = None
        self._wrapper_classes = wrappers if wrappers else []

    @property
    def env(self) -> gym.Env:
        assert self._env is not None, "Environment not initialized"
        return self._env

    @capture_traceback
    def UpdateState(self, request: StateUpdate, context) -> State:
        msg_type = request.WhichOneof("update")
        if msg_type == "reset":
            env_settings = request.reset.environments[0]
            seed = _seed_from_proto(env_settings)
            options = env_settings.options
            self._last_reset_obs, self._last_reset_info = self.env.reset(
                seed=seed, options=dict(options.items())
            )
            response = State(
                initial_state=InitialState(
                    environment_states={
                        0: InitialEnvironmentState(
                            agent_states={
                                self._agent_id: InitialAgentState(
                                    observations=make_generic(
                                        to_proto(
                                            self.env.observation_space,
                                            self._last_reset_obs,
                                        )
                                    ),
                                    info=_info_as_str_map(self._last_reset_info),
                                )
                            }
                        )
                    }
                )
            )
            return response

        elif msg_type == "step":
            action = from_proto(request.step.environments[0].updates[self._agent_id])
            obs, reward, terminated, truncated, info = self.env.step(action)
            return State(
                training_state=TrainingState(
                    environment_states=[
                        EnvironmentState(
                            agent_states={
                                self._agent_id: AgentState(
                                    observations=make_generic(
                                        to_proto(self.env.observation_space, obs)
                                    ),
                                    reward=float(reward),
                                    terminated=terminated,
                                    truncated=truncated,
                                    info=_info_as_str_map(info),
                                )
                            }
                        )
                    ]
                )
            )
        elif request.status == CLOSED:
            return State()
        elif request.status == ERROR:
            return State()
        else:
            raise ValueError("Invalid update message")

    @capture_traceback
    def RequestInitialTrainingState(
        self, request: InitialStateRequest, context
    ) -> InitialState:
        assert (
            self._last_reset_obs is not None
        ), "RequestInitialTrainingState requires a prior UpdateState(reset); last reset observation is unset."
        return InitialState(
            environment_states={
                0: InitialEnvironmentState(
                    agent_states={
                        self._agent_id: InitialAgentState(
                            observations=make_generic(
                                to_proto(
                                    self.env.observation_space,
                                    self._last_reset_obs,
                                )
                            ),
                            info=_info_as_str_map(self._last_reset_info),
                        )
                    }
                )
            }
        )

    @capture_traceback
    def RequestTrainingDefinition(
        self, request: TrainingDefinitionRequest, context
    ) -> TrainingDefinition:
        return TrainingDefinition(
            environment_definitions=[
                EnvironmentDefinition(
                    agent_definitions={
                        self._agent_id: AgentDefinition(
                            obs_space=make_generic(
                                space_to_proto(self.env.observation_space)
                            ),
                            action_space=make_generic(
                                space_to_proto(self._env.action_space)
                            ),
                        )
                    }
                )
            ]
        )

    @capture_traceback
    def StartGymConnector(
        self, request: GymConnectorStartRequest, context
    ) -> GymConnectorStartResponse:
        self._env = wrap(self._env_factory(), self._wrapper_classes)
        return GymConnectorStartResponse()

    def __del__(self):
        if self._env is not None:
            self._env.close()


class VecGymToGymServiceServicer(GymServiceServicer):
    def __init__(
        self,
        env_id: List[Callable[..., gym.Env]],
        wrappers: Optional[list] = None,
    ):

        self._env_factory = [
            functools.partial(wrap, env_func, wrappers) for env_func in env_id
        ]
        self._n_envs = len(self._env_factory)

        self._agent_id = "single_agent"
        self._last_reset_obs = None
        self._last_reset_info = None
        self._envs: Optional[List[gym.Env]] = None
        self._wrapper_classes = wrappers if wrappers else []
        self._autoreset_envs = np.array(
            [False for _ in range(self._n_envs)], dtype=np.bool_
        )

    @property
    def envs(self) -> List[gym.Env]:
        assert self._envs is not None, "Environments not initialized"
        return self._envs

    @capture_traceback
    def UpdateState(self, request: StateUpdate, context) -> State:
        msg_type = request.WhichOneof("update")
        if msg_type == "reset":

            initial_state = InitialState()
            obs = [None for _ in range(self._n_envs)]
            rewards = [0.0 for _ in range(self._n_envs)]

            for i in range(self._n_envs):
                env_settings = request.reset.environments[i]
                seed = _seed_from_proto(env_settings)
                options = env_settings.options
                obs, info = self.envs[i].reset(seed=seed, options=dict(options.items()))

                initial_agent_state: InitialAgentState = (
                    initial_state.environment_states[i].agent_states[self._agent_id]
                )
                fill_generic(
                    to_proto(self.envs[0].observation_space, obs),
                    initial_agent_state.observations,
                )

                initial_agent_state.info.update(
                    ((k, str(v)) for k, v in info.items())
                )

            return State(initial_state=initial_state)

        elif msg_type == "step":

            # if we just get the action from the updates, it will generate an empty item
            actions = [{} for _ in range(self._n_envs)]
            for i in range(self._n_envs):
                for agent_id, action_proto in request.step.environments[
                    i
                ].updates.items():
                    actions[i][agent_id] = from_proto(action_proto)
                    if agent_id != self._agent_id:
                        raise ValueError(f"Agent {agent_id} is not the single agent")

            obs = [None for _ in range(self._n_envs)]
            rewards = [0.0 for _ in range(self._n_envs)]
            terminations = [False for _ in range(self._n_envs)]
            truncations = [False for _ in range(self._n_envs)]
            infos = [{} for _ in range(self._n_envs)]

            initial_obs = {}
            initial_infos = {}

            for i, action in enumerate(actions):
                # adapted from Gymnasium SyncVectorEnv Step function
                if self.autoreset_mode == AutoResetType.NEXT_STEP:
                    if self._autoreset_envs[i]:
                        obs[i], infos[i] = self.envs[i].reset()
                        rewards[i] = 0.0
                        terminations[i] = False
                        truncations[i] = False
                    else:
                        (
                            obs[i],
                            rewards[i],
                            terminations[i],
                            truncations[i],
                            infos[i],
                        ) = self.envs[i].step(actions[i][self._agent_id])
                elif self.autoreset_mode == AutoResetType.DISABLED:
                    # assumes that the user has correctly autoreset
                    assert not self._autoreset_envs[i], f"{self._autoreset_envs}"
                    (
                        obs[i],
                        rewards[i],
                        terminations[i],
                        truncations[i],
                        infos[i],
                    ) = self.envs[i].step(actions[i][self._agent_id])
                elif self.autoreset_mode == AutoResetType.SAME_STEP:
                    (
                        obs[i],
                        rewards[i],
                        terminations[i],
                        truncations[i],
                        infos[i],
                    ) = self.envs[i].step(actions[i][self._agent_id])

                    if terminations[i] or truncations[i]:
                        initial_obs[i], initial_infos[i] = self.envs[i].reset()

            self._autoreset_envs = np.logical_or(terminations, truncations)

            # complete the output message to the client
            output_state = State()
            training_state = output_state.training_state

            for i in range(self._n_envs):
                agent_state: AgentState = (
                    training_state.environment_states.add().agent_states[self._agent_id]
                )

                # fill in the env state portion
                fill_generic(
                    to_proto(self.envs[0].observation_space, obs[i]),
                    agent_state.observations,
                )

                agent_state.info.update(((k, str(v)) for k, v in infos[i].items()))
                agent_state.reward = rewards[i]
                agent_state.terminated = terminations[i]
                agent_state.truncated = truncations[i]

                # fill in the initial observations from a samestep self-reset
                if i in initial_obs:
                    initial_agent_state = output_state.initial_state.environment_states[
                        i
                    ].agent_states[self._agent_id]
                    fill_generic(
                        to_proto(self.envs[0].observation_space, initial_obs[i]),
                        initial_agent_state.observations,
                    )
                    initial_agent_state.info.update(
                        ((k, str(v)) for k, v in initial_infos[i].items())
                    )

            return output_state

        elif request.status == CLOSED:
            return State()
        elif request.status == ERROR:
            return State()
        else:
            raise ValueError("Invalid update message")

    @capture_traceback
    def RequestInitialTrainingState(
        self, request: InitialStateRequest, context
    ) -> InitialState:
        return InitialState()

    @capture_traceback
    def RequestTrainingDefinition(
        self, request: TrainingDefinitionRequest, context
    ) -> TrainingDefinition:

        env_defn_list = [None for _ in range(self._n_envs)]

        for i in range(self._n_envs):
            env_defn_list[i] = EnvironmentDefinition(
                agent_definitions={
                    self._agent_id: AgentDefinition(
                        obs_space=make_generic(
                            space_to_proto(self.envs[0].observation_space)
                        ),
                        action_space=make_generic(
                            space_to_proto(self.envs[0].action_space)
                        ),
                    )
                }
            )
        return TrainingDefinition(environment_definitions=env_defn_list)

    @capture_traceback
    def StartGymConnector(
        self, request: GymConnectorStartRequest, context
    ) -> GymConnectorStartResponse:
        self._envs = [x() for x in self._env_factory]
        self.autoreset_mode = request.autoreset_type
        return GymConnectorStartResponse()

    def __del__(self):
        if self._envs is not None:
            for env in self._envs:
                env.close()
        self._envs = None
