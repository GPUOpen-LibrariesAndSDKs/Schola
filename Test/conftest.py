# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""Root pytest fixtures and helpers for the Schola plugin Python tests."""

from concurrent import futures
from typing import Callable, List
from unittest.mock import MagicMock
import grpc

import gymnasium as gym
import numpy as np
import pytest
from functools import cache
from typing import Dict, List, Optional, Any, Tuple
import pytest

from schola.core.protocols.base_protocol import BaseRLProtocol
from schola.core.simulators.base_simulator import BaseSimulator
from schola.core.utils.dict_helpers import map_dict
from schola.core.model import StateMetadata
from schola.gym import env


@pytest.fixture(scope="function")
def mock_protocol_and_simulator():
    """Build a mock protocol + simulator pair that pass the
    ``supported_protocols`` isinstance check in the SB3 / RLlib env
    constructors (``VecEnv.__init__`` / ``BaseRayEnv.__init__``).

    ``MagicMock(spec=BaseRLProtocol)`` makes ``isinstance(protocol, BaseRLProtocol)``
    return ``True``. ``simulator.supported_protocols`` is overridden to a real
    tuple so the isinstance check uses a concrete class object. All other
    lifecycle calls (``protocol.start()``, ``simulator.start(...)``,
    ``protocol.send_startup_msg(...)``) auto-resolve as MagicMock no-ops, which
    is what lets the env-options tests drive the real env ``__init__``
    end-to-end without spinning up a gRPC server.
    """
    protocol = MagicMock(spec=BaseRLProtocol)
    simulator = MagicMock(spec=BaseSimulator)
    simulator.supported_protocols = (BaseRLProtocol,)
    return protocol, simulator


@pytest.fixture(scope="function")
def stub_protocol_class():
    """Real ``BaseRLProtocol`` subclass for tests that need a *class*, not an
    instance: ``ScholaEnvRunner.make_env`` does ``issubclass(...)`` then
    ``cls(**protocol_args)``, which a ``MagicMock(spec=...)`` can't satisfy.

    Each instance records its ``__init__`` kwargs on ``init_kwargs`` and
    appends itself to ``cls.instances``. ``get_definition`` / ``send_reset_msg``
    are MagicMocks pre-armed to ``cls.NUM_ENVS`` so the real env ``__init__``
    runs without a live server; re-arm ``instance.get_definition.return_value``
    for a different shape. Scenario constants (``NUM_ENVS``, ``AGENT_ID``, ...)
    are class attributes tests can read directly.
    """

    class _StubProtocol(BaseRLProtocol):
        NUM_ENVS: int = 2
        OBS_SPACE = gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32)
        ACTION_SPACE = gym.spaces.Discrete(2)
        AGENT_ID = "agent_0"

        instances: list = []
        # Shadow the ``@property`` descriptor on ``BaseProtocol`` with a
        # plain class attribute so ``simulator.start(protocol.properties)``
        # reads a real dict without needing an instance-level setter.
        properties = {}

        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.start = MagicMock()
            self.close = MagicMock()
            self.send_startup_msg = MagicMock()
            self.get_definition = MagicMock(
                return_value=(
                    [[self.AGENT_ID] for _ in range(self.NUM_ENVS)],
                    [{self.AGENT_ID: ""} for _ in range(self.NUM_ENVS)],
                    {i: {self.AGENT_ID: self.OBS_SPACE} for i in range(self.NUM_ENVS)},
                    {
                        i: {self.AGENT_ID: self.ACTION_SPACE}
                        for i in range(self.NUM_ENVS)
                    },
                )
            )
            self.send_reset_msg = MagicMock(
                return_value=(
                    [
                        {self.AGENT_ID: np.zeros((1,), dtype=np.float32)}
                        for _ in range(self.NUM_ENVS)
                    ],
                    [{self.AGENT_ID: {}} for _ in range(self.NUM_ENVS)],
                )
            )
            self.send_action_msg = MagicMock()
            type(self).instances.append(self)

        def __bool__(self) -> bool:
            return True

    return _StubProtocol


@pytest.fixture(scope="function")
def stub_simulator_class():
    """Real ``BaseSimulator`` subclass paired with ``stub_protocol_class``.

    ``supported_protocols`` is ``(BaseRLProtocol,)`` so the
    ``isinstance(protocol, simulator.supported_protocols)`` check in
    ``RayVecEnv.__init__`` / ``VecEnv.__init__`` accepts any
    ``BaseRLProtocol`` subclass -- including the stub protocol class.

    Like ``stub_protocol_class``, each instance records its ``__init__``
    kwargs on ``init_kwargs`` and appends itself to ``cls.instances`` so
    tests can assert on what the SUT instantiated it with.
    """

    class _StubSimulator(BaseSimulator):
        instances: list = []
        supported_protocols = (BaseRLProtocol,)

        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.start = MagicMock()
            self.stop = MagicMock()
            type(self).instances.append(self)

        def __bool__(self) -> bool:
            return True

    return _StubSimulator


from .envs.gym_server import GymToGymServiceServicer, VecGymToGymServiceServicer
from .envs.imitation_server import GymToImitationServiceServicer
from .envs.pettingzoo_server import (
    PettingZooToGymServiceServicer,
    VecPettingZooToGymServiceServicer,
)
import schola.generated.GymConnector_pb2_grpc as gym_connector_grpc
import schola.generated.ImitationConnector_pb2_grpc as imitation_connector_grpc
import minigrid
from minigrid.wrappers import DictObservationSpaceWrapper
import gymnasium as gym


@pytest.fixture(scope="function")
def make_env_server():

    servers = []

    def _make_env_server(
        env_name: str | Callable[..., gym.Env], wrappers: list = None, port: int = 0
    ):
        servicer = GymToGymServiceServicer(env_name, wrappers)
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        gym_connector_grpc.add_GymServiceServicer_to_server(servicer, server)
        port = server.add_insecure_port(f"[::]:{port}")
        server.start()
        servers.append(server)
        return port

    yield _make_env_server

    for server in servers:
        server.stop(0)

    for server in servers:
        server.wait_for_termination()


@pytest.fixture(scope="function")
def make_vec_env_server():

    servers = []

    def _make_env_server(
        env_name: List[Callable[..., gym.Env]],
        wrappers: Optional[list] = None,
        port: int = 0,
    ):
        servicer = VecGymToGymServiceServicer(env_name, wrappers)
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        gym_connector_grpc.add_GymServiceServicer_to_server(servicer, server)
        port = server.add_insecure_port(f"[::]:{port}")
        server.start()
        servers.append(server)
        return port

    yield _make_env_server

    for server in servers:
        server.stop(0)

    for server in servers:
        server.wait_for_termination()


@pytest.fixture(scope="function")
def make_pettingzoo_env_server():
    """Create a PettingZoo environment server (single environment)."""
    servers = []
    servicers = []

    def _make_env_server(
        env_name: str | Callable, wrappers: Optional[list] = None, port: int = 0
    ):
        servicer = PettingZooToGymServiceServicer(env_name, wrappers)
        # Set max message sizes to 100MB to handle large environment definitions
        options = [
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ]
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=options)
        gym_connector_grpc.add_GymServiceServicer_to_server(servicer, server)
        port = server.add_insecure_port(f"[::]:{port}")
        server.start()
        servers.append(server)
        servicers.append(servicer)
        return port

    yield _make_env_server

    # Close environments explicitly from this (main) thread before stopping the server.
    # This prevents __del__ from firing on a background gRPC thread, which would
    # call pygame's SDL teardown off the main thread and cause a segfault.
    for servicer in servicers:
        servicer.close()

    for server in servers:
        server.stop(0)

    for server in servers:
        server.wait_for_termination()


@pytest.fixture(scope="function")
def make_vec_pettingzoo_env_server():
    """Create a vectorized PettingZoo environment server (multiple environments)."""
    servers = []
    servicers = []

    def _make_env_server(
        env_funcs: List[Callable], wrappers: Optional[list] = None, port: int = 0
    ):
        servicer = VecPettingZooToGymServiceServicer(env_funcs, wrappers)
        # Set max message sizes to 100MB to handle large messages
        options = [
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ]
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=options)
        gym_connector_grpc.add_GymServiceServicer_to_server(servicer, server)
        port = server.add_insecure_port(f"[::]:{port}")
        server.start()
        servers.append(server)
        servicers.append(servicer)
        return port

    yield _make_env_server

    # Close environments explicitly from this (main) thread before stopping the server.
    # This prevents __del__ from firing on a background gRPC thread, which would
    # call pygame's SDL teardown off the main thread and cause a segfault.
    for servicer in servicers:
        servicer.close()

    for server in servers:
        server.stop(0)

    for server in servers:
        server.wait_for_termination()


@pytest.fixture(
    scope="function",
    params=[
        ("CartPole-v1", None),
        ("MountainCar-v0", None),
        ("MiniGrid-DoorKey-5x5-v0", [DictObservationSpaceWrapper]),
    ],
    ids=lambda x: x[0],
)
def gym_id_and_wrappers(request):
    env_id, wrappers = request.param
    return env_id, wrappers


from gymnasium.spaces import flatdim, flatten_space, unflatten
from gymnasium.vector.utils import batch_space
import onnx
import numpy as np
import onnxruntime as ort


@pytest.fixture(scope="function")
def onnx_model_checker():
    def _check_onnx_model(
        model_path,
        observation_space,
        action_space,
        state_shapes: Dict[str, Tuple[int, ...]] = None,
        metadata: Dict[str, Dict[str, str]] = None,
    ):
        """Check that the ONNX model exists and has the correct input and output names."""
        assert model_path.exists(), f"ONNX file not created at {model_path}"
        batch_size = 2
        if not isinstance(observation_space, gym.spaces.Dict):
            observation_space = gym.spaces.Dict({"obs": observation_space})

        # then we can batch the flat observation space (otherwise we get the wrong dimensions)
        batched_observation_space = batch_space(observation_space, n=batch_size)

        if not isinstance(action_space, gym.spaces.Dict):
            action_space = gym.spaces.Dict({"action": action_space})

        action_space = batch_space(action_space, n=batch_size)

        if state_shapes is None:
            state_shapes = {}

        # Check that the model has the correct input and output names
        model = onnx.load(model_path)

        input_names = [input.name for input in model.graph.input]
        output_names = [output.name for output in model.graph.output]

        assert set(input_names) == set(observation_space.spaces.keys()) | {
            f"state_in_{k}" for k in state_shapes.keys()
        }, "Input names should be the keys of the observation space or state inputs"
        assert set(output_names) == set(action_space.spaces.keys()) | {
            f"state_out_{k}" for k in state_shapes.keys()
        }, "Output names should be the keys of the action space or 'state_out'"

        # check the metadata of the model (embedded on state inputs)
        if metadata is not None:
            found_metadata = 0
            for inp in model.graph.input:
                if inp.name in metadata:
                    inp_metadata = metadata[inp.name]
                    found_metadata_key = 0
                    for prop in inp.metadata_props:
                        if prop.key in inp_metadata:
                            assert (
                                inp_metadata[prop.key] == prop.value
                            ), f"Model metadata for {inp.name} should match the expected metadata"
                            found_metadata_key += 1
                    assert found_metadata_key == len(
                        inp_metadata
                    ), f"Expected to find {len(inp_metadata)} metadata keys for {inp.name}. Found {found_metadata_key}"
                    found_metadata += 1
            assert found_metadata == len(
                metadata
            ), f"Expected to find metadata for {len(metadata)} state inputs. Found {found_metadata}"
        # Run one step of inference on the model and check that the outputs match the expected shapes/spaces

        input_data = batched_observation_space.sample()

        # no-op if state_shapes is empty
        for k, v in state_shapes.items():
            input_data[f"state_in_{k}"] = np.random.rand(batch_size, *v).astype(
                np.float32
            )

        ort_sess = ort.InferenceSession(model_path)
        outputs = ort_sess.run(None, input_data)

        output_states = {
            k: v for k, v in zip(output_names, outputs) if k.startswith("state_out_")
        }
        output_actions = {
            k: v
            for k, v in zip(output_names, outputs)
            if not k.startswith("state_out_")
        }

        # check the output actions
        for action_name, action in output_actions.items():
            assert action_space.spaces[action_name].contains(
                action
            ), f"Expected Action '{action_name}' to be in action space. Got {action}"

        # check the state outputs
        assert set(output_states.keys()) == {
            f"state_out_{k}" for k in state_shapes.keys()
        }, f"Expected output states to be the keys of the state shapes. Got {output_states.keys()} != {state_shapes.keys()}"
        for state_name, state in output_states.items():
            assert state.shape == (
                batch_size,
                *state_shapes[state_name[len("state_out_") :]],
            ), f"Expected output state '{state_name}' to have shape {state_shapes[state_name[len('state_out_'):]]}. Got {state.shape}"

    return _check_onnx_model
