# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""Root pytest fixtures and helpers for the Schola plugin Python tests."""

from concurrent import futures
import itertools
from pathlib import Path
from textwrap import dedent
from typing import Callable, List, cast
from unittest.mock import MagicMock
import grpc

import gymnasium as gym
import numpy as np
import pytest
from functools import cache
from typing import Dict, List, Optional, Any, Tuple
import pytest
from torch import is_nonzero

from schola.core.protocols.base_protocol import BaseRLProtocol
from schola.core.simulators.base_simulator import BaseSimulator
from schola.core.utils.dict_helpers import map_dict
from schola.core.model import StateMetadata
from schola.gym import env


@pytest.fixture(scope="function")
def make_spaces_yaml(tmp_path: Path):
    """Write a model-space YAML file under ``tmp_path`` and return its path."""

    def _make(yaml_text: str, name: str = "spaces.yaml") -> Path:
        path = tmp_path / name
        path.write_text(dedent(yaml_text).strip() + "\n", encoding="utf-8")
        return path

    return _make


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

        # These satisfy BaseRLProtocol's @abstractmethods at class-definition time.
        # __init__ shadows each one with a per-instance MagicMock for call assertions.

        def close(self) -> None: ...

        def start(self) -> None: ...

        def send_startup_msg(self, *args: Any, **kwargs: Any) -> None: ...

        def get_definition(self, *args: Any, **kwargs: Any) -> Any: ...

        def send_reset_msg(
            self, seeds: list[Any] | None = None, options: list[Any] | None = None
        ) -> Any: ...

        def send_action_msg(self, *args: Any, **kwargs: Any) -> Any: ...

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


from schola.core.simulators.gym import (
    GymToGymServiceServicer,
    VecGymToGymServiceServicer,
)
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


from pathlib import Path

from gymnasium.vector.utils import batch_space, concatenate, create_empty_array
import onnx
import onnxruntime as ort


def _check_inputs_have_valid_shape(
    model: onnx.ModelProto,
) -> None:
    """Require every exported input to support dynamic batching."""
    graph_inputs = {inp.name: inp for inp in model.graph.input}
    for input_name, inp in graph_inputs.items():
        dims = inp.type.tensor_type.shape.dim
        assert (
            dims
        ), f"Expected input '{input_name}' to have non-empty dimensions. Must have at least one dimension."

        batch_dim = dims[0]
        assert (
            not batch_dim.dim_value and batch_dim.dim_param
        ), f"Expected input '{input_name}' to have a dynamic batch dimension. Got {batch_dim}"


def check_onnx_model_loads(model_path) -> onnx.ModelProto:
    """Load an ONNX model, failing if the file is missing or cannot be parsed."""
    path = Path(model_path)
    assert path.exists(), f"ONNX file not created at {model_path}"
    try:
        return onnx.load(path)
    except Exception as exc:
        raise AssertionError(
            f"Failed to load ONNX model at {model_path}: {exc}"
        ) from exc


def check_onnx_metadata(
    model: onnx.ModelProto,
    observation_space,
    action_space,
    state_shapes: Dict[str, Tuple[int, ...]] | None = None,
    metadata: Dict[str, Dict[str, str]] | None = None,
) -> None:
    """Check graph input and output names, shapes, and embedded metadata.

    ``metadata`` is keyed by input name, then by metadata property key. When
    ``metadata`` is ``None``, the embedded-metadata comparison is skipped.
    """
    if state_shapes is None:
        state_shapes = {}

    if not isinstance(observation_space, gym.spaces.Dict):
        observation_space = gym.spaces.Dict({"obs": observation_space})

    if not isinstance(action_space, gym.spaces.Dict):
        action_space = gym.spaces.Dict({"action": action_space})

    input_names = [model_input.name for model_input in model.graph.input]
    output_names = [output.name for output in model.graph.output]

    expected_input_names = set(observation_space.spaces.keys()) | {
        f"state_in_{k}" for k in state_shapes.keys()
    }
    assert (
        set(input_names) == expected_input_names
    ), "Input names should be the keys of the observation space or state inputs"
    assert set(output_names) == set(action_space.spaces.keys()) | {
        f"state_out_{k}" for k in state_shapes.keys()
    }, "Output names should be the keys of the action space or 'state_out'"
    _check_inputs_have_valid_shape(model)

    if metadata is None:
        return

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


def _box_edge_samples(space: gym.spaces.Box) -> list[np.ndarray]:
    """Upper bound, lower bound, and the zero vector when each lies in ``space``."""
    samples: list = []
    if np.all(np.isfinite(space.high)):
        samples.append(space.high)
    if np.all(np.isfinite(space.low)):
        samples.append(space.low)
    if np.all(space.low < 0) and np.all(space.high > 0):
        samples.append(np.zeros(space.shape, dtype=space.dtype))
    return samples


def _has_fewer_combinations(counts, num_samples: int) -> bool:
    """Return whether the cartesian product of ``counts`` is smaller than ``num_samples``."""
    total = 1
    for count in counts:
        total *= int(count)
        if total >= num_samples:
            return False
    return True


def _discrete_edge_samples(space: gym.spaces.Discrete, num_samples: int) -> list:
    """Every legal value when there are fewer than ``num_samples`` of them."""
    count = int(space.n)
    if count >= num_samples:
        return []
    start = int(space.start)
    return [space.dtype.type(value) for value in range(start, start + count)]


def _multi_discrete_edge_samples(
    space: gym.spaces.MultiDiscrete, num_samples: int
) -> list[np.ndarray]:
    """Every combination when there are fewer than ``num_samples`` of them."""
    counts = space.nvec.reshape(-1)
    starts = space.start.reshape(-1)
    if not _has_fewer_combinations(counts, num_samples):
        return []
    ranges = [
        range(int(start), int(start) + int(count))
        for start, count in zip(starts, counts)
    ]
    return [
        np.array(combo, dtype=space.dtype).reshape(space.shape)
        for combo in itertools.product(*ranges)
    ]


def _multi_binary_edge_samples(space: gym.spaces.MultiBinary, num_samples: int) -> list[np.ndarray]:
    """Every 0/1 combination when there are fewer than ``num_samples`` of them."""
    num_bits = 1
    for dim in space.shape:
        num_bits *= int(dim)
    if not _has_fewer_combinations(itertools.repeat(2, num_bits), num_samples):
        return []
    return [
        np.array(bits, dtype=space.dtype).reshape(space.shape)
        for bits in itertools.product((0, 1), repeat=num_bits)
    ]


def _edge_samples(space: gym.Space, num_samples: int) -> list:
    """Edge-case samples for ``space``. Empty when the space is sampled at random."""
    if isinstance(space, gym.spaces.Box):
        return _box_edge_samples(space)
    if isinstance(space, gym.spaces.Discrete):
        return _discrete_edge_samples(space, num_samples)
    if isinstance(space, gym.spaces.MultiDiscrete):
        return _multi_discrete_edge_samples(space, num_samples)
    if isinstance(space, gym.spaces.MultiBinary):
        return _multi_binary_edge_samples(space, num_samples)
    if isinstance(space, gym.spaces.Dict):
        if len(space.spaces) == 1:
            key, subspace = next(iter(space.spaces.items()))
            return [{key: sample} for sample in _edge_samples(subspace, num_samples)]
        return _dict_edge_samples(space, num_samples)
    return []


def _dict_edge_samples(space: gym.spaces.Dict, num_samples: int) -> list:
    """One row per subspace edge case. Other subspaces are sampled at random."""
    samples = []
    for key, subspace in space.spaces.items():
        for edge in _edge_samples(subspace, num_samples):
            samples.append(
                {
                    other_key: edge if other_key == key else other_space.sample()
                    for other_key, other_space in space.spaces.items()
                }
            )
    return samples

def _sample_batch_size(batched: np.ndarray | Dict[str, Any]) -> int:
    while isinstance(batched, dict):
        batched = next(iter(batched.values()))
    return int(np.asarray(batched).shape[0])


def make_fake_space_samples(space: gym.Space, num_samples: int) -> Any:
    """Build a batch of fake samples from ``space``.

    The batch starts with edge cases and is then filled with random samples
    until it is ``num_samples`` long. Edge cases are kept even when there are
    more of them than ``num_samples``.

    Box samples are the upper bound, the lower bound, and the zero vector when
    each value is finite and inside the space. Discrete samples include every
    legal value when there are fewer than ``num_samples`` options; otherwise
    the batch is random. MultiDiscrete and MultiBinary (binary) samples do the
    same over combinations of their elements.

    A dictionary with one subspace uses that subspace's samples. A dictionary
    with several subspaces contributes every edge case of each subspace. The
    subspace under test is fixed to that edge, and the others are random. The
    remaining rows, up to ``num_samples``, are fully random.
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    samples = _edge_samples(space, num_samples)
    remaining = num_samples - len(samples)
    if remaining > 0:
        samples.extend(space.sample() for _ in range(remaining))
    return concatenate(space, samples, create_empty_array(space, n=len(samples)))


def check_onnx_runtime(
    model: onnx.ModelProto,
    observation_space,
    action_space,
    state_shapes: Dict[str, Tuple[int, ...]] | None = None,
    num_samples: int = 32,
) -> None:
    """Run inference on fake samples and check that the outputs match the spaces.

    ``num_samples`` is the number of inputs to draw. Edge cases from
    ``make_fake_space_samples`` are always included, so the batch can be longer
    than ``num_samples``.
    """
    if state_shapes is None:
        state_shapes = {}

    if not isinstance(observation_space, gym.spaces.Dict):
        observation_space = gym.spaces.Dict({"obs": observation_space})

    input_data = make_fake_space_samples(observation_space, num_samples)
    batch_size = _sample_batch_size(input_data)

    if not isinstance(action_space, gym.spaces.Dict):
        action_space = gym.spaces.Dict({"action": action_space})

    action_space = cast(gym.spaces.Dict, batch_space(action_space, n=batch_size))

    output_names = [output.name for output in model.graph.output]

    # no-op if state_shapes is empty
    for k, v in state_shapes.items():
        input_data[f"state_in_{k}"] = np.random.rand(batch_size, *v).astype(
            np.float32
        )

    ort_sess = ort.InferenceSession(model.SerializeToString())
    outputs = ort_sess.run(None, input_data)

    output_states = {
        k: v for k, v in zip(output_names, outputs) if k.startswith("state_out_")
    }
    output_actions = {
        k: v
        for k, v in zip(output_names, outputs)
        if not k.startswith("state_out_")
    }

    for action_name, action in output_actions.items():
        assert action_space.spaces[action_name].contains(
            action
        ), f"Expected Action '{action_name}' to be in action space. Got {action}"

    assert set(output_states.keys()) == {
        f"state_out_{k}" for k in state_shapes.keys()
    }, f"Expected output states to be the keys of the state shapes. Got {output_states.keys()} != {state_shapes.keys()}"
    for state_name, state in output_states.items():
        assert state.shape == (
            batch_size,
            *state_shapes[state_name[len("state_out_") :]],
        ), f"Expected output state '{state_name}' to have shape {state_shapes[state_name[len('state_out_'):]]}. Got {state.shape}"
