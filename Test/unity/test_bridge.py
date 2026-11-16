# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Wire and real-gRPC adapter tests; no Unity Editor installation required."""

from concurrent.futures import ThreadPoolExecutor
import json
import socket
import struct

import grpc
import pytest

from schola.generated import GymConnector_pb2 as connector
from schola.generated import GymConnector_pb2_grpc as services
from schola.generated import Points_pb2 as points
from schola.generated import StateUpdates_pb2 as updates
from schola.unity.bridge import (
    MAX_MESSAGE_BYTES,
    UnityBridgeServer,
    UnityConnection,
    UnityError,
    _decode_tree,
    _encode_point,
)


def box(value=0):
    return [dict(parent=-1, kind="box", values=[value], shape=[1])]


def test_nested_point_roundtrip_preserves_keys_shape_and_values():
    point = points.Point()
    point.dict_point.values["nested"].dict_point.values["move"].box_point.values.extend(
        [1, 2]
    )
    point.dict_point.values["nested"].dict_point.values["move"].box_point.shape.extend(
        [1, 2]
    )
    point.dict_point.values["choice"].discrete_point.value = 3
    point.dict_point.values["flags"].multi_binary_point.values.extend([True, False, True])
    assert _decode_tree(_encode_point(point), space=False) == point


def test_multi_binary_space_and_point_roundtrip():
    space = _decode_tree(
        [dict(parent=-1, kind="multi_binary", shape=[4])],
        space=True,
    )
    assert space.multi_binary_space.shape == 4

    empty = _decode_tree(
        [dict(parent=-1, kind="multi_binary", shape=[0])],
        space=True,
    )
    assert empty.multi_binary_space.shape == 0

    point = points.Point()
    point.multi_binary_point.values.extend([False, True])
    assert _decode_tree(_encode_point(point), space=False) == point
    assert _encode_point(point) == [
        dict(parent=-1, key=None, kind="multi_binary", boolValues=[False, True])
    ]


@pytest.mark.parametrize(
    "nodes",
    [
        [],
        [dict(parent=1, kind="dict")],
        [dict(parent=-1, kind="dict"), dict(parent=10, key="x", kind="dict")],
        [
            dict(parent=-1, kind="dict"),
            dict(parent=0, key="x", kind="dict"),
            dict(parent=0, key="x", kind="dict"),
        ],
        [dict(parent=-1, kind="box", values=[1], shape=[2])],
        [dict(parent=-1, kind="box", values=[float("nan")])],
        [dict(parent=-1, kind="multi_binary", boolValues=[1, 0])],
    ],
)
def test_malformed_tree_is_rejected(nodes):
    with pytest.raises(ValueError):
        _decode_tree(nodes, space=False)


@pytest.mark.parametrize(
    "nodes",
    [
        [dict(parent=-1, kind="multi_binary", shape=[-1])],
        [dict(parent=-1, kind="multi_binary", shape=[])],
        [dict(parent=-1, kind="multi_binary", shape=3)],
    ],
)
def test_malformed_multi_binary_space_is_rejected(nodes):
    with pytest.raises(ValueError):
        _decode_tree(nodes, space=True)


def test_space_keeps_unbounded_dimensions_and_float32_shape():
    result = _decode_tree(
        [
            dict(
                parent=-1,
                kind="box",
                shape=[2],
                dimensions=[{}, dict(hasLow=True, low=-1, hasHigh=True, high=1)],
            )
        ],
        space=True,
    )
    assert not result.box_space.dimensions[0].HasField("low")
    assert result.box_space.dimensions[1].high == 1
    assert list(result.box_space.shape_dimensions) == [2]
    assert result.box_space.dtype == 0


@pytest.fixture
def endpoint():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(5)
    with ThreadPoolExecutor(max_workers=1) as pool:
        yield listener, pool
    listener.close()


def test_real_grpc_adapter_maps_reset_step_and_same_step_initial_state(endpoint):
    listener, pool = endpoint
    commands = []

    def unity_peer():
        with listener.accept()[0] as peer:
            peer.settimeout(5)
            while True:
                header = UnityConnection._read_exact(peer, 4)
                size = struct.unpack("!I", header)[0]
                request = json.loads(UnityConnection._read_exact(peer, size))
                commands.append(request)
                command = request["command"]
                response = dict(version=1, ok=True)
                if command == "definition":
                    space = [
                        dict(
                            parent=-1,
                            kind="box",
                            dimensions=[
                                dict(hasLow=True, low=-1, hasHigh=True, high=1)
                            ],
                        )
                    ]
                    response.update(observationSpace=space, actionSpace=space)
                elif command == "reset":
                    response.update(observations=box(), info=[])
                elif command == "step":
                    response.update(
                        observations=box(0.5),
                        reward=1.5,
                        terminated=True,
                        truncated=False,
                        info=[dict(key="reason", value="goal")],
                        hasInitial=True,
                        initialObservations=box(),
                        initialInfo=[],
                    )
                data = json.dumps(response).encode()
                # Send fragments to verify framing handles partial reads.
                frame = struct.pack("!I", len(data)) + data
                for offset in range(0, len(frame), 7):
                    peer.sendall(frame[offset : offset + 7])
                if command == "close":
                    break

    future = pool.submit(unity_peer)
    with UnityBridgeServer(listener.getsockname()[1], port=0) as adapter:
        with grpc.secure_channel(
            f"127.0.0.1:{adapter.port}", grpc.local_channel_credentials()
        ) as channel:
            stub = services.GymServiceStub(channel)
            stub.StartGymConnector(connector.GymConnectorStartRequest(), timeout=5)
            definition = stub.RequestTrainingDefinition(
                connector.TrainingDefinitionRequest(), timeout=5
            )
            assert (
                definition.environment_definitions[0]
                .agent_definitions["SingleAgent"]
                .action_space.box_space.dimensions[0]
                .high
                == 1
            )
            reset = updates.StateUpdate(reset=updates.Reset())
            reset.reset.environments[0].seed = 42
            reset.reset.environments[0].options["difficulty"] = "easy"
            initial = stub.UpdateState(reset, timeout=5)
            assert initial.HasField("initial_state")
            step = updates.StateUpdate(step=updates.Step())
            step.step.environments.add().updates["SingleAgent"].box_point.values.append(
                0.5
            )
            state = stub.UpdateState(step, timeout=5)
            agent = state.training_state.environment_states[0].agent_states[
                "SingleAgent"
            ]
            assert agent.reward == 1.5 and agent.terminated and not agent.truncated
            assert agent.info["reason"] == "goal"
            assert agent.observations.box_point.values[0] == 0.5
            assert (
                state.initial_state.environment_states[0]
                .agent_states["SingleAgent"]
                .observations.box_point.values[0]
                == 0
            )
            # Bad IDs never reach Unity and leave the session usable.
            bad = updates.StateUpdate(step=updates.Step())
            bad.step.environments.add().updates["unknown"].discrete_point.value = 0
            with pytest.raises(grpc.RpcError) as error:
                stub.UpdateState(bad, timeout=5)
            assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
            stub.UpdateState(updates.StateUpdate(status=updates.CLOSED), timeout=5)
    future.result(timeout=5)
    assert [item["command"] for item in commands] == [
        "start",
        "definition",
        "reset",
        "step",
        "close",
    ]
    assert commands[2]["hasSeed"] and commands[2]["seed"] == 42
    assert commands[2]["options"] == [dict(key="difficulty", value="easy")]


@pytest.mark.parametrize("failure", ["length", "version", "unity", "timeout"])
def test_transport_failures_are_bounded_and_reported(endpoint, failure):
    listener, pool = endpoint

    def peer():
        with listener.accept()[0] as stream:
            stream.recv(4096)
            if failure == "timeout":
                # Wait until the client times out and closes, without a sleep.
                stream.recv(1)
                return
            if failure == "length":
                stream.sendall(struct.pack("!I", MAX_MESSAGE_BYTES + 1))
                return
            response = dict(
                version=2 if failure == "version" else 1,
                ok=False,
                error="reset required",
            )
            data = json.dumps(response).encode()
            stream.sendall(struct.pack("!I", len(data)) + data)

    future = pool.submit(peer)
    connection = UnityConnection(listener.getsockname()[1], timeout=0.2)
    expected = (
        UnityError
        if failure == "unity"
        else (TimeoutError if failure == "timeout" else ConnectionError)
    )
    try:
        with pytest.raises(expected):
            connection.request("step")
    finally:
        connection.close()
    future.result(timeout=5)
