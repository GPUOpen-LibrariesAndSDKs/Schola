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
    VERSION,
    UnityBridgeServer,
    UnityConnection,
    UnityError,
    _decode_tree,
    _encode_point,
)


def box(value=0):
    return [dict(parent=-1, kind="box", values=[value], shape=[1])]


def box_space(low=-1, high=1):
    return [
        dict(
            parent=-1,
            kind="box",
            dimensions=[dict(hasLow=True, low=low, hasHigh=True, high=high)],
        )
    ]


def agent_definition(agent_id, low=-1, high=1):
    space = box_space(low, high)
    return dict(id=agent_id, observationSpace=space, actionSpace=space)


def agent_state(
    agent_id,
    value=0,
    reward=0,
    terminated=False,
    truncated=False,
    info=None,
):
    return dict(
        id=agent_id,
        observations=box(value),
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=info or [],
    )


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
        [dict(parent=-1, kind="multi_binary", shape=[0])],
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


def _serve(endpoint, handler):
    listener, pool = endpoint

    def unity_peer():
        with listener.accept()[0] as peer:
            peer.settimeout(5)
            while True:
                header = UnityConnection._read_exact(peer, 4)
                size = struct.unpack("!I", header)[0]
                request = json.loads(UnityConnection._read_exact(peer, size))
                response = handler(request)
                data = json.dumps(response).encode()
                # Send fragments to verify framing handles partial reads.
                frame = struct.pack("!I", len(data)) + data
                for offset in range(0, len(frame), 7):
                    peer.sendall(frame[offset : offset + 7])
                if request["command"] == "close":
                    break

    return pool.submit(unity_peer), listener.getsockname()[1]


def test_real_grpc_adapter_maps_reset_step_and_same_step_initial_state(endpoint):
    commands = []

    def handler(request):
        commands.append(request)
        command = request["command"]
        response = dict(version=VERSION, ok=True)
        if command == "definition":
            response["agents"] = [agent_definition("SingleAgent")]
        elif command == "reset":
            response["states"] = [agent_state("SingleAgent")]
        elif command == "step":
            response.update(
                states=[
                    agent_state(
                        "SingleAgent",
                        value=0.5,
                        reward=1.5,
                        terminated=True,
                        info=[dict(key="reason", value="goal")],
                    )
                ],
                hasInitial=True,
                initialStates=[agent_state("SingleAgent")],
            )
        return response

    future, port = _serve(endpoint, handler)
    with UnityBridgeServer(port, port=0) as adapter:
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
    assert [entry["id"] for entry in commands[3]["actions"]] == ["SingleAgent"]
    assert commands[3]["actions"][0]["action"][0]["values"] == [0.5]


def test_multi_agent_definition_reset_step_and_partial_actions(endpoint):
    commands = []

    def handler(request):
        commands.append(request)
        command = request["command"]
        response = dict(version=VERSION, ok=True)
        if command == "definition":
            response["agents"] = [
                agent_definition("agent_0"),
                agent_definition("agent_1", low=-2, high=2),
            ]
        elif command == "reset":
            response["states"] = [
                agent_state("agent_0"),
                agent_state("agent_1"),
            ]
        elif command == "step":
            response.update(
                states=[
                    agent_state("agent_0", value=0.5, reward=1, terminated=True),
                    agent_state("agent_1", value=1.5, reward=2, truncated=True),
                ],
                hasInitial=True,
                initialStates=[
                    agent_state("agent_0"),
                    agent_state("agent_1"),
                ],
            )
        return response

    future, port = _serve(endpoint, handler)
    with UnityBridgeServer(port, port=0) as adapter:
        with grpc.secure_channel(
            f"127.0.0.1:{adapter.port}", grpc.local_channel_credentials()
        ) as channel:
            stub = services.GymServiceStub(channel)
            stub.StartGymConnector(connector.GymConnectorStartRequest(), timeout=5)
            definition = stub.RequestTrainingDefinition(
                connector.TrainingDefinitionRequest(), timeout=5
            )
            agents = definition.environment_definitions[0].agent_definitions
            assert set(agents) == {"agent_0", "agent_1"}
            assert agents["agent_1"].action_space.box_space.dimensions[0].high == 2

            initial = stub.UpdateState(updates.StateUpdate(reset=updates.Reset()), timeout=5)
            assert set(initial.initial_state.environment_states[0].agent_states) == {
                "agent_0",
                "agent_1",
            }

            # RLlib may omit a finished agent; a subset must still reach Unity.
            step = updates.StateUpdate(step=updates.Step())
            env = step.step.environments.add()
            env.updates["agent_1"].box_point.values.append(1.5)
            state = stub.UpdateState(step, timeout=5)
            training = state.training_state.environment_states[0].agent_states
            assert training["agent_0"].terminated
            assert training["agent_1"].truncated
            assert set(state.initial_state.environment_states[0].agent_states) == {
                "agent_0",
                "agent_1",
            }

            bad = updates.StateUpdate(step=updates.Step())
            bad.step.environments.add().updates["agent_9"].box_point.values.append(0)
            with pytest.raises(grpc.RpcError) as error:
                stub.UpdateState(bad, timeout=5)
            assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
            stub.UpdateState(updates.StateUpdate(status=updates.CLOSED), timeout=5)
    future.result(timeout=5)
    assert [entry["id"] for entry in commands[3]["actions"]] == ["agent_1"]
    assert commands[3]["actions"][0]["action"][0]["values"] == [1.5]


def test_multi_agent_vecenv_sees_one_slot_per_agent(endpoint):
    pytest.importorskip("stable_baselines3")
    from schola.core.protocols.protobuf.grpc_protocol import GrpcProtocol
    from schola.core.simulators.external_simulator import ExternalSimulator
    from schola.sb3.env import VecEnv

    def handler(request):
        command = request["command"]
        response = dict(version=VERSION, ok=True)
        if command == "definition":
            response["agents"] = [
                agent_definition("agent_0"),
                agent_definition("agent_1"),
            ]
        elif command == "reset":
            response["states"] = [
                agent_state("agent_0"),
                agent_state("agent_1"),
            ]
        elif command == "step":
            response["states"] = [
                agent_state("agent_0", value=0.1, reward=1),
                agent_state("agent_1", value=0.2, reward=2),
            ]
        return response

    future, port = _serve(endpoint, handler)
    with UnityBridgeServer(port, port=0) as adapter:
        env = VecEnv(
            ExternalSimulator(), GrpcProtocol("127.0.0.1", adapter.port)
        )
        try:
            assert env.num_envs == 2
            obs = env.reset()
            assert obs.shape == (2, 1)
        finally:
            env.close()
    future.result(timeout=5)


def test_multi_agent_rayenv_uses_agent_ids(endpoint):
    pytest.importorskip("ray")
    from schola.core.protocols.protobuf.grpc_protocol import GrpcProtocol
    from schola.core.simulators.external_simulator import ExternalSimulator
    from schola.rllib.env import RayEnv

    def handler(request):
        command = request["command"]
        response = dict(version=VERSION, ok=True)
        if command == "definition":
            response["agents"] = [
                agent_definition("agent_0"),
                agent_definition("agent_1"),
            ]
        elif command == "reset":
            response["states"] = [
                agent_state("agent_0"),
                agent_state("agent_1"),
            ]
        elif command == "step":
            response["states"] = [
                agent_state("agent_0", value=0.1, reward=1),
                agent_state("agent_1", value=0.2, reward=2),
            ]
        return response

    future, port = _serve(endpoint, handler)
    with UnityBridgeServer(port, port=0) as adapter:
        env = RayEnv(GrpcProtocol("127.0.0.1", adapter.port), ExternalSimulator())
        try:
            obs, _ = env.reset()
            assert set(obs) == {"agent_0", "agent_1"}
        finally:
            env.close()
    future.result(timeout=5)


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
                version=1 if failure == "version" else VERSION,
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


def test_version_mismatch_names_both_sides(endpoint):
    listener, pool = endpoint

    def peer():
        with listener.accept()[0] as stream:
            stream.recv(4096)
            response = dict(version=1, ok=True)
            data = json.dumps(response).encode()
            stream.sendall(struct.pack("!I", len(data)) + data)

    future = pool.submit(peer)
    connection = UnityConnection(listener.getsockname()[1], timeout=0.2)
    try:
        with pytest.raises(ConnectionError) as error:
            connection.request("start")
        assert "version 1" in str(error.value)
        assert f"version {VERSION}" in str(error.value)
    finally:
        connection.close()
    future.result(timeout=5)
