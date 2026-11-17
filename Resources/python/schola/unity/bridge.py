# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Expose a Unity TrainingServer through Schola's existing GymService contract.

The adapter owns a loopback-only, length-prefixed JSON connection to Unity. It
uses the existing protobuf schema and gRPC clients without native Unity plugins.
One adapter serves one environment with one agent, ``SingleAgent``.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import math
import socket
import struct
import threading
import time
from typing import Any

import grpc

from schola.generated import Definitions_pb2 as definitions
from schola.generated import GymConnector_pb2 as connector
from schola.generated import GymConnector_pb2_grpc as services
from schola.generated import Points_pb2 as points
from schola.generated import Spaces_pb2 as spaces
from schola.generated import State_pb2 as states
from schola.generated import StateUpdates_pb2 as updates

AGENT_ID = "SingleAgent"
VERSION = 1
MAX_MESSAGE_BYTES = 4 * 1024 * 1024


class UnityError(RuntimeError):
    """Unity rejected an operation; reset the environment before stepping again."""


def _shape(node: dict, count: int) -> list[int]:
    shape = node.get("shape") or [count]
    if not shape or any(type(size) is not int or size <= 0 for size in shape):
        raise ValueError("Invalid Box shape")
    if math.prod(shape) != count:
        raise ValueError("Box shape does not match its data")
    return shape


def _decode_tree(nodes: list[dict], *, space: bool):
    if not nodes or len(nodes) > 10000 or nodes[0].get("parent") != -1:
        raise ValueError("Invalid Unity value tree")
    result = []
    children: dict[int, set[str]] = {}
    for index, node in enumerate(nodes):
        value = spaces.Space() if space else points.Point()
        kind = node["kind"]
        if kind == "dict":
            (value.dict_space if space else value.dict_point).SetInParent()
        elif kind == "box":
            if space:
                box = value.box_space
                dimensions = node.get("dimensions") or []
                box.shape_dimensions.extend(_shape(node, len(dimensions)))
                for item in dimensions:
                    dimension = box.dimensions.add()
                    if item.get("hasLow"):
                        dimension.low = item["low"]
                    if item.get("hasHigh"):
                        dimension.high = item["high"]
            else:
                box = value.box_point
                data = node.get("values") or []
                if not all(math.isfinite(x) for x in data):
                    raise ValueError("Non-finite Unity Box value")
                box.values.extend(data)
                box.shape.extend(_shape(node, len(data)))
            box.dtype = 0  # FLOAT32, matching Unity's float-based Box types.
        elif kind == "discrete":
            if space:
                if node["high"] <= 0:
                    raise ValueError("Discrete space must have positive size")
                value.discrete_space.high = node["high"]
            else:
                value.discrete_point.value = node["value"]
        elif kind == "multi_binary":
            if space:
                shape = node.get("shape")
                if (
                    not isinstance(shape, list)
                    or len(shape) != 1
                    or type(shape[0]) is not int
                    or shape[0] <= 0
                ):
                    raise ValueError("Invalid MultiBinary shape")
                value.multi_binary_space.shape = shape[0]
            else:
                data = node.get("boolValues") or []
                if not isinstance(data, list) or not all(type(flag) is bool for flag in data):
                    raise ValueError("MultiBinary values must be booleans")
                value.multi_binary_point.values.extend(data)
        else:
            raise ValueError(f"Unsupported Unity type: {kind}")
        result.append(value)
        if index:
            parent = node["parent"]
            if type(parent) is not int or not 0 <= parent < index:
                raise ValueError("Invalid Unity tree parent")
            if nodes[parent]["kind"] != "dict":
                raise ValueError("Unity tree parent must be a Dict")
            key = node["key"]
            if not isinstance(key, str) or key in children.setdefault(parent, set()):
                raise ValueError("Invalid or duplicate Unity key")
            children[parent].add(key)
    # Attach children in reverse order so each copied protobuf subtree is complete.
    for index in range(len(nodes) - 1, 0, -1):
        node = nodes[index]
        parent = result[node["parent"]]
        target = parent.dict_space.spaces if space else parent.dict_point.values
        target[node["key"]].CopyFrom(result[index])
    return result[0]


def _encode_point(point: points.Point) -> list[dict]:
    nodes = []

    def visit(value, parent=-1, key=None, depth=0):
        if depth > 32 or len(nodes) >= 10000:
            raise ValueError("Action tree is too deep or too large")
        node = {"parent": parent, "key": key}
        index = len(nodes)
        nodes.append(node)
        kind = value.WhichOneof("point")
        if kind == "dict_point":
            node["kind"] = "dict"
            for child_key in sorted(value.dict_point.values):
                visit(value.dict_point.values[child_key], index, child_key, depth + 1)
        elif kind == "box_point":
            if value.box_point.dtype != 0:
                raise ValueError("Unity Box actions require FLOAT32")
            data = list(value.box_point.values)
            if not all(math.isfinite(x) for x in data):
                raise ValueError("Actions must be finite")
            node.update(kind="box", values=data, shape=list(value.box_point.shape))
            _shape(node, len(data))
        elif kind == "discrete_point":
            node.update(kind="discrete", value=value.discrete_point.value)
        elif kind == "multi_binary_point":
            node.update(
                kind="multi_binary",
                boolValues=list(value.multi_binary_point.values),
            )
        else:
            raise ValueError(f"Unsupported Unity action type: {kind}")

    visit(point)
    return nodes


def _info(entries) -> dict[str, str]:
    return {entry["key"]: entry["value"] for entry in entries or []}


class UnityConnection:
    """Serialized local transport. A failed exchange closes the stream to prevent stale replies."""

    def __init__(self, port: int = 8001, timeout: float = 30):
        if not 1 <= port <= 65535 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Invalid Unity port or timeout")
        self.port = port
        self.timeout = timeout
        self._socket: socket.socket | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _read_exact(
        stream: socket.socket, size: int, deadline: float | None = None
    ) -> bytes:
        data = bytearray()
        while len(data) < size:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Unity request deadline expired")
                stream.settimeout(remaining)
            chunk = stream.recv(size - len(data))
            if not chunk:
                raise ConnectionError("Unity disconnected")
            data.extend(chunk)
        return bytes(data)

    def request(self, command: str, *, timeout: float | None = None, **kwargs) -> dict:
        timeout = min(self.timeout, timeout) if timeout is not None else self.timeout
        deadline = time.monotonic() + timeout
        if timeout <= 0 or not self._lock.acquire(timeout=timeout):
            raise TimeoutError("Unity request deadline expired")
        try:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                raise TimeoutError("Unity request deadline expired")
            if self._socket is None:
                self._socket = socket.create_connection(
                    ("127.0.0.1", self.port), timeout
                )
                self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._socket.settimeout(timeout)
            data = json.dumps(
                dict(version=VERSION, command=command, **kwargs), allow_nan=False
            ).encode("utf-8")
            if len(data) > MAX_MESSAGE_BYTES:
                raise ValueError("Unity request is too large")
            self._socket.sendall(struct.pack("!I", len(data)) + data)
            (size,) = struct.unpack("!I", self._read_exact(self._socket, 4, deadline))
            if not 0 < size <= MAX_MESSAGE_BYTES:
                raise ConnectionError("Invalid Unity response length")
            response = json.loads(self._read_exact(self._socket, size, deadline))
            if response.get("version") != VERSION:
                raise ConnectionError("Unsupported Unity training protocol version")
            if not response.get("ok"):
                raise UnityError(response.get("error") or "Unity rejected the request")
            return response
        except UnityError:
            raise
        except Exception:
            self._disconnect()
            raise
        finally:
            self._lock.release()

    def _disconnect(self):
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def close(self):
        with self._lock:
            self._disconnect()


class UnityGymService(services.GymServiceServicer):
    """Translate the existing three Schola RPCs to Unity's main-thread session."""

    def __init__(self, connection: UnityConnection):
        self.connection = connection

    def _request(self, context, command, **kwargs):
        try:
            return self.connection.request(
                command, timeout=context.time_remaining(), **kwargs
            )
        except TimeoutError as error:
            context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, str(error))
        except UnityError as error:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
        except (ValueError, KeyError, TypeError) as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        except (OSError, ConnectionError) as error:
            context.abort(grpc.StatusCode.UNAVAILABLE, str(error))

    def StartGymConnector(self, request, context):
        self._request(context, "start", autoreset=request.autoreset_type)
        return connector.GymConnectorStartResponse()

    def RequestTrainingDefinition(self, request, context):
        response = self._request(context, "definition")
        definition = definitions.TrainingDefinition()
        agent = definition.environment_definitions.add().agent_definitions[AGENT_ID]
        agent.agent_type = "Unity"
        agent.obs_space.CopyFrom(_decode_tree(response["observationSpace"], space=True))
        agent.action_space.CopyFrom(_decode_tree(response["actionSpace"], space=True))
        return definition

    @staticmethod
    def _initial(target, observation, info):
        agent = target.agent_states[AGENT_ID]
        agent.observations.CopyFrom(_decode_tree(observation, space=False))
        agent.info.update(_info(info))

    def UpdateState(self, request, context):
        if request.status == updates.CLOSED:
            self._request(context, "close")
            return states.State()
        if request.status != updates.GOOD:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "Unsupported communicator status"
            )
        kind = request.WhichOneof("update")
        result = states.State()
        if kind == "reset":
            if any(env_id != 0 for env_id in request.reset.environments):
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT, "Unity exposes environment 0 only"
                )
            settings = request.reset.environments.get(0)
            kwargs: dict[str, Any] = {}
            if settings is not None:
                kwargs["hasSeed"] = settings.HasField("seed")
                kwargs["seed"] = settings.seed
                kwargs["options"] = [
                    dict(key=key, value=value)
                    for key, value in settings.options.items()
                ]
            response = self._request(context, "reset", **kwargs)
            self._initial(
                result.initial_state.environment_states[0],
                response["observations"],
                response.get("info"),
            )
        elif kind == "step":
            if len(request.step.environments) != 1 or set(
                request.step.environments[0].updates
            ) != {AGENT_ID}:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Expected environment 0 / SingleAgent",
                )
            try:
                action = _encode_point(request.step.environments[0].updates[AGENT_ID])
            except ValueError as error:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            response = self._request(context, "step", action=action)
            agent = result.training_state.environment_states.add().agent_states[
                AGENT_ID
            ]
            agent.observations.CopyFrom(
                _decode_tree(response["observations"], space=False)
            )
            agent.reward = response["reward"]
            agent.terminated = response["terminated"]
            agent.truncated = response["truncated"]
            agent.info.update(_info(response.get("info")))
            if response.get("hasInitial"):
                self._initial(
                    result.initial_state.environment_states[0],
                    response["initialObservations"],
                    response.get("initialInfo"),
                )
        else:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Expected reset or step")
        return result


class UnityBridgeServer:
    """Context-managed gRPC adapter bound to loopback with local credentials.

    Parameters
    ----------
    unity_port : int
        Port of the Unity TrainingServer component.
    port : int
        Python gRPC listening port; zero chooses an available port.
    timeout : float
        Maximum time to wait for Unity per request, in seconds.
    """

    def __init__(self, unity_port: int = 8001, port: int = 8000, timeout: float = 30):
        if not 0 <= port <= 65535:
            raise ValueError("Invalid gRPC port")
        self.connection = UnityConnection(unity_port, timeout)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.server = grpc.server(
            self._executor,
            options=[
                ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
                ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
            ],
        )
        services.add_GymServiceServicer_to_server(
            UnityGymService(self.connection), self.server
        )
        self.port = self.server.add_secure_port(
            f"127.0.0.1:{port}", grpc.local_server_credentials()
        )
        if not self.port:
            self._executor.shutdown(wait=False)
            raise OSError("Could not bind Unity gRPC adapter")

    def __enter__(self):
        self.server.start()
        return self

    def __exit__(self, *_):
        self.server.stop(0).wait()
        self._executor.shutdown(wait=True)
        self.connection.close()
