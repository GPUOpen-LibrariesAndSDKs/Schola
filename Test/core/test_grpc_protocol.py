# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""gRPC protocol message layout: env id is the repeated-field index."""

import numpy as np
from gymnasium.spaces import Discrete

from schola.core.protocols.protobuf.deserialize import from_proto
from schola.core.protocols.protobuf.grpc_protocol import GrpcProtocol
from schola.generated.State_pb2 import State


def test_prepare_action_msg():
    """Slot ``i`` of ``step.environments`` is environment ``i``, not dict order."""
    protocol = GrpcProtocol("localhost", 0)
    msg = protocol.prepare_action_msg(
        {1: {"a": np.int64(4)}, 0: {"a": np.int64(7)}},
        {"a": Discrete(8)},
    )
    actions = [int(from_proto(env.updates["a"])) for env in msg.step.environments]
    assert actions[0] == 7 and actions[1] == 4


def test_send_reset_msg(monkeypatch):
    """Observations come back in environment-id order, including sparse map keys."""
    response = State()
    for env_id, value in ((0, 5), (1, 9)):
        agent_state = response.initial_state.environment_states[env_id].agent_states[
            "a"
        ]
        agent_state.observations.discrete_point.value = value

    protocol = GrpcProtocol("localhost", 0)

    class _Stub:
        def UpdateState(self, _state_update):
            return response

    monkeypatch.setattr(GrpcProtocol, "gym_stub", property(lambda _self: _Stub()))

    observations, _infos = protocol.send_reset_msg()
    assert observations[0]["a"] == 5
    assert observations[1]["a"] == 9
