# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Optional integration tests against TrainingIntegrationHost in a real Unity Editor.

Start the batch host described in Documentation~/training.md, then set
SCHOLA_TEST_UNITY_PORT. Without that explicit opt-in these tests are skipped.
Set SCHOLA_TEST_UNITY_AGENTS=2 on the host for the multi-agent live cases.
"""

import os

import numpy as np
import pytest

from schola.core.protocols.base_protocol import AutoResetType
from schola.core.protocols.protobuf.grpc_protocol import GrpcProtocol
from schola.core.simulators.external_simulator import ExternalSimulator
from schola.sb3.env import VecEnv
from schola.sb3.utils import VecMergeDictActionWrapper
from schola.unity.bridge import UnityBridgeServer

pytestmark = pytest.mark.skipif(
    not os.environ.get("SCHOLA_TEST_UNITY_PORT"), reason="Real Unity host not requested"
)

_MULTI_AGENT_HOST = os.environ.get("SCHOLA_TEST_UNITY_AGENTS") == "2"


@pytest.fixture
def adapter():
    with UnityBridgeServer(int(os.environ["SCHOLA_TEST_UNITY_PORT"]), port=0) as server:
        yield server


@pytest.mark.skipif(_MULTI_AGENT_HOST, reason="Single-agent Unity host required")
def test_real_unity_discovery_actions_terminal_observation_and_reset(adapter):
    env = VecMergeDictActionWrapper(
        VecEnv(ExternalSimulator(), GrpcProtocol("127.0.0.1", adapter.port))
    )
    try:
        obs = env.reset()
        np.testing.assert_array_equal(obs["position"], [[0, 0, 0]])
        assert env.action_space.shape == (3,)
        np.testing.assert_allclose(env.action_space.low, [-0.1] * 3)
        obs, reward, done, info = env.step(np.array([[0.1, 0, 0]], dtype=np.float32))
        np.testing.assert_allclose(obs["position"], [[0.1, 0, 0]])
        assert not done[0]
        obs, reward, done, info = env.step(np.array([[0.1, 0, 0]], dtype=np.float32))
        assert done[0] and not info[0]["TimeLimit.truncated"]
        np.testing.assert_allclose(
            info[0]["terminal_observation"]["position"], [0.2, 0, 0]
        )
        np.testing.assert_array_equal(obs["position"], [[0, 0, 0]])
        # A different episode truncates instead of terminating, preserving that distinction.
        for _ in range(4):
            obs, reward, done, info = env.step(np.zeros((1, 3), dtype=np.float32))
        assert done[0] and info[0]["TimeLimit.truncated"]
    finally:
        env.close()


@pytest.mark.skipif(_MULTI_AGENT_HOST, reason="Single-agent Unity host required")
@pytest.mark.parametrize("mode", [AutoResetType.NEXT_STEP, AutoResetType.DISABLED])
def test_real_unity_other_reset_modes(adapter, mode):
    protocol = GrpcProtocol("127.0.0.1", adapter.port)
    protocol.start()
    try:
        protocol.send_startup_msg(mode)
        _, _, _, spaces = protocol.get_definition()
        protocol.send_reset_msg()
        action = {
            0: {"SingleAgent": {"translate": np.array([0.1, 0, 0], dtype=np.float32)}}
        }
        action_spaces = spaces[0]
        protocol.send_action_msg(action, action_spaces)
        final = protocol.send_action_msg(action, action_spaces)
        assert final[2][0]["SingleAgent"]
        if mode == AutoResetType.NEXT_STEP:
            reset = protocol.send_action_msg(action, action_spaces)
            assert not reset[2][0]["SingleAgent"]
            np.testing.assert_array_equal(
                reset[0][0]["SingleAgent"]["position"], [0, 0, 0]
            )
        else:
            import grpc

            with pytest.raises(grpc.RpcError):
                protocol.send_action_msg(action, action_spaces)
            protocol.send_reset_msg()
    finally:
        protocol.close()


@pytest.mark.skipif(_MULTI_AGENT_HOST, reason="Single-agent Unity host required")
def test_real_unity_ppo_training_saves_reloadable_checkpoint(adapter, tmp_path):
    from stable_baselines3 import PPO

    env = VecMergeDictActionWrapper(
        VecEnv(ExternalSimulator(), GrpcProtocol("127.0.0.1", adapter.port))
    )
    try:
        model = PPO(
            "MultiInputPolicy",
            env,
            n_steps=16,
            batch_size=16,
            n_epochs=1,
            seed=7,
            device="cpu",
        )
        model.learn(total_timesteps=32)
        assert model.num_timesteps == 32
        checkpoint = tmp_path / "unity_ppo.zip"
        model.save(checkpoint)
        loaded = PPO.load(checkpoint, env=env)
        action, _ = loaded.predict(env.reset(), deterministic=True)
        assert action.shape == (1, 3) and np.isfinite(action).all()
        env.step(action)
    finally:
        env.close()


@pytest.mark.skipif(not _MULTI_AGENT_HOST, reason="Two-agent Unity host not requested")
def test_real_unity_multi_agent_shared_policy_slots(adapter):
    env = VecMergeDictActionWrapper(
        VecEnv(ExternalSimulator(), GrpcProtocol("127.0.0.1", adapter.port))
    )
    try:
        assert env.num_envs == 2
        obs = env.reset()
        np.testing.assert_array_equal(obs["position"], [[0, 0, 0], [0, 0, 1]])
        actions = np.array([[0.1, 0, 0], [0.1, 0, 0]], dtype=np.float32)
        obs, reward, done, info = env.step(actions)
        np.testing.assert_allclose(obs["position"], [[0.1, 0, 0], [0.1, 0, 1]])
        assert not done.any()
        obs, reward, done, info = env.step(actions)
        assert done.all()
        np.testing.assert_array_equal(obs["position"], [[0, 0, 0], [0, 0, 1]])
    finally:
        env.close()


@pytest.mark.skipif(not _MULTI_AGENT_HOST, reason="Two-agent Unity host not requested")
@pytest.mark.parametrize("mode", [AutoResetType.NEXT_STEP, AutoResetType.DISABLED])
def test_real_unity_multi_agent_other_reset_modes(adapter, mode):
    protocol = GrpcProtocol("127.0.0.1", adapter.port)
    protocol.start()
    try:
        protocol.send_startup_msg(mode)
        ids, _, _, spaces = protocol.get_definition()
        assert ids == [["agent_0", "agent_1"]]
        protocol.send_reset_msg()
        action = {
            0: {
                "agent_0": {"translate": np.array([0.1, 0, 0], dtype=np.float32)},
                "agent_1": {"translate": np.array([0.1, 0, 0], dtype=np.float32)},
            }
        }
        action_spaces = spaces[0]
        protocol.send_action_msg(action, action_spaces)
        final = protocol.send_action_msg(action, action_spaces)
        assert final[2][0]["agent_0"] and final[2][0]["agent_1"]
        if mode == AutoResetType.NEXT_STEP:
            reset = protocol.send_action_msg(action, action_spaces)
            assert not reset[2][0]["agent_0"] and not reset[2][0]["agent_1"]
            np.testing.assert_array_equal(
                reset[0][0]["agent_0"]["position"], [0, 0, 0]
            )
            np.testing.assert_array_equal(
                reset[0][0]["agent_1"]["position"], [0, 0, 1]
            )
        else:
            import grpc

            with pytest.raises(grpc.RpcError):
                protocol.send_action_msg(action, action_spaces)
            protocol.send_reset_msg()
    finally:
        protocol.close()
