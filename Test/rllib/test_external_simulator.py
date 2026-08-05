# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Tests for the ExternalSimulator and the Kubernetes-oriented distributed
workflow.  These tests simulate the real-life scenario where:

  - The UE game server is an *external* process (like a K8s pod) that is
    already running before training starts.
  - The Python side connects to it via gRPC without launching a subprocess.
  - Multiple "workers" each talk to their own gRPC server on a fixed port
    (different-pod topology — every pod uses the same port).
  - URL templates (``{worker_index}``) allow per-worker connection routing
    through the config without environment variable overrides.

No actual Unreal Engine is required — we use the existing mock gRPC
servers (``GymToGymServiceServicer`` / ``VecGymToGymServiceServicer``)
from the test infrastructure.
"""

import pytest

from schola.core.simulators.external_simulator import ExternalSimulator
from schola.core.simulators.unreal.executable_simulator import UnrealExecutable
from schola.core.protocols.protobuf.grpc_protocol import GrpcProtocol
from schola.rllib.env import RayEnv, RayVecEnv

# ---------------------------------------------------------------------------
# ExternalSimulator unit tests
# ---------------------------------------------------------------------------


class TestExternalSimulator:

    def test_start_stop_are_noop(self):
        sim = ExternalSimulator()
        sim.start({"Port": 8000})
        sim.stop()

    def test_bool_always_true(self):
        sim = ExternalSimulator()
        assert bool(sim) is True

    def test_supports_grpc_protocol(self):
        sim = ExternalSimulator()
        assert GrpcProtocol in sim.supported_protocols

    def test_readiness_timeout_stored(self):
        sim = ExternalSimulator(readiness_timeout=30)
        assert sim.readiness_timeout == 30

    def test_get_simulator_args_roundtrip(self):
        """get_simulator_args should return kwargs that reproduce the instance."""
        sim = ExternalSimulator(readiness_timeout=60)
        args = sim.get_simulator_args()
        clone = ExternalSimulator(**args)
        assert clone.readiness_timeout == 60

    def test_get_simulator_args_empty_when_defaults(self):
        """Default-constructed instance produces an empty dict (no noise in env_config)."""
        sim = ExternalSimulator()
        assert sim.get_simulator_args() == {}


# ---------------------------------------------------------------------------
# ExternalSimulator + gRPC integration (simulates K8s external-pod scenario)
# ---------------------------------------------------------------------------


class TestExternalSimulatorWithGrpc:
    """Simulate connecting to a UE process that's already running externally."""

    def test_connect_insecure_single_env(self, make_env_server):
        """Connect to a single externally-managed env via insecure gRPC."""
        port = make_env_server("CartPole-v1")

        sim = ExternalSimulator()
        protocol = GrpcProtocol(url="localhost", port=port, credential_mode="insecure")
        env = RayEnv(protocol, sim)

        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert len(obs) > 0

        actions = {
            agent_id: env.single_action_spaces[agent_id].sample() for agent_id in obs
        }
        obs2, rewards, terms, truncs, infos = env.step(actions)
        assert isinstance(rewards, dict)
        assert "__all__" in terms

        env.close()

    def test_connect_insecure_vec_env(self, make_vec_env_server, make_env):
        """Connect to a vectorized externally-managed env."""
        envs = [make_env("CartPole-v1", i) for i in range(3)]
        port = make_vec_env_server(envs)

        sim = ExternalSimulator()
        protocol = GrpcProtocol(url="localhost", port=port, credential_mode="insecure")
        env = RayVecEnv(protocol, sim)

        assert env.num_envs == 3
        obs_list, info_list = env.reset()
        assert len(obs_list) == 3

        actions = [
            {agent_id: env.single_action_spaces[agent_id].sample() for agent_id in obs}
            for obs in obs_list
        ]
        obs_list2, rewards, terms, truncs, infos = env.step(actions)
        assert len(rewards) == 3

        env.close()

    def test_multiple_steps_produce_valid_data(self, make_env_server):
        """Run many steps to verify the connection stays healthy."""
        port = make_env_server("CartPole-v1")
        sim = ExternalSimulator()
        protocol = GrpcProtocol(url="localhost", port=port, credential_mode="insecure")
        env = RayEnv(protocol, sim)

        obs, _ = env.reset()
        for _ in range(50):
            actions = {aid: env.single_action_spaces[aid].sample() for aid in obs}
            obs, rewards, terms, truncs, infos = env.step(actions)
            if terms.get("__all__", False) or truncs.get("__all__", False):
                obs, _ = env.reset()

        env.close()


# ---------------------------------------------------------------------------
# Deferred path validation in UnrealExecutable
# ---------------------------------------------------------------------------


class TestDeferredPathValidation:

    def test_validate_path_false_skips_check(self, tmp_path):
        """With validate_path=False, a non-existent path is accepted at __init__."""
        fake_path = tmp_path / "does_not_exist.exe"
        sim = UnrealExecutable(executable_path=fake_path, validate_path=False)
        assert sim.executable_path == fake_path

    def test_validate_path_true_raises(self, tmp_path):
        """With validate_path=True (default), a non-existent path raises."""
        fake_path = tmp_path / "does_not_exist.exe"
        with pytest.raises(FileNotFoundError):
            UnrealExecutable(executable_path=fake_path, validate_path=True)

    def test_start_validates_path(self, tmp_path):
        """Even with validate_path=False, start() re-validates and raises."""
        fake_path = tmp_path / "does_not_exist.exe"
        sim = UnrealExecutable(executable_path=fake_path, validate_path=False)
        with pytest.raises(FileNotFoundError):
            sim.start({"Port": 8000})

    def test_get_spawn_args_sets_validate_false(self, tmp_path):
        """get_spawn_args() produces validate_path=False for serialization."""
        real_file = tmp_path / "game.exe"
        real_file.write_text("fake")
        sim = UnrealExecutable(executable_path=real_file)
        args = sim.get_spawn_args()
        assert args["validate_path"] is False


# ---------------------------------------------------------------------------
# gRPC credential modes
# ---------------------------------------------------------------------------


class TestCredentialModes:

    def test_local_credential_mode(self, make_env_server):
        """Default 'local' mode still works for same-machine connections."""
        port = make_env_server("CartPole-v1")
        sim = ExternalSimulator()
        protocol = GrpcProtocol(url="localhost", port=port, credential_mode="local")
        env = RayEnv(protocol, sim)
        obs, _ = env.reset()
        assert len(obs) > 0
        env.close()

    def test_insecure_credential_mode(self, make_env_server):
        """'insecure' mode connects without authentication."""
        port = make_env_server("CartPole-v1")
        sim = ExternalSimulator()
        protocol = GrpcProtocol(url="localhost", port=port, credential_mode="insecure")
        env = RayEnv(protocol, sim)
        obs, _ = env.reset()
        assert len(obs) > 0
        env.close()

    def test_invalid_credential_mode_raises(self):
        with pytest.raises(ValueError, match="credential_mode"):
            GrpcProtocol(url="localhost", port=8000, credential_mode="bogus") # type: ignore


# ---------------------------------------------------------------------------
# Client-only mode (client_only) — insecure channel regression tests
# ---------------------------------------------------------------------------


class TestClientOnlySocket:
    """Verify that insecure mode skips local socket creation entirely."""

    def test_insecure_mode_skips_socket_creation(self, make_env_server):
        """Insecure GrpcProtocol should not create a TCP socket."""
        port = make_env_server("CartPole-v1")
        protocol = GrpcProtocol(url="localhost", port=port, credential_mode="insecure")
        protocol.start()
        assert (
            protocol.tcp_socket is None
        ), "Insecure mode should not allocate a TCP socket"
        assert protocol.is_started
        assert bool(protocol) is True
        protocol.close()

    def test_local_mode_creates_socket(self, make_env_server):
        """Default local mode should still create a TCP socket."""
        port = make_env_server("CartPole-v1")
        protocol = GrpcProtocol(url="localhost", port=port, credential_mode="local")
        protocol.start()
        assert protocol.has_socket, "Local mode should allocate a TCP socket"
        protocol.close()

    def test_insecure_port_none_raises(self):
        """Insecure mode with no port must fail early with a clear message."""
        protocol = GrpcProtocol(
            url="ue-sim-svc.default.svc.cluster.local",
            port=None,
            credential_mode="insecure",
        )
        with pytest.raises(ValueError, match="explicit port is required"):
            protocol.start()

    def test_insecure_bool_false_after_close(self, make_env_server):
        """After close(), __bool__ should be False even in client-only mode."""
        port = make_env_server("CartPole-v1")
        protocol = GrpcProtocol(url="localhost", port=port, credential_mode="insecure")
        protocol.start()
        assert bool(protocol) is True
        protocol.close()
        assert bool(protocol) is False


# ---------------------------------------------------------------------------
# environment_start_timeout propagation — regression test
# ---------------------------------------------------------------------------


class TestTimeoutPropagation:

    def test_timeout_in_protocol_args(self):
        """env_config protocol_args must include environment_start_timeout."""
        from schola.scripts.common.settings import GrpcProtocolConfig

        cfg = GrpcProtocolConfig(port=8000, environment_start_timeout=120)
        protocol_args = {
            "url": cfg.url,
            "port": cfg.port,
            "credential_mode": cfg.credential_mode.value,
            "environment_start_timeout": cfg.environment_start_timeout,
        }
        assert protocol_args["environment_start_timeout"] == 120

        proto = GrpcProtocol(**protocol_args)
        assert proto.environment_start_timeout == 120

    def test_default_timeout_when_missing(self):
        """If timeout is not supplied, GrpcProtocol should use its default."""
        proto = GrpcProtocol(url="localhost", port=8000, credential_mode="insecure")
        assert proto.environment_start_timeout == 45


# ---------------------------------------------------------------------------
# resolve_protocol_args: port offset + URL template expansion
# ---------------------------------------------------------------------------


class TestResolveProtocolArgs:
    """
    Verify that resolve_protocol_args correctly applies port offsets and
    expands ``{worker_index}`` URL templates — the two mechanisms that
    replace the former environment-variable overrides.
    """

    def test_fixed_port_mode_no_offset(self, make_env_server):
        """With port_offset_mode='fixed', worker_index does NOT change the port."""
        from schola.rllib.env_runner import ScholaEnvRunner

        port = make_env_server("CartPole-v1")

        resolved = ScholaEnvRunner.resolve_protocol_args(
            protocol_args={
                "url": "localhost",
                "port": port,
                "credential_mode": "insecure",
            },
            port_offset_mode="fixed",
            worker_index=5,
        )

        assert resolved["url"] == "localhost"
        assert resolved["port"] == port
        assert resolved["credential_mode"] == "insecure"

    def test_per_worker_mode_adds_offset(self):
        """With port_offset_mode='per_worker', worker_index is added to port."""
        from schola.rllib.env_runner import ScholaEnvRunner

        base_port = 9000

        resolved = ScholaEnvRunner.resolve_protocol_args(
            protocol_args={"url": "localhost", "port": base_port},
            port_offset_mode="per_worker",
            worker_index=3,
        )

        assert resolved["url"] == "localhost"
        assert resolved["port"] == base_port + 3

    def test_url_template_expanded(self):
        """A URL containing {worker_index} is expanded with the worker index."""
        from schola.rllib.env_runner import ScholaEnvRunner

        resolved = ScholaEnvRunner.resolve_protocol_args(
            protocol_args={"url": "ue-{worker_index}", "port": 50051},
            port_offset_mode="fixed",
            worker_index=2,
        )
        assert resolved["url"] == "ue-2"
        assert resolved["port"] == 50051

    def test_url_template_unchanged_without_placeholder(self):
        """A plain URL without {worker_index} is left as-is."""
        from schola.rllib.env_runner import ScholaEnvRunner

        resolved = ScholaEnvRunner.resolve_protocol_args(
            protocol_args={"url": "localhost", "port": 50051},
            port_offset_mode="fixed",
            worker_index=1,
        )
        assert resolved["url"] == "localhost"

    def test_url_template_with_no_worker_index(self):
        """When worker_index is None, the template is not expanded."""
        from schola.rllib.env_runner import ScholaEnvRunner

        resolved = ScholaEnvRunner.resolve_protocol_args(
            protocol_args={"url": "ue-{worker_index}", "port": 50051},
            port_offset_mode="fixed",
            worker_index=None,
        )
        assert resolved["url"] == "ue-{worker_index}"


# ---------------------------------------------------------------------------
# K8s-like multi-pod simulation: multiple independent gRPC servers
# ---------------------------------------------------------------------------


class TestMultiPodSimulation:
    """
    Simulates the real K8s different-pod topology: each "worker pod" has its
    own gRPC server (mock UE) running independently.  We spin up N servers
    on different ports and verify that N independent ExternalSimulator+
    GrpcProtocol connections work correctly in parallel.
    """

    def test_two_independent_pods(self, make_env_server):
        """Two workers each connect to their own mock UE server."""
        port_a = make_env_server("CartPole-v1")
        port_b = make_env_server("CartPole-v1")

        sim_a = ExternalSimulator()
        proto_a = GrpcProtocol(url="localhost", port=port_a, credential_mode="insecure")
        env_a = RayEnv(proto_a, sim_a)

        sim_b = ExternalSimulator()
        proto_b = GrpcProtocol(url="localhost", port=port_b, credential_mode="insecure")
        env_b = RayEnv(proto_b, sim_b)

        obs_a, _ = env_a.reset(seed=42)
        obs_b, _ = env_b.reset(seed=99)

        assert isinstance(obs_a, dict)
        assert isinstance(obs_b, dict)

        # Run 20 steps on each independently
        for _ in range(20):
            act_a = {aid: env_a.single_action_spaces[aid].sample() for aid in obs_a}
            obs_a, rw_a, t_a, tr_a, i_a = env_a.step(act_a)
            if t_a.get("__all__") or tr_a.get("__all__"):
                obs_a, _ = env_a.reset()

            act_b = {aid: env_b.single_action_spaces[aid].sample() for aid in obs_b}
            obs_b, rw_b, t_b, tr_b, i_b = env_b.step(act_b)
            if t_b.get("__all__") or tr_b.get("__all__"):
                obs_b, _ = env_b.reset()

        env_a.close()
        env_b.close()

    def test_four_pod_vec_envs(self, make_vec_env_server, make_env):
        """Four workers, each with a 2-env vectorized mock UE server."""
        ports = []
        for worker_idx in range(4):
            envs = [make_env("CartPole-v1", worker_idx * 10 + i) for i in range(2)]
            port = make_vec_env_server(envs)
            ports.append(port)

        ray_envs = []
        for port in ports:
            sim = ExternalSimulator()
            proto = GrpcProtocol(url="localhost", port=port, credential_mode="insecure")
            ray_envs.append(RayVecEnv(proto, sim))

        for env in ray_envs:
            assert env.num_envs == 2
            obs, _ = env.reset()
            assert len(obs) == 2

        for env in ray_envs:
            obs, _ = env.reset()
            for _ in range(10):
                actions = [
                    {aid: env.single_action_spaces[aid].sample() for aid in o}
                    for o in obs
                ]
                obs, rw, t, tr, i = env.step(actions)

        for env in ray_envs:
            env.close()


# ---------------------------------------------------------------------------
# End-to-end training with ExternalSimulator (verifies training outcome)
# ---------------------------------------------------------------------------


@pytest.mark.xdist_group(name="ray-cluster")
class TestExternalSimulatorTraining:

    def test_training_with_external_simulator(
        self, make_vec_env_server, make_env, ray_cluster
    ):
        """
        Full training loop using ExternalSimulator + insecure gRPC,
        mimicking a K8s deployment where UE is an external pod.
        ScholaEnvRunner always creates a RayVecEnv, so we use the
        vectorized server even for a single env.
        Verifies that training produces a non-None result and that
        sampled timesteps increase across iterations.
        """
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.connectors.env_to_module import FlattenObservations
        from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
        from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec
        from ray.rllib.policy.policy import PolicySpec
        from schola.rllib.env_runner import ScholaEnvRunner

        envs = [make_env("CartPole-v1", 0)]
        port = make_vec_env_server(envs)

        config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            .training(
                num_epochs=2,
                train_batch_size=128,
                minibatch_size=32,
            )
            .environment(
                env_config={
                    "protocol": GrpcProtocol,
                    "protocol_args": {
                        "url": "localhost",
                        "port": port,
                        "credential_mode": "insecure",
                    },
                    "port_offset_mode": "fixed",
                    "simulator": ExternalSimulator,
                    "simulator_args": {},
                },
            )
            .framework("torch")
            .env_runners(
                env_runner_cls=ScholaEnvRunner,
                num_env_runners=0,
                env_to_module_connector=lambda env: FlattenObservations(
                    input_observation_space=env.single_observation_space,
                    input_action_space=env.single_action_space,
                    multi_agent=True,
                ),
            )
            .multi_agent(
                policies={"shared_policy": PolicySpec()},
                policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
            )
            .rl_module(
                rl_module_spec=MultiRLModuleSpec(
                    rl_module_specs={
                        "shared_policy": RLModuleSpec(
                            model_config=DefaultModelConfig(
                                fcnet_hiddens=[32, 32],
                                vf_share_layers=True,
                            )
                        )
                    }
                ),
            )
        )

        algo = config.build_algo()

        result1 = algo.train()
        assert result1 is not None

        ts1 = result1["num_env_steps_sampled_lifetime"]
        assert ts1 > 0, "Should have sampled some timesteps"

        result2 = algo.train()
        ts2 = result2["num_env_steps_sampled_lifetime"]
        assert ts2 > ts1, "Timesteps should increase across training iterations"

        algo.stop()

    def test_training_multi_env_external_simulator(
        self, make_vec_env_server, make_env, ray_cluster
    ):
        """
        Training with a multi-environment ExternalSimulator (2 envs),
        verifying training outcome with the ScholaEnvRunner.
        """
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.connectors.env_to_module import FlattenObservations
        from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
        from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec
        from ray.rllib.policy.policy import PolicySpec
        from schola.rllib.env_runner import ScholaEnvRunner

        envs = [make_env("CartPole-v1", i) for i in range(2)]
        port = make_vec_env_server(envs)

        config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            .training(
                num_epochs=2,
                train_batch_size=128,
                minibatch_size=32,
            )
            .environment(
                env_config={
                    "protocol": GrpcProtocol,
                    "protocol_args": {
                        "url": "localhost",
                        "port": port,
                        "credential_mode": "insecure",
                    },
                    "port_offset_mode": "fixed",
                    "simulator": ExternalSimulator,
                    "simulator_args": {},
                },
            )
            .framework("torch")
            .env_runners(
                env_runner_cls=ScholaEnvRunner,
                num_env_runners=0,
                env_to_module_connector=lambda env: FlattenObservations(
                    input_observation_space=env.single_observation_space,
                    input_action_space=env.single_action_space,
                    multi_agent=True,
                ),
            )
            .multi_agent(
                policies={"shared_policy": PolicySpec()},
                policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
            )
            .rl_module(
                rl_module_spec=MultiRLModuleSpec(
                    rl_module_specs={
                        "shared_policy": RLModuleSpec(
                            model_config=DefaultModelConfig(
                                fcnet_hiddens=[32, 32],
                                vf_share_layers=True,
                            )
                        )
                    }
                ),
            )
        )

        algo = config.build_algo()
        result = algo.train()
        assert result is not None
        assert result["num_env_steps_sampled_lifetime"] > 0
        algo.stop()

    def test_training_with_remote_worker(
        self, make_vec_env_server, make_env, ray_cluster
    ):
        """
        Cluster-like training with num_env_runners=1 (a real remote worker).

        This is the critical test that exercises the actual cluster codepath:
        - Ray serializes env_config and ships it to a remote EnvRunner process
        - The remote ScholaEnvRunner deserializes the config and constructs
          GrpcProtocol + ExternalSimulator from the serialized dict
        - credential_mode, environment_start_timeout, and port_offset_mode
          must survive the serialization round-trip
        - The remote worker connects to the gRPC server and collects samples

        With num_env_runners=0, everything runs in the main process and
        serialization is never tested.  This test catches bugs that only
        appear when config is shipped across process boundaries.
        """
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.connectors.env_to_module import FlattenObservations
        from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
        from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec
        from ray.rllib.policy.policy import PolicySpec
        from schola.rllib.env_runner import ScholaEnvRunner

        envs = [make_env("CartPole-v1", 0)]
        port = make_vec_env_server(envs)

        config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            .training(
                num_epochs=2,
                train_batch_size=128,
                minibatch_size=32,
            )
            .environment(
                env_config={
                    "protocol": GrpcProtocol,
                    "protocol_args": {
                        "url": "localhost",
                        "port": port,
                        "credential_mode": "insecure",
                        "environment_start_timeout": 90,
                    },
                    "port_offset_mode": "fixed",
                    "simulator": ExternalSimulator,
                    "simulator_args": {},
                },
            )
            .framework("torch")
            .env_runners(
                env_runner_cls=ScholaEnvRunner,
                num_env_runners=1,
                env_to_module_connector=lambda env, spaces=None, device=None: FlattenObservations(
                    input_observation_space=(
                        env.single_observation_space
                        if env is not None
                        else spaces["__env_single__"][0]
                    ),
                    input_action_space=(
                        env.single_action_space
                        if env is not None
                        else spaces["__env_single__"][1]
                    ),
                    multi_agent=True,
                ),
            )
            .multi_agent(
                policies={"shared_policy": PolicySpec()},
                policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
            )
            .rl_module(
                rl_module_spec=MultiRLModuleSpec(
                    rl_module_specs={
                        "shared_policy": RLModuleSpec(
                            model_config=DefaultModelConfig(
                                fcnet_hiddens=[32, 32],
                                vf_share_layers=True,
                            )
                        )
                    }
                ),
            )
        )

        algo = config.build_algo()

        result = algo.train()
        assert result is not None
        ts = result["num_env_steps_sampled_lifetime"]
        assert ts > 0, "Remote worker should have sampled timesteps"

        algo.stop()
