# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""
RLlib Environment Runner for Schola/Unreal Engine. Use this class to use the RayVecEnv with Schola.
"""

import logging

from ray.rllib.callbacks.utils import make_callback
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.utils.annotations import override
from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner

from schola.core.protocols.base_protocol import BaseRLProtocol
from schola.core.simulators.base_simulator import BaseSimulator
from schola.rllib.env import RayVecEnv

logger = logging.getLogger("ray.rllib")


class ScholaEnvRunner(MultiAgentEnvRunner):
    """
    ``MultiAgentEnvRunner`` that instantiates ``RayVecEnv`` with Schola wiring.

    Notes
    -----
    Expects ``protocol`` and ``simulator`` classes inside ``env_config`` and
    assigns distinct gRPC ports per worker when a base port is provided.
    """

    @staticmethod
    def resolve_protocol_args(
        protocol_args: dict,
        port_offset_mode: str = "per_worker",
        worker_index: "int | None" = None,
    ) -> dict:
        """Apply per-worker URL template expansion and port-offset logic.

        All configuration flows through ``env_config`` (the ``EnvContext``).
        No environment variables are read here; the training driver is
        responsible for reading any external settings at the script level
        and placing them into the config before submission.

        URL templates may contain ``{worker_index}`` which is expanded using
        the worker's index from ``EnvContext`` (e.g. ``"ue-{worker_index}"``
        becomes ``"ue-1"`` on worker 1).

        Parameters
        ----------
        protocol_args:
            Base protocol keyword arguments (``url``, ``port``, ...).
        port_offset_mode:
            ``"per_worker"`` adds *worker_index* to the base port (single-host
            default).  ``"fixed"`` keeps the same port on every worker (K8s
            pods with isolated networks).
        worker_index:
            The Ray worker index, if available.

        Returns
        -------
        dict
            A **copy** of *protocol_args* with overrides applied.
        """
        args = dict(protocol_args)

        if worker_index is not None:
            url = args.get("url")
            if isinstance(url, str) and "{worker_index}" in url:
                args["url"] = url.format(worker_index=worker_index)

        port = args.get("port")
        if (
            port is not None
            and port_offset_mode == "per_worker"
            and worker_index is not None
        ):
            args["port"] = port + worker_index

        return args

    @override(EnvRunner)
    def make_env(self):
        # If an env already exists, try closing it first (to allow it to properly
        # cleanup).
        if self.env is not None:
            try:
                self.env.close()
            except Exception as e:
                logger.warning(
                    "Tried closing the existing env (multi-agent), but failed with "
                    f"error: {e.args[0]}"
                )
            del self.env

        env_ctx = self.config.env_config
        if not isinstance(env_ctx, EnvContext):
            env_ctx = EnvContext(
                env_ctx,
                worker_index=self.worker_index,
                num_workers=self.config.num_env_runners,
                remote=self.config.remote_worker_envs,
            )

        assert "protocol" in env_ctx, "Protocol must be provided in the env_config"
        assert "simulator" in env_ctx, "Simulator must be provided in the env_config"
        assert issubclass(
            env_ctx["protocol"], BaseRLProtocol
        ), "Protocol must be a BaseRLProtocol"
        assert issubclass(
            env_ctx["simulator"], BaseSimulator
        ), "Simulator must be a BaseSimulator"

        protocol_args = self.resolve_protocol_args(
            dict(env_ctx.get("protocol_args", {})),
            port_offset_mode=env_ctx.get("port_offset_mode", "per_worker"),
            worker_index=getattr(self, "worker_index", None),
        )

        self.env = RayVecEnv(
            env_ctx["protocol"](**protocol_args),
            env_ctx["simulator"](**env_ctx.get("simulator_args", {})),
            env_config=env_ctx,
        )

        self.num_envs: int = self.env.num_envs
        if self.num_envs != self.config.num_envs_per_env_runner:
            logger.warning(
                f"Ignoring 'num_envs_per_env_runner' setting because the number of environments ({self.num_envs}) does not match the number of environments per env runner ({self.config.num_envs_per_env_runner})"
            )
            self.config.num_envs_per_env_runner = self.num_envs

        if not self.config.disable_env_checking:
            logger.warning(
                "Environment checking setting is ignored when using the ScholaEnvRunner"
            )

        # Set the flag to reset all envs upon the next `sample()` call.
        self._needs_initial_reset = True

        # Call the `on_environment_created` callback.
        make_callback(
            "on_environment_created",
            callbacks_objects=self._callbacks,
            callbacks_functions=self.config.callbacks_on_environment_created,
            kwargs=dict(
                env_runner=self,
                metrics_logger=self.metrics,
                env=self.env.unwrapped,
                env_context=env_ctx,
            ),
        )
