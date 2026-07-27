# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Shared settings dataclasses for RLlib scripts (algorithms, resources, logging).
"""

from typing import Annotated, Any, Dict, Type
from dataclasses import dataclass, field

from cyclopts import Parameter, validators

from schola.scripts.common.settings import (
    AllSimulatorConfigs,
    EnvironmentSettings,
    ExternalSimulatorConfig,
    IgnoreParameter,
)


class RllibAlgorithmSpecificSettings:
    """
    Base class for RLlib algorithm-specific settings. This class is intended to be inherited by specific algorithm settings classes (e.g., PPOSettings, IMPALASettings, etc.).
    """

    def get_settings_dict(self) -> Dict[str, Any]:
        """
        Get the settings as a dictionary keyed by the correct parameter name in Ray
        """
        ...

    @classmethod
    def get_parser(cls):
        """
        Add the settings to the parser or subparser
        """
        ...


@dataclass
class PPOSettings(RllibAlgorithmSpecificSettings):
    """
    Dataclass for PPO (Proximal Policy Optimization) algorithm specific settings. This class defines the parameters used in the PPO algorithm, including GAE lambda, clip parameter, and whether to use GAE.
    """

    gae_lambda: Annotated[
        float, Parameter(validator=validators.Number(gte=0.0, lte=1.0))
    ] = 0.95
    "The lambda parameter for Generalized Advantage Estimation (GAE). This controls the trade-off between bias and variance in the advantage estimation."

    clip_param: Annotated[float, Parameter(validator=validators.Number(gte=0.0))] = 0.2
    "The clip parameter for the PPO algorithm. This is the epsilon value used in the clipped surrogate objective function. It helps to limit the policy update step size to prevent large changes that could lead to performance collapse."

    use_gae: bool = True
    "Whether to use Generalized Advantage Estimation (GAE) for advantage calculation. GAE is a method to reduce the variance of the advantage estimates while keeping bias low. If set to False, the standard advantage calculation will be used instead."

    @property
    def rllib_config(self) -> Type["PPOConfig"]:  # type: ignore
        from ray.rllib.algorithms.ppo.ppo import PPOConfig

        return PPOConfig

    @property
    def name(self) -> str:
        return "PPO"

    def get_settings_dict(self):
        return {
            "lambda_": self.gae_lambda,
            "use_gae": self.use_gae,
            "clip_param": self.clip_param,
        }


@dataclass
class SACSettings(RllibAlgorithmSpecificSettings):
    """
    Dataclass for SAC (Soft Actor-Critic) algorithm specific settings. This class defines the parameters used in the SAC algorithm, including soft target network updates and entropy regularization.
    """

    tau: Annotated[float, Parameter(validator=validators.Number(gte=0.0, lte=1.0))] = (
        0.005
    )
    "Soft update coefficient for target networks. Controls how quickly target networks track the main networks. Lower values (e.g., 0.005) mean slower updates, which can improve stability."

    target_entropy: str = "auto"
    "Target entropy for automatic temperature tuning. Set to 'auto' to automatically calculate based on action space dimensionality, or provide a float value for manual control."

    initial_alpha: Annotated[float, Parameter(validator=validators.Number(gt=0.0))] = (
        1.0
    )
    "Initial temperature/alpha value for entropy regularization. Higher values encourage more exploration."

    n_step: Annotated[int, Parameter(validator=validators.Number(gte=1))] = 1
    "Number of steps for n-step returns. Using n>1 can help with credit assignment in sparse reward environments."

    twin_q: bool = True
    "Whether to use twin Q networks (double Q-learning). This helps reduce overestimation bias in Q-value estimates."

    @property
    def rllib_config(self) -> Type["SACConfig"]:  # type: ignore
        from ray.rllib.algorithms.sac.sac import SACConfig

        return SACConfig

    @property
    def name(self) -> str:
        return "SAC"

    def get_settings_dict(self):
        return {
            "tau": self.tau,
            "target_entropy": self.target_entropy,
            "initial_alpha": self.initial_alpha,
            "n_step": self.n_step,
            "twin_q": self.twin_q,
        }


@dataclass
class IMPALASettings(RllibAlgorithmSpecificSettings):
    """
    Dataclass for IMPALA (Importance Weighted Actor-Learner Architecture) algorithm specific settings. This class defines the parameters used in the IMPALA algorithm, including V-trace settings for off-policy correction.
    """

    vtrace: bool = True
    "Whether to use the V-trace algorithm for off-policy correction in the IMPALA algorithm. V-trace is a method to correct the bias introduced by using off-policy data for training. It helps to ensure that the value estimates are more accurate and stable."

    vtrace_clip_rho_threshold: Annotated[
        float, Parameter(validator=validators.Number(gte=0.0))
    ] = 1.0
    "The clip threshold for V-trace rho values."

    vtrace_clip_pg_rho_threshold: Annotated[
        float, Parameter(validator=validators.Number(gte=0.0))
    ] = 1.0
    "The clip threshold for V-trace rho values in the policy gradient."

    @property
    def rllib_config(self) -> Type["IMPALAConfig"]:  # type: ignore
        from ray.rllib.algorithms.impala.impala import IMPALAConfig

        return IMPALAConfig

    @property
    def name(self) -> str:
        return "IMPALA"

    def get_settings_dict(self):
        return {
            "vtrace": self.vtrace,
            "vtrace_clip_rho_threshold": self.vtrace_clip_rho_threshold,
            "vtrace_clip_pg_rho_threshold": self.vtrace_clip_pg_rho_threshold,
        }


@dataclass
class APPOSettings(IMPALASettings, PPOSettings):
    """
    Dataclass for APPO (Asynchronous Proximal Policy Optimization) algorithm specific settings. This class inherits from both IMPALASettings and PPOSettings to combine the settings for both algorithms. This allows for the use of both V-trace for off-policy correction and PPO for policy optimization in a single algorithm.
    """

    @property
    def rllib_config(self) -> Type["APPOConfig"]:  # type: ignore
        from ray.rllib.algorithms.appo.appo import APPOConfig

        return APPOConfig

    @property
    def name(self) -> str:
        return "APPO"

    def get_settings_dict(self):
        return {
            **IMPALASettings.get_settings_dict(self),
            **PPOSettings.get_settings_dict(self),
        }


@dataclass
class ResourceSettings:
    """
    Dataclass for resource settings used in the RLlib training process. This class defines the parameters for allocating computational resources, including the number of GPUs and CPUs to use for the training job. These settings help to control how resources are allocated for the training process, which can impact performance and training times. This is especially important when running on a cluster or distributed environment.
    """

    num_gpus: Annotated[int, Parameter(validator=validators.Number(gte=0))] = 0
    "The number of GPUs to use for the training process. This specifies how many GPUs are available for the RLlib training job. If set to 0, it will default to CPU training. This can be used to leverage GPU acceleration for faster training times if available."

    num_cpus: Annotated[int, Parameter(validator=validators.Number(gte=1))] = 1
    "The total number of CPUs to use for the training process. This specifies how many CPU cores are available for the RLlib training job. This can be used to parallelize the training process across multiple CPU cores, which can help to speed up training times."

    num_learners: Annotated[int, Parameter(validator=validators.Number(gte=0))] = 0
    "The number of learner processes to use for the training job. This specifies how many parallel learner processes will be used to train the model. Each learner will process a portion of the training data and update the model weights independently. This can help to speed up training times by leveraging multiple CPU cores or GPUs."

    num_cpus_for_main_process: Annotated[
        int, Parameter(validator=validators.Number(gte=1))
    ] = 1
    "The number of CPUs to allocate for the main process. This is the number of CPU cores that will be allocated to the main process that manages the training job. This can be used to ensure that the main process has enough resources to handle the workload and manage the learner processes effectively."

    num_cpus_per_learner: Annotated[
        int, Parameter(validator=validators.Number(gte=1))
    ] = 1
    "The number of CPUs to allocate for each learner process. This specifies how many CPU cores will be allocated to each individual learner process that is used for training. This can be used to ensure that each learner has enough resources to handle its workload and process the training data efficiently."

    num_gpus_per_learner: Annotated[
        int, Parameter(validator=validators.Number(gte=0))
    ] = 0
    "The number of GPUs to allocate for each learner process. This specifies how many GPUs will be allocated to each individual learner process that is used for training."

    using_cluster: bool = False
    "Whether Ray is running on a predefined cluster, or if one should be created as part of the launch script."

    @property
    def name(self) -> str:
        return "Resource Settings"


@dataclass
class LoggingSettings:
    """
    Dataclass for logging settings used in the RLlib training process. This class defines the verbosity levels for logging in both the Schola environment and RLlib. These settings help to control the amount of logging information generated during the training process, which can be useful for debugging and understanding the training process. Adjusting these settings can help to balance the amount of information logged against performance and readability of the logs.
    """

    schola_verbosity: Annotated[
        int, Parameter(validator=validators.Number(gte=0, lte=2))
    ] = 0  # Errors out < zero, but warns on > 3. Discuss if this behaviour is correct!
    "Verbosity level for the Schola environment. This controls the level of detail in the logs generated by the Schola environment. A higher value will produce more detailed logs, which can be useful for debugging and understanding the training process. Default is 0 (no additional logging)."

    rllib_verbosity: Annotated[
        int, Parameter(validator=validators.Number(gte=0, lte=3))
    ] = 1
    "Verbosity level for RLlib. This controls the level of detail in the logs generated by RLlib. A higher value will produce more detailed logs, which can be useful for debugging and understanding the training process. Default is 1 (standard logging)."

    @property
    def name(self) -> str:
        return "Logging Settings"

    @property
    def rllib_log_level(self) -> str:
        return {
            0: "ERROR",
            1: "WARN",
            2: "INFO",
            3: "DEBUG",
        }[self.rllib_verbosity]


@dataclass
class RllibEnvironmentSettings(EnvironmentSettings[AllSimulatorConfigs]):
    """
    Dataclass for configuring the environment settings for RLlib training.
    """

    simulator_settings: Annotated[
        AllSimulatorConfigs,
        IgnoreParameter,
    ] = field(default_factory=ExternalSimulatorConfig)
