# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Cyclopts dataclasses for Stable-Baselines3 training with Schola (PPO, SAC, checkpoints, resume).
"""

from __future__ import annotations

from typing import Annotated, List, Optional, Type, Union, Any, Dict

from schola.scripts.common.settings import (
    ActivationFunctionEnum,
    EnvironmentSettings,
    CheckpointSettings,
    Sb3LauncherExtension,
)
from dataclasses import dataclass, field
from pathlib import Path
from cyclopts import App, Parameter, validators

from cyclopts import types

from schola.scripts.sb3.settings import (
    BaseSACSettings,
    BasePPOSettings,
    Sb3BaseLoggingSettings,
)


@dataclass
class BaseSb3AlgorithmSettings:
    """
    Shared rollout and optimizer settings for on-policy SB3 algorithms.
    """

    learning_rate: Annotated[
        float, Parameter(validator=validators.Number(gt=0.0), alias="-l")
    ] = 0.0003
    "Learning rate for the optimizer."

    n_steps: Annotated[int, Parameter(validator=validators.Number(gte=1))] = 2048
    "Number of steps to run for each environment per update. This is the number of timesteps collected before updating the policy."

    batch_size: Annotated[
        int, Parameter(validator=validators.Number(gte=1), alias="-b")
    ] = 64
    "Minibatch size for each update. This is the number of timesteps used in each batch for training the policy. Must be a divisor of `n_steps`."

    def __post_init__(self):
        if self.n_steps % self.batch_size != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must divide n_steps ({self.n_steps})."
            )


@dataclass
class PPOTrainSettings(BaseSb3AlgorithmSettings, BasePPOSettings):
    """
    Dataclass for configuring the settings of the Proximal Policy Optimization (PPO) algorithm. This includes parameters for the learning process, such as learning rate, batch size, number of steps, and other hyperparameters that control the behavior of the PPO algorithm.
    """

    n_epochs: Annotated[int, Parameter(validator=validators.Number(gte=1))] = 10
    "Number of epochs to update the policy. This is the number of times the model will iterate over the collected data during training. More epochs can lead to better convergence but also overfitting."

    gamma: Annotated[
        float, Parameter(validator=validators.Number(gte=0.0, lte=1.0))
    ] = 0.99
    "Discount factor for future rewards. This determines how much the agent values future rewards compared to immediate rewards. A value of 0.99 means that future rewards are discounted by 1% per time step."

    gae_lambda: Annotated[
        float, Parameter(validator=validators.Number(gte=0.0, lte=1.0))
    ] = 0.95
    "Lambda parameter for Generalized Advantage Estimation (GAE). This parameter helps to balance bias and variance in the advantage estimation. A value of 1.0 corresponds to standard advantage estimation, while lower values will reduce variance but may introduce bias."

    clip_range: Annotated[float, Parameter(validator=validators.Number(gte=0.0))] = 0.2
    "Clipping range for the policy update. This is the maximum amount by which the new policy can differ from the old policy during training. This helps to prevent large updates that can destabilize training."

    normalize_advantage: bool = True
    "Whether to normalize the advantages. Normalizing the advantages can help to stabilize training by ensuring that they have a mean of 0 and a standard deviation of 1. This can lead to more consistent updates to the policy."

    ent_coef: Annotated[float, Parameter(validator=validators.Number(gte=0.0))] = 0.0
    "Coefficient for the entropy term in the loss function. This encourages exploration by adding a penalty for certainty in the policy's action distribution. A higher value will encourage more exploration, while a lower value will make the policy more deterministic. Set to 0.0 to disable entropy regularization."

    vf_coef: Annotated[float, Parameter(validator=validators.Number(gte=0.0))] = 0.5
    "Coefficient for the value function loss in the overall loss function. This determines how much weight is given to the value function loss compared to the policy loss. A higher value will put more emphasis on accurately estimating the value function, while a lower value will prioritize the policy update."

    max_grad_norm: Annotated[float, Parameter(validator=validators.Number(gte=0.0))] = (
        0.5
    )
    "Maximum gradient norm for clipping. This is used to prevent exploding gradients by scaling down the gradients if their norm exceeds this value. This can help to stabilize training, especially in environments with high variance in the rewards or gradients."

    use_sde: bool = False
    "Whether to use State Dependent Exploration (SDE). This can help to improve exploration by adapting the exploration noise based on the current state of the environment. When set to True, it will use SDE for exploration instead of the standard exploration strategy."

    sde_sample_freq: int = -1
    "Frequency at which to sample the SDE noise. This determines how often the noise is sampled when using State Dependent Exploration (SDE). A value of -1 means that it will sample the noise at every step, while a positive integer will specify the number of steps between samples. This can help to control the exploration behavior of the agent."

    target_kl: Annotated[
        Optional[float], Parameter(validator=validators.Number(gt=0.0))
    ] = None
    "Approximate KL-divergence threshold used to early-stop the PPO update epoch loop when the policy moves too far from the rollout distribution. When set, SB3 stops the inner epoch loop as soon as the running approximate KL exceeds this value, which stabilizes training when ``train/approx_kl`` would otherwise blow past ``clip_range``. Typical values are 0.01-0.05. Leave as None to disable (SB3's default)."

    def __post_init__(self):
        super().__post_init__()
        if self.normalize_advantage and self.batch_size <= 1:
            raise ValueError(
                "normalize_advantage=True requires batch_size > 1 to compute a valid variance."
            )
        if self.sde_sample_freq != -1 and self.sde_sample_freq <= 0:
            raise ValueError(
                f"Invalid sde_sample_freq={self.sde_sample_freq}. Must be -1 (every step) or a positive integer (>0)."
            )


@dataclass
class SACTrainSettings(BaseSb3AlgorithmSettings, BaseSACSettings):
    """
    Dataclass for configuring the settings of the Soft Actor-Critic (SAC) algorithm. This includes parameters for the learning process, such as learning rate, buffer size, batch size, and other hyperparameters that control the behavior of the SAC algorithm.
    """

    learning_rate: Annotated[
        float, Parameter(validator=validators.Number(gt=0.0), alias="-l")
    ] = 0.0003
    "Learning rate for the optimizer. This controls how much to adjust the model parameters in response to the estimated error each time the model weights are updated. A lower value means slower learning, while a higher value means faster learning."

    buffer_size: Annotated[int, Parameter(validator=validators.Number(gte=1))] = 1000000
    "Size of the replay buffer. This is the number of transitions (state, action, reward, next state) that can be stored in the buffer. A larger buffer allows for more diverse samples to be used for training, which can improve performance but also increases memory usage."

    learning_starts: Annotated[int, Parameter(validator=validators.Number(gte=0))] = 100
    "Number of timesteps before learning starts. This is the number of steps to collect in the replay buffer before the first update to the policy. This allows the agent to gather initial experience and helps to stabilize training by ensuring that there are enough samples to learn from."

    batch_size: Annotated[
        int, Parameter(validator=validators.Number(gte=1), alias="-b")
    ] = 256
    "Minibatch size for each update. This is the number of samples drawn from the replay buffer to perform a single update to the policy. A larger batch size can lead to more stable updates but requires more memory. Must be less than or equal to `buffer_size`."

    tau: Annotated[float, Parameter(validator=validators.Number(gte=0.0, lte=1.0))] = (
        0.005
    )
    "Soft update parameter for the target networks. This controls how much the target networks are updated towards the main networks during training. A smaller value (e.g., 0.005) means that the target networks are updated slowly, which can help to stabilize training. This is typically a small value between 0 and 1."

    gamma: Annotated[
        float, Parameter(validator=validators.Number(gte=0.0, lte=1.0))
    ] = 0.99
    "Discount factor for future rewards. This determines how much the agent values future rewards compared to immediate rewards. A value of 0.99 means that future rewards are discounted by 1% per time step. This is important for balancing the trade-off between short-term and long-term rewards in reinforcement learning."

    train_freq: Annotated[int, Parameter(validator=validators.Number(gte=1))] = 1
    "Frequency of training the policy. This determines how often the model is updated during training. A value of 1 means that the model is updated every time step, while a higher value (e.g., 2) means that the model is updated every other time step. This can help to control the trade-off between exploration and exploitation during training."

    gradient_steps: Annotated[int, Parameter(validator=validators.Number(gte=1))] = 1
    "Number of gradient steps to take during each training update. This specifies how many times to update the model parameters using the sampled minibatch from the replay buffer. A value of 1 means that the model is updated once per training step, while a higher value (e.g., 2) means that the model is updated multiple times. This can help to improve convergence but may also lead to overfitting if set too high."

    replay_buffer_kwargs: Optional[Dict[str, Any]] = None
    "Additional keyword arguments to pass to the replay buffer constructor. This allows for further customization of the replay buffer's behavior and settings when it is instantiated. For example, you can specify parameters like `buffer_size`, `seed`, or any other parameters supported by your custom replay buffer class. This can help to tailor the replay buffer to your specific needs or environment requirements."

    optimize_memory_usage: bool = False
    "Whether to optimize memory usage for the replay buffer. When set to True, it will use a more memory-efficient implementation of the replay buffer, which can help to reduce memory consumption during training. This is particularly useful when working with large environments or limited hardware resources. Note that this may slightly affect the performance of the training process, as it may introduce some overhead in accessing the samples."

    ent_coef: float | str = "auto"
    "Coefficient for the entropy term in the loss function. This encourages exploration by adding a penalty for certainty in the policy's action distribution. A higher value will encourage more exploration, while a lower value will make the policy more deterministic. When set to 'auto', it will automatically adjust the coefficient based on the average entropy of the actions taken by the policy. This can help to balance exploration and exploitation during training."

    target_update_interval: Annotated[
        int, Parameter(validator=validators.Number(gte=1))
    ] = 1
    "Interval for updating the target networks. This determines how often the target networks are updated with the main networks' weights. A value of 1 means that the target networks are updated every training step, while a higher value (e.g., 2) means that they are updated every other step. This can help to control the stability of training by ensuring that the target networks are kept up-to-date with the latest policy parameters."

    target_entropy: float | str = "auto"
    "Target entropy for the entropy regularization. This is used to encourage exploration by setting a target for the average entropy of the actions taken by the policy. When set to 'auto', it will automatically calculate the target entropy based on the dimensionality of the action space (e.g., -dimensionality of the action space). This helps to balance exploration and exploitation during training by encouraging the agent to explore more diverse actions."

    use_sde: bool = False
    "Whether to use State Dependent Exploration (SDE). This can help to improve exploration by adapting the exploration noise based on the current state of the environment. When set to True, it will use SDE for exploration instead of the standard exploration strategy. This can lead to more efficient exploration in complex environments, but may also introduce additional computational overhead."

    sde_sample_freq: int = -1
    "Frequency at which to sample the SDE noise. This determines how often the noise is sampled when using State Dependent Exploration (SDE). A value of -1 means that it will sample the noise at every step, while a positive integer will specify the number of steps between samples. This can help to control the exploration behavior of the agent. A higher frequency can lead to more diverse exploration, while a lower frequency may lead to more stable but less exploratory behavior."

    def __post_init__(self):
        super().__post_init__()
        if self.batch_size > self.buffer_size:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be <= buffer_size ({self.buffer_size})."
            )
        if self.learning_starts >= self.buffer_size:
            raise ValueError(
                f"learning_starts ({self.learning_starts}) must be < buffer_size ({self.buffer_size})."
            )
        if self.sde_sample_freq != -1 and self.sde_sample_freq <= 0:
            raise ValueError(
                f"Invalid sde_sample_freq={self.sde_sample_freq}. Must be -1 (every step) or a positive integer (>0)."
            )
        # Validate replay_buffer_kwargs according to supported parameters of the default SB3 ReplayBuffer
        if self.replay_buffer_kwargs is not None:
            if not isinstance(self.replay_buffer_kwargs, dict):
                raise ValueError(
                    "replay_buffer_kwargs must be a dict or None (got type "
                    f"{type(self.replay_buffer_kwargs).__name__})."
                )

            allowed_keys = {"handle_timeout_termination"}
            reserved_keys = {
                # Arguments supplied internally by the algorithm / env setup
                "buffer_size",
                "observation_space",
                "action_space",
                "device",
                "n_envs",
                "optimize_memory_usage",
            }
            provided_keys = set(self.replay_buffer_kwargs.keys())

            # Reject unknown keys (anything not explicitly allowed). This also covers reserved keys.
            unknown = provided_keys - allowed_keys
            if unknown:
                raise KeyError(
                    "Unsupported keys in replay_buffer_kwargs: "
                    f"{sorted(unknown)}. Allowed keys: {sorted(allowed_keys)}."
                )

            # Type checks for allowed keys
            if "handle_timeout_termination" in self.replay_buffer_kwargs:
                val = self.replay_buffer_kwargs["handle_timeout_termination"]
                if not isinstance(val, bool):
                    raise KeyError(
                        "replay_buffer_kwargs['handle_timeout_termination'] must be a bool (got "
                        f"{type(val).__name__})."
                    )
        # Always warn about non-multiple relationship (previously mistakenly nested behind replay_buffer_kwargs block)
        # TODO: Is this necessary/useful?
        # ~ noamonti 2025/9/10
        # if self.learning_starts % self.train_freq != 0:
        #     print_warning(
        #         f"Learning starts: ({self.learning_starts}) is not a multiple of train_freq: ({self.train_freq}). This may delay the start of training."
        #     )


@dataclass
class Sb3ResumeSettings:
    """
    Dataclass for holding arguments related to resuming training from a saved state.
    """

    resume_from: Annotated[
        Optional[Path],
        Parameter(
            validator=validators.Path(exists=True, file_okay=True, dir_okay=False),
            alias="-r",
        ),
    ] = None
    "Path to a saved model to resume training from. This allows for continuing training from a previously saved checkpoint. The path should point to a valid model file created by Stable Baselines3. If set to None, training will start from scratch."

    load_vecnormalize: Annotated[
        Optional[Path],
        Parameter(
            validator=validators.Path(exists=True, file_okay=True, dir_okay=False)
        ),
    ] = None
    "Path to a saved vector normalization statistics file to load when resuming training. This allows for loading the normalization statistics from a previous training session, ensuring that the observations are normalized consistently when resuming training. If set to None, it will not load any vector normalization statistics."

    load_replay_buffer: Annotated[
        Optional[Path],
        Parameter(
            validator=validators.Path(exists=True, file_okay=True, dir_okay=False)
        ),
    ] = None
    "Path to a saved replay buffer to load when resuming training. This allows for loading a previously saved replay buffer, which can be useful for continuing training with the same set of experiences. The path should point to a valid replay buffer file created by Stable Baselines3. If set to None, it will not load any replay buffer, and a new one will be created instead."

    reset_timestep: bool = False
    "Whether to reset the internal timestep counter when resuming training from a saved model. When set to True, it will reset the timestep counter to 0."


@dataclass
class Sb3LoggingSettings(Sb3BaseLoggingSettings):
    """
    Dataclass for configuring logging settings for the training process.
    """

    enable_tensorboard: bool = False
    "Whether to enable TensorBoard logging."

    log_dir: types.Directory = Path("./logs")
    "Directory to save TensorBoard logs. (Will be created if it doesn't exist when tensorboard is enabled.)"

    log_freq: Annotated[int, Parameter(validator=validators.Number(gte=0))] = 10
    "Frequency of logging training metrics to TensorBoard. This determines how often (in terms of training steps) the training metrics will be logged to TensorBoard. A value of 10 means that every 10 training steps, the metrics will be recorded."

    callback_verbosity: Annotated[
        int, Parameter(validator=validators.Number(gte=0, lte=2))
    ] = 0
    "Verbosity level for callbacks. This controls the level of detail in the output from any callbacks used during training."

    def __post_init__(self):
        # create log_dir eagerly only if tensorboard will be used
        if self.enable_tensorboard and self.log_dir.exists() is False:
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create TensorBoard log directory '{self.log_dir}': {e}"
                ) from e


@dataclass
class Sb3CheckpointSettings(CheckpointSettings):
    """
    SB3-specific checkpoint settings.
    """

    save_replay_buffer: bool = False
    "Whether to save the replay buffer when saving a checkpoint. This allows for resuming training from the same state of the replay buffer."

    save_vecnormalize: bool = False
    "Whether to save the vector normalization statistics when saving a checkpoint. This is useful for environments where observations need to be normalized, and it allows for consistent normalization when resuming training."


@dataclass
class Sb3NetworkArchitectureSettings:
    """
    Network architecture settings for SB3 algorithms.
    """

    policy_parameters: Annotated[List[int], Parameter(consume_multiple=True)] = field(
        default_factory=lambda: [256, 256]
    )
    "A list of layer widths representing the policy network architecture. This defines the number of neurons in each hidden layer of the policy network. For example, [64, 64] would create a policy network with two hidden layers, each containing 64 neurons. If set to None, it will use the default architecture defined by the algorithm."

    critic_parameters: Annotated[List[int], Parameter(consume_multiple=True)] = field(
        default_factory=lambda: [256, 256]
    )
    "A list of layer widths representing the critic (value function) network architecture. This defines the number of neurons in each hidden layer of the critic network. For example, [64, 64] would create a critic network with two hidden layers, each containing 64 neurons. This is only applicable for algorithms that use a critic (e.g., SAC). If set to None, it will use the default architecture defined by the algorithm."

    activation: ActivationFunctionEnum = ActivationFunctionEnum.ReLU
    "Activation function to use in the policy and critic networks. This determines the non-linear activation function applied to each layer of the neural networks. The choice of activation function can affect the performance of the model and may depend on the specific characteristics of the environment."

    def __post_init__(self):
        if self.activation not in list(ActivationFunctionEnum):
            raise ValueError(
                f"activation must be one of {list(ActivationFunctionEnum)} (got '{self.activation}')."
            )


IgnoreParameter = Parameter(show=False, parse=False)


@dataclass
class Sb3TrainingSettings:
    """
    Top-level training run options for the SB3 launcher (mirrors RLlib's ``TrainingSettings`` grouping).
    """

    timesteps: Annotated[
        int, Parameter(validator=validators.Number(gte=1), alias="-t")
    ] = 3000
    "Total number of timesteps to train the agent. This is the total number of environment steps that will be used for training. This should be set based on the complexity of the environment and the desired training duration. A higher value will typically lead to better performance but will also increase training time."

    pbar: bool = False
    "Whether to display a progress bar during training. Requires TQDM and Rich to be installed."

    disable_eval: bool = False
    "Whether to disable running evaluation after training. When set to True, it will skip evaluation after training completes."


@dataclass
class Sb3TrainScriptSettings:
    """
    Top level dataclass for configuring the script arguments used in the SB3 launcher.
    """

    training_settings: Annotated[
        Sb3TrainingSettings, Parameter(group="Training Arguments", name="*")
    ] = field(default_factory=Sb3TrainingSettings)
    "Timesteps, progress bar, and evaluation options for the training run."

    logging_settings: Annotated[
        Sb3LoggingSettings, Parameter(group="Logging Arguments", name="*")
    ] = field(default_factory=Sb3LoggingSettings)
    "Settings for enabling logging and configuring the logging directory."

    resume_settings: Annotated[
        Sb3ResumeSettings, Parameter(group="Resume Arguments", name="*")
    ] = field(default_factory=Sb3ResumeSettings)
    "Settings for resuming training from a checkpoint."

    checkpoint_settings: Annotated[
        Sb3CheckpointSettings, Parameter(group="Checkpoint Arguments", name="*")
    ] = field(default_factory=Sb3CheckpointSettings)
    "Settings for enabling checkpointing and configuring the checkpoint directory."

    network_architecture_settings: Annotated[
        Sb3NetworkArchitectureSettings,
        Parameter(group="Network Architecture Arguments", name="*"),
    ] = field(default_factory=Sb3NetworkArchitectureSettings)
    "Settings for configuring the neural network architecture used for training."

    algorithm_settings: Annotated[
        Union[PPOTrainSettings, SACTrainSettings], Parameter(show=False, parse=False)
    ] = field(default_factory=PPOTrainSettings)
    "The settings for the training algorithm to use. This can be either `PPOSettings` or `SACSettings`, depending on the chosen algorithm. This property allows for easy switching between different algorithms (e.g., PPO or SAC) by simply changing the instance of the settings class. The default is `PPOSettings`, which is suitable for most environments unless specified otherwise."

    @property
    def name_prefix(self):
        return (
            self.checkpoint_settings.name_prefix_override
            if self.checkpoint_settings.name_prefix_override is not None
            else self.algorithm_settings.name.lower()
        )

    environment_settings: Annotated[
        EnvironmentSettings, Parameter(group="Environment Arguments", name="*")
    ] = field(default_factory=EnvironmentSettings)
    "Settings for the environment to use during training"
