# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Settings dataclasses for the RLlib training command.
"""

from typing import Annotated
from pathlib import Path
from dataclasses import dataclass, field

from cyclopts import Parameter, validators

from schola.scripts.common.settings import (
    ActivationFunctionEnum,
    CheckpointSettings,
)

from schola.scripts.rllib.settings import (
    APPOSettings,
    IMPALASettings,
    LoggingSettings,
    PPOSettings,
    ResourceSettings,
    RllibEnvironmentSettings,
    SACSettings,
)


@dataclass
class TrainingSettings:
    """
    Dataclass for generic training settings used in the RLlib training process. This class defines the parameters for training, including the number of timesteps, learning rate, minibatch size, and other hyperparameters that control the training process. These settings are applicable to any RLlib algorithm and can be customized based on the specific requirements of the training job.
    """

    timesteps: Annotated[
        int, Parameter(validator=validators.Number(gte=1), alias="-t")
    ] = 3000
    "Stopping threshold for sampled environment steps (num_env_steps_sampled_lifetime). By default this is an absolute lifetime cap so the same command works for both fresh runs and --resume-from without hand-tuning totals. When --reset-timestep is set, this value is instead treated as additional steps to train beyond the checkpoint."

    learning_rate: Annotated[
        float, Parameter(validator=validators.Number(gt=0), alias="-l")
    ] = 0.0003
    "The learning rate for any chosen algorithm. This controls how much to adjust the model weights in response to the estimated error each time the model weights are updated. A smaller value means slower learning, while a larger value means faster learning."

    minibatch_size: Annotated[
        int, Parameter(validator=validators.Number(gte=1), alias="-b")
    ] = 128
    "The size of the minibatch for training. This is the number of samples used in each iteration of training to update the model weights. A larger batch size can lead to more stable estimates of the gradient, but requires more memory and can slow down training if too large."

    train_batch_size_per_learner: Annotated[
        int, Parameter(validator=validators.Number(gte=1))
    ] = 256
    "The number of samples given to each learner during training. Must be divisible by minibatch_size."

    num_epochs: Annotated[
        int, Parameter(validator=validators.Number(gte=1), alias="num_sgd_iter")
    ] = 5
    "The number of training epochs for each batch. This is the number of passes to make over the whole training batch. More epochs can lead to better convergence, but also increases the training time. Alias for num_sgd_iter."

    gamma: Annotated[
        float, Parameter(validator=validators.Number(gte=0.0, lte=1.0))
    ] = 0.99
    "The discount factor for the reinforcement learning algorithm. This is used to calculate the present value of future rewards. A value of 0.99 means that future rewards are discounted by 1% for each time step into the future. This helps to balance the importance of immediate versus future rewards in the training process. A value closer to 1.0 will prioritize future rewards more heavily, while a value closer to 0 will prioritize immediate rewards."

    @property
    def name(self) -> str:
        return "Training Settings"

    def __post_init__(self):
        if self.minibatch_size > self.train_batch_size_per_learner:
            raise ValueError(
                f"minibatch_size ({self.minibatch_size}) cannot exceed train_batch_size_per_learner ({self.train_batch_size_per_learner})."
            )
        if self.train_batch_size_per_learner % self.minibatch_size != 0:
            raise ValueError(
                f"train_batch_size_per_learner ({self.train_batch_size_per_learner}) must be a multiple of minibatch_size ({self.minibatch_size})."
            )


@dataclass
class ResumeSettings:
    """
    Dataclass for resume settings used in the RLlib training process. This class defines the parameters for resuming training from a saved checkpoint. This allows you to continue training from a previously saved model checkpoint instead of starting from scratch. This is useful for long training jobs or if you want to experiment with different hyperparameters without losing progress.
    """

    resume_from: Annotated[
        Path | None,
        Parameter(
            validator=validators.Path(exists=True, file_okay=True, dir_okay=True),
            alias="-r",
        ),
    ] = None
    "Path to a trusted RLlib checkpoint to resume from: the directory that contains `rllib_checkpoint.json` (for example `.../checkpoint_000000`). Same-algorithm checkpoints restore Tune state (optimizer, step counts, env-runners). A BC or MARWIL checkpoint into PPO or IMPALA loads only RLModule weights into a fresh run. Network flags must match the imitation run. SAC is not supported. Only use checkpoints from trusted sources because RLlib checkpoint metadata is read with pickle."

    reset_timestep: bool = False
    "Whether to treat --timesteps as additional steps to run beyond the checkpoint rather than an absolute lifetime cap. When False (default), --timesteps is the absolute num_env_steps_sampled_lifetime stop target, so the same command can resume training without modification. When True, the restored step count is added to --timesteps, matching the behaviour of SB3's reset_num_timesteps=True."

    @property
    def name(self) -> str:
        return "Resume Settings"


@dataclass
class NetworkArchitectureSettings:
    """
    Dataclass for network architecture settings used in the RLlib training process. This class defines the parameters for the neural network architecture used for policy and value function approximation. This includes the hidden layer sizes, activation functions, and whether to use an attention mechanism. These settings help to control the complexity and capacity of the neural network model used in the training process.
    """

    fcnet_hiddens: Annotated[list[int], Parameter(consume_multiple=True)] = field(
        default_factory=lambda: [512, 512]
    )
    "The hidden layer architecture for the fully connected network. This specifies the number of neurons in each hidden layer of the neural network used for the policy and value function approximation. The default is [512, 512], which means two hidden layers with 512 neurons each. This can be adjusted based on the complexity of the problem and the size of the input state space."

    activation: ActivationFunctionEnum = ActivationFunctionEnum.ReLU
    "The activation function to use for the fully connected network. This specifies the non-linear activation function applied to each neuron in the hidden layers of the neural network. The default is ReLU (Rectified Linear Unit), which is a commonly used activation function in deep learning due to its simplicity and effectiveness. Other options may include Tanh, Sigmoid, etc. This can be adjusted based on the specific requirements of the problem and the architecture of the neural network."

    use_lstm: bool = False
    "Whether to use an LSTM layer in the model. This specifies whether to include an LSTM layer in the neural network architecture. LSTM is a type of recurrent neural network that is designed to process sequential data."

    lstm_cell_size: Annotated[int, Parameter(validator=validators.Number(gte=1))] = 64
    "The size of the LSTM cell. This specifies the number of neurons in the LSTM cell. The default is 64, which is a common choice for many applications."

    max_seq_len: Annotated[int, Parameter(validator=validators.Number(gte=1))] = 20
    "Maximum sequence length for stateful (e.g. LSTM) models. Used in config.rl_module(model_config={...}). Only relevant when use_lstm is True."

    @property
    def name(self) -> str:
        return "Network Architecture Settings"

    def __post_init__(self):
        if not self.fcnet_hiddens:
            raise ValueError(
                "fcnet_hiddens must contain at least one positive layer size."
            )
        bad = [h for h in self.fcnet_hiddens if h <= 0]
        if bad:
            raise ValueError(
                f"fcnet_hiddens has non-positive entries {bad}; all layer sizes must be > 0."
            )


@dataclass
class RllibScriptSettings:
    """
    Top level dataclass for RLlib script arguments. This class aggregates all the settings required for configuring the RLlib training process. It includes settings for training, algorithms, logging, resuming from checkpoints, network architecture, and resource allocation. This allows for a comprehensive configuration of the RLlib training job in a structured manner.
    """

    training_settings: Annotated[
        TrainingSettings, Parameter(group="Training Arguments", name="*")
    ] = field(default_factory=TrainingSettings)
    "Settings for configuring the training process."

    algorithm_settings: Annotated[
        PPOSettings | SACSettings | APPOSettings | IMPALASettings,
        Parameter(show=False, parse=False),
    ] = field(default_factory=PPOSettings)
    "Settings for configuring the training algorithm to use."

    logging_settings: Annotated[
        LoggingSettings, Parameter(group="Logging Arguments", name="*")
    ] = field(default_factory=LoggingSettings)
    "Settings for enabling logging and configuring the logging directory."

    resume_settings: Annotated[
        ResumeSettings, Parameter(group="Resume Arguments", name="*")
    ] = field(default_factory=ResumeSettings)
    "Settings for resuming training from a checkpoint."

    network_architecture_settings: Annotated[
        NetworkArchitectureSettings,
        Parameter(group="Network Architecture Arguments", name="*"),
    ] = field(default_factory=NetworkArchitectureSettings)
    "Settings for configuring the neural network architecture used for training."

    resource_settings: Annotated[
        ResourceSettings, Parameter(group="Resource Arguments", name="*")
    ] = field(default_factory=ResourceSettings)
    "Settings for configuring the resource allocation for the training process."

    checkpoint_settings: Annotated[
        CheckpointSettings, Parameter(group="Checkpoint Arguments", name="*")
    ] = field(default_factory=CheckpointSettings)
    "Settings for checkpoints"

    environment_settings: Annotated[
        RllibEnvironmentSettings,
        Parameter(group="Environment Arguments", name="*"),
    ] = field(default_factory=RllibEnvironmentSettings)
    "Settings for the environment to use during training"


# Deprecated spellings; prefer RllibScriptSettings.
RLlibScriptArgs = RllibScriptSettings
TrainingArgs = TrainingSettings
ResourceArgs = ResourceSettings
LoggingArgs = LoggingSettings
ResumeArgs = ResumeSettings
NetworkArchitectureArgs = NetworkArchitectureSettings
CheckpointArgs = CheckpointSettings
EnvironmentArgs = RllibEnvironmentSettings
