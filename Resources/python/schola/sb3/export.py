# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""ONNX export helpers for Stable-Baselines3 policies trained or deployed with Schola."""

import logging
from typing import Generic, Optional, Tuple, TypeVar

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.td3.policies import TD3Policy

logger = logging.getLogger(__name__)

from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.preprocessing import preprocess_obs
from schola.core.model import ScholaModel
from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary
import stable_baselines3 as sb3
import torch as th
import gymnasium as gym
from pathlib import Path
from stable_baselines3.common.base_class import BaseAlgorithm

# The below code is adapted from https://github.com/DLR-RM/stable-baselines3/blob/v2.2.1/docs/guide/export.rst

# The MIT License
#
# Copyright (c) 2019 Antonin Raffin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Modifications Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All Rights Reserved.

T = TypeVar('T', bound=BasePolicy)

class SB3ScholaModel(ScholaModel, Generic[T]):
    """
    A PyTorch Module that wraps a stable baselines policy and extracts the necessary components to export to ONNX.

    Parameters
    ----------
    policy : stable_baselines3.common.policies.BasePolicy
        The policy to wrap.
    action_space : gym.spaces.Dict
        The action space to wrap.

    Attributes
    ----------
    extractor : _CombinedExtractorWrapper
        The wrapped feature extractor.

    action_net : th.nn.Module
        The wrapped action network.
    """

    def __init__(self, policy: T, action_space: Optional[gym.Space] = None):
        super().__init__(
            observation_space=policy.observation_space,
            action_space=action_space or policy.action_space,
        )
        self.policy = policy
        self.policy.set_training_mode(False)

    def get_logits(self, x) -> th.Tensor: ...

    def forward(self, *args) -> Tuple[th.Tensor, ...]:
        """
        Forward pass of the model.

        Parameters
        ----------
        *args : th.Tensor
            Observation inputs followed by state inputs (if stateful).
            Observation inputs correspond to self.input_obs_keys.
            State inputs correspond to self.input_state_keys.

        Returns
        -------
        Tuple[th.Tensor, ...]
            Action outputs followed by state outputs (if stateful).
            Action outputs correspond to self.output_action_keys.
        """
        # Separate observation inputs from state inputs
        args_iter = iter(args)
        obs_dict = {k: v for k, v in zip(self.input_obs_keys, args_iter)}

        # Consume state inputs (if any) - SB3 models are not stateful, so this will be empty
        # We consume them to ensure the iterator is properly exhausted
        if len(list(args_iter)) > 0:
            logger.warning(
                "State inputs are not supported for default SB3 export. Ignoring state inputs."
            )

        # Reconstruct observation for the policy, by applying default SB3 preprocessing
        # If observation_space_is_natively_dict is False, we wrapped it in a Dict with key "obs" so unwrap

        if self.observation_space_is_natively_dict:
            x = obs_dict
        else:
            x = obs_dict["obs"]

        logits = self.get_logits(x)
        action_outputs = self.make_outputs(logits)

        # No State Outputs for default SB3 export
        return tuple(action_outputs)


class SB3PPOModel(SB3ScholaModel[ActorCriticPolicy]):
    """
    ``SB3ScholaModel`` specialization for PPO models.

    See Also
    --------
    SB3ScholaModel
    """

    def get_logits(self, x):
        # discard the extracted features from the value function
        if self.policy.share_features_extractor:
            x = self.policy.extract_features(
                x
            )  # returns a single tensor of the extracted features
        else:
            x, _ = self.policy.extract_features(
                x
            )  # returns a tuple of (actor features, value function features)
        x, _ = self.policy.mlp_extractor(x)
        x = self.policy.action_net(x)
        return x


# A2C and PPO share the same policies
class SB3A2CModel(SB3PPOModel):
    """
    Same model ONNX path as ``SB3PPOModel`` (A2C policies match PPO layout).

    See Also
    --------
    SB3PPOModel
    """

    ...


# All possible policies for BC share the same policies
class SB3BCModel(SB3PPOModel):
    """
    Same model ONNX path as ``SB3PPOModel`` for behavioral cloning policies.

    See Also
    --------
    SB3PPOModel
    """

    ...


class SB3SACModel(SB3ScholaModel[SACPolicy]):
    """
    ONNX export view of an SAC actor (Gaussian policy mean path).

    See Also
    --------
    SB3ScholaModel
    """

    def get_logits(self, x):
        x = self.policy.actor.extract_features(x, self.policy.actor.features_extractor)
        x = self.policy.actor.latent_pi(x)
        x = self.policy.actor.mu(x)
        return x


class SB3TD3Model(SB3ScholaModel[TD3Policy]):
    """
    ONNX export view of a TD3 deterministic actor.

    See Also
    --------
    SB3ScholaModel
    """

    def get_logits(self, x):
        x = self.policy.actor.extract_features(x, self.policy.actor.features_extractor)
        x = self.policy.actor.mu(x)
        return x


# DDPG and TD3 share the same policies
class SB3DDPGModel(SB3TD3Model):
    """
    Same deterministic actor export as ``SB3TD3Model`` for DDPG.

    See Also
    --------
    SB3TD3Model
    """

    ...


class SB3DQNModel(SB3ScholaModel):
    """
    ONNX export of DQN Q-network logits for discrete actions.

    See Also
    --------
    SB3ScholaModel
    """

    def get_logits(self, x):
        x = self.policy.q_net(x)
        return x


def get_scholasb3_model(
    model: BaseAlgorithm, action_space: gym.Space | None = None
) -> ScholaModel:
    """
    Get the ScholaModel for a stable baselines algorithm.

    Parameters
    ----------
    model : stable_baselines3.common.base_class.BaseAlgorithm
        The model to get the ScholaModel for.
    action_space : gym.Space | None
        The action space to use for the model. Useful for when the model has a merged dictionary action space (e.g. Dict with two Box Subspaces).
    
    Returns
    -------
    ScholaModel
        The ScholaModel for the given model.
    """
    if isinstance(model, sb3.PPO):
        return SB3PPOModel(model.policy, action_space)
    elif isinstance(model, sb3.A2C):
        return SB3A2CModel(model.policy, action_space)
    elif isinstance(model, sb3.SAC):
        return SB3SACModel(model.policy, action_space)
    elif isinstance(model, sb3.TD3):
        return SB3TD3Model(model.policy, action_space)
    elif isinstance(model, sb3.DDPG):
        return SB3DDPGModel(model.policy, action_space)
    elif isinstance(model, sb3.DQN):
        return SB3DQNModel(model.policy, action_space)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


# end of adapted code
def save_model_as_onnx(
    model: BaseAlgorithm, export_path: str, action_space: gym.Space | None = None
) -> None:
    """
    Save a stable baselines model as ONNX.

    Parameters
    ----------
    model : stable_baselines3.common.base_class.BaseAlgorithm
        The model to save as ONNX.
    export_path : str
        The path to save the model to.
    action_space : gym.Space
        The action space to use for the model. Useful for when the model has a merged dictionary action space (e.g. Dict with two Box Subspaces).
    """
    schola_model = get_scholasb3_model(model, action_space)
    schola_model.save_as_onnx(export_path)


def convert_ckpt_to_onnx_for_unreal(
    trainer=sb3.PPO,
    model_path="./ckpt/ppo_final.zip",
    export_path="./ckpt/OnnxFiles/Model.onnx",
) -> None:
    """
    Convert a stable baselines model to ONNX for use in Unreal.

    Parameters
    ----------
    trainer : stable_baselines3.common.base_class.BaseAlgorithm
        The trainer to load the model from.
    model_path : str
        The path to the model to convert.
    export_path : str
        The path to save the converted model to.
    """
    model = trainer.load(model_path, device="cpu")
    action_space = getattr(model, "__unmerged_action_space", None)
    model = get_scholasb3_model(model, action_space)
    model.save_as_onnx(export_path)
