# Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Collection of custom callback classes for training NPCRL through Schola with stable baselines3
"""

import sys
import time
from typing import List
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import numpy as np

ENV_AXIS = 0
INTERVAL_AXIS = 1


class SingleEnvRewardCallback(BaseCallback):
    """
    Callback for logging rewards and steps taken by a single environment inside a vector environment.

    Parameters
    ----------
    verbose : int
        Verbosity level.
    id : int
        The id of the environment to log rewards and steps for.
    frequency : int
        The frequency at which to log the rewards and steps taken.

    Attributes
    ----------
    episode_reward : float
        The reward for the current episode.
    episode_rewards : List[float]
        The rewards for each episode.
    episode_steps : int
        The number of steps taken in the current episode.
    step_count : List[int]
        The number of steps taken in each episode.
    last_logging_interval : int
        The last interval that was logged.
    logging_interval_size : int
        The frequency at which to log the rewards and steps taken.
    id : int
        The id of the environment to log rewards and steps for.

    """

    def __init__(self, verbose=0, id=0, frequency=10):
        super().__init__(verbose)
        self.episode_reward = 0
        self.episode_rewards = []
        self.episode_steps = 0
        self.step_count = []
        self.last_logging_interval = 0
        self.logging_interval_size = frequency
        self.id = id

    @property
    def ready_to_log(self) -> bool:
        """
        Returns whether the environment is ready to log, by checking if there are more episodes completed than `self.logging_interval_size` since we last logged.

        Returns
        -------
        bool
            Whether the environment is ready to log.
        """
        return len(self.episode_rewards) >= (
            self.last_logging_interval + self.logging_interval_size
        )

    def _on_step(self) -> bool:
        self.episode_steps += 1
        self.episode_reward += self.locals["rewards"][self.id]
        if self.locals["dones"][self.id]:
            self.episode_rewards.append(self.episode_reward)
            self.step_count.append(self.episode_steps)
            self.episode_steps = 0
            self.episode_reward = 0
        return True

    def get_reward_interval(self) -> List[int]:
        """
        Returns the rewards for the last logging interval.

        Returns
        -------
        List[float]
            The rewards for the last logging interval.
        """
        return self.episode_rewards[
            self.last_logging_interval : self.last_logging_interval
            + self.logging_interval_size
        ]

    def get_step_interval(self) -> List[int]:
        """
        Returns the steps taken for each episode in the last logging interval.

        Returns
        -------
        List[int]
            The steps taken for each episode in the last logging interval.
        """
        return self.step_count[
            self.last_logging_interval : self.last_logging_interval
            + self.logging_interval_size
        ]

    def increment_logging_interval(self) -> None:
        """
        Increments the logging interval by `self.logging_interval_size` steps.
        """
        self.last_logging_interval += self.logging_interval_size


class RewardCallback(CallbackList):
    """
    Callback for logging rewards and steps taken by each environment in a multi-env setting.

    Parameters
    ----------
    verbose : int, default=0
        Verbosity level.
    frequency : int, default=10
        The frequency at which to log the rewards and steps taken.
    num_envs : int, default=1
        The number of environments to log rewards and steps for.

    Attributes
    ----------
    num_envs : int
        The number of environments to log rewards and steps for.
    callbacks : List[SingleEnvRewardCallback]
        The list of RewardLoggingCallbacks for each environment.
    summarize_every : int
        The frequency at which to log the rewards and steps taken.
    curr_logging_interval : int
        The current logging interval.
    start_time : int
        The time at which the callback was created.
    """

    def __init__(self, verbose: int = 0, frequency: int = 10, num_envs: int = 1):
        # don't do the CallbackList init since it's 1 line, could clone CallbackList at a later date
        BaseCallback.__init__(self, verbose)

        self.num_envs = num_envs
        self.callbacks = [
            SingleEnvRewardCallback(verbose, i, frequency) for i in range(self.num_envs)
        ]

        self.summarize_every = frequency
        self.curr_logging_interval = 0
        self.start_time = time.time_ns()

    @property
    def ready_to_log(self) -> bool:
        """
        Returns whether all environments are ready to log.

        Returns
        -------
        bool
            Whether all environments are ready to log.
        """
        return all([cb.ready_to_log for cb in self.callbacks])

    def _init_callback(self):
        for callback in self.callbacks:
            callback.init_callback(self.model)

    def _on_step(self) -> bool:
        for callback in self.callbacks:
            callback.on_step()

        if self.ready_to_log:
            reward_interval = [
                callback.get_reward_interval() for callback in self.callbacks
            ]
            stepcount_interval = [
                callback.get_step_interval() for callback in self.callbacks
            ]

            self.logger.record(
                "rewards/mean_reward",
                np.mean(reward_interval),
            )

            self.logger.record(
                "rewards/max_reward",
                np.mean(np.max(reward_interval, axis=INTERVAL_AXIS)),
            )

            self.logger.record(
                "rewards/min_reward",
                np.mean(np.min(reward_interval, axis=INTERVAL_AXIS)),
            )

            self.logger.record(
                "rewards/mean_steps",
                np.mean(stepcount_interval),
            )

            time_elapsed = max(
                (time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon
            )

            self.logger.record("time/elapsed_time", time_elapsed)

            for callback in self.callbacks:
                callback.increment_logging_interval()

        return True


import warnings

# Below code is adapted from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/callbacks.py

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

# Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

try:
    from tqdm import TqdmExperimentalWarning

    # Remove experimental warning
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    from tqdm.rich import tqdm
except ImportError:
    # Rich not installed, we only throw an error
    # if the progress bar is used
    tqdm = None


class CustomProgressBarCallback(BaseCallback):
    """
    Adapted version of the Progress bar from Sb3 that starts from the last timestep when resuming training from a checkpoint.

    See Also
    --------
    stable_baselines3.common.callbacks.ProgressBarCallback : The original progress bar callback from stable baselines3.
    """

    pbar: tqdm # type: ignore

    def __init__(self) -> None:
        super().__init__()
        if tqdm is None:
            raise ImportError(
                "You must install tqdm and rich in order to use the progress bar callback. "
                "It is included if you install stable-baselines with the extra packages: "
                "`pip install stable-baselines3[extra]`"
            )

    def _on_training_start(self) -> None:
        # Initialize progress bar
        # Remove timesteps that were done in previous training sessions
        self.pbar = tqdm(  
            initial=self.model.num_timesteps, total=self.locals["total_timesteps"]
        ) # pyright: ignore[reportOptionalCall]

    def _on_step(self) -> bool:
        # Update progress bar, we do num_envs steps per call to `env.step()`
        self.pbar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self) -> None:
        # Flush and close progress bar
        self.pbar.refresh()
        self.pbar.close()


# end of adapted code
