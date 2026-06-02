# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Evaluate a trained Stable-Baselines3 policy against a Schola-backed environment.
"""

import logging
from typing import List, Tuple, cast, Any

from cyclopts import App
from schola.scripts.common.command_template import MetaAlgCommand
from schola.scripts.sb3.settings import BasePPOSettings, BaseSACSettings
from schola.scripts.sb3.eval.settings import Sb3EvalScriptSettings

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

logger = logging.getLogger(__name__)


def main(args: Sb3EvalScriptSettings) -> Tuple[float, float]:
    """
    Load a saved SB3 policy and run ``stable_baselines3.common.evaluation.evaluate_policy``.

    Parameters
    ----------
    args : Sb3EvalScriptSettings
        CLI / script configuration.

    Returns
    -------
    tuple[float, float]
        Mean and standard deviation of episodic return over ``n_eval_episodes``.
    """

    import gymnasium as gym
    import numpy as np
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
    from stable_baselines3.common.vec_env import VecNormalize

    from schola.core.error_manager import ScholaErrorContextManager
    from schola.sb3.async_env import AsyncVecEnv
    from schola.sb3.env import VecEnv
    from schola.sb3.utils import VecMergeDictActionWrapper
    from schola.core.simulators.unreal.executable_simulator import UnrealExecutable

    env = None
    try:
        with ScholaErrorContextManager():
            sim_args = args.environment_settings.simulator_settings
            protocol_args = args.environment_settings.protocol_settings
            n_sim = sim_args.num_simulators

            if n_sim == 1:
                env = VecEnv(
                    sim_args.make(),
                    protocol_args.make(),
                    verbosity=args.logging_settings.schola_verbosity,
                )
            else:
                primary = cast(UnrealExecutable, sim_args.make())
                simulators = [primary] + primary.spawn_executables(n_sim - 1)
                async_protocols = protocol_args.make_n_async(n_sim)
                env = AsyncVecEnv(
                    simulators,
                    async_protocols,
                    verbosity=args.logging_settings.schola_verbosity,
                )

            if isinstance(env.action_space, gym.spaces.Dict):
                logger.warning(
                    "SB3 doesn't support dictionary action spaces natively. Merging actions "
                    "for evaluation (same wrapper as training)."
                )
                env = VecMergeDictActionWrapper(env)

            model: BaseAlgorithm = args.algorithm_settings.constructor.load(
                str(args.checkpoint), env=env
            )

            if args.vecnormalize is not None:
                env = VecNormalize.load(str(args.vecnormalize), env)
                env.training = False
                env.norm_reward = True
                model.set_env(env)

            if args.environment_settings.env_options:
                # Inherited from SB3's `set_options`
                env.set_options(options=args.environment_settings.env_options)

            monitored = VecMonitor(env)
            ev: tuple[List[float], List[float]] = evaluate_policy(
                model,
                monitored,
                n_eval_episodes=args.n_eval_episodes,
                deterministic=args.deterministic,
                return_episode_rewards=True,
            )  # type: ignore

            episode_rewards, episode_lengths = ev[0], ev[1]
            mean_reward = float(np.mean(episode_rewards))
            std_reward = float(np.std(episode_rewards))

            logger.info(
                "Evaluation complete: mean_reward=%.4f +/- %.4f (over %d episodes)",
                mean_reward,
                std_reward,
                args.n_eval_episodes,
            )
            # print out the per episode rewards
            per_episode_reward_str = "Per episode rewards: \n"
            for episode_reward, episode_length in zip(episode_rewards, episode_lengths):  # type: ignore
                per_episode_reward_str += f"\tEpisode reward: {episode_reward:.4f}, Episode length: {episode_length}\n"
            logger.info(per_episode_reward_str)

            monitored.close()
            return mean_reward, std_reward
    except Exception:
        if env is not None:
            env.close()
        raise


app = App(name="eval", help="Evaluate a trained Stable-Baselines3 policy")


class MetaEvalSB3Command(MetaAlgCommand[Sb3EvalScriptSettings]):
    """
    ``MetaEvalSB3Command`` configuration for Stable-Baselines3 (PPO and SAC).

    See Also
    --------
    MetaAlgCommand
    """

    @property
    def algorithm_table(self):
        return {
            "sac": BaseSACSettings,
            "ppo": BasePPOSettings,
        }

    @property
    def algorithm_help(self):
        return {
            "sac": "Evaluate a model trained using Soft Actor-Critic(SAC) with StableBaselines3.",
            "ppo": "Evaluate a model trained using Proximal Policy Optimization(PPO) with StableBaselines3.",
        }


app = MetaEvalSB3Command(app, Sb3EvalScriptSettings, main, logger).make()
