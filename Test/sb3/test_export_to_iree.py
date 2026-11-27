# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for exporting SB3 policies to IREE (.vmfb).

Mirrors ``test_export_to_onnx.py``. Cases, environments, and helpers shared
by both live in ``export_test_common.py``.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import pytest

pytest.importorskip("iree.turbine")
pytest.importorskip("iree.runtime")

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from schola.sb3.export import save_model_as_iree
from schola.sb3.utils import VecMergeDictActionWrapper
from schola.scripts.sb3.sb3_to_iree import export_iree_app
from Test.conftest import (
    check_iree_metadata,
    check_iree_model_loads,
    check_iree_runtime,
    run_iree_model,
)
from Test.export_test_common import (
    action_from_outputs,
    assert_recorded_actions_match,
    batch_observation,
    box_action_space,
    dict_obs_space,
    discrete_action_space,
)
from Test.sb3.export_test_common import (
    ExportParams,
    action_in_exported_space,
    binary_dict_action_space,
    box_dict_action_space,
    discrete_dict_action_space,
    export_function_cases,
    multi_discrete_dict_action_space,
)


def evaluate_iree_policy(
    vm_module,
    model_metadata,
    env: gym.Env,
    action_space: gym.Space,
    n_eval_episodes: int = 1,
) -> None:
    """IREE counterpart of ``evaluate_onnx_policy``. ``env.step`` records each action."""
    episodes_completed = 0
    observation, _ = env.reset()
    while episodes_completed < n_eval_episodes:
        outputs = run_iree_model(
            vm_module, model_metadata, batch_observation(observation, env.observation_space)
        )
        action = action_from_outputs(outputs, action_space)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            episodes_completed += 1
            if episodes_completed < n_eval_episodes:
                observation, _ = env.reset()


def check_iree_matches_sb3(
    model: BaseAlgorithm,
    vm_module,
    model_metadata,
    params: ExportParams,
    action_space: gym.Space | None = None,
    episode_length: int = 32,
) -> None:
    """Fixed-seed episodes should make the exported policy match SB3.

    ``action_space`` is the space the IREE module emits. It defaults to
    ``params.action_space``. Pass the merged space when the export collapsed a
    dictionary action space and did not keep the original keys.
    """
    exported_action_space = (
        params.action_space if action_space is None else action_space
    )
    env_cls = params.get_env_class(
        record_actions=True, episode_length=episode_length
    )
    eval_env = make_vec_env(env_cls, 1)
    evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=1,
        deterministic=True,
        warn=False,
    )
    sb3_actions: list[Any] = eval_env.get_attr("recorded_actions")[0]

    iree_eval_env = env_cls()
    evaluate_iree_policy(
        vm_module,
        model_metadata,
        iree_eval_env,
        action_space=exported_action_space,
        n_eval_episodes=1,
    )
    assert len(iree_eval_env.recorded_actions) == episode_length
    assert_recorded_actions_match(
        [action_in_exported_space(a, exported_action_space) for a in sb3_actions],
        iree_eval_env.recorded_actions,
        exported_action_space,
    )


class TestExportFunction:
    # Test exporting SB3 policies to IREE, tests with every algorithm from SB3
    @pytest.mark.parametrize("export_params", export_function_cases, ids=str)
    def test_export_sb3_policy_to_iree(self, tmp_path, export_params):
        params: ExportParams = export_params

        env = make_vec_env(
            params.get_env_class(), 2, seed=1
        )  # prevents an error with automatic seeding
        if isinstance(params.action_space, gym.spaces.Dict):
            env = VecMergeDictActionWrapper(env)

        model: BaseAlgorithm = params.make_algo()(params.policy_type, env, verbose=1)

        save_model_as_iree(
            model, str(tmp_path / "test.vmfb"), action_space=params.action_space
        )

        vm_module, model_metadata = check_iree_model_loads(tmp_path / "test.vmfb")
        check_iree_metadata(
            model_metadata,
            params.observation_space,
            params.action_space,
            state_shapes=params.state_shapes,
            metadata=params.metadata,
        )
        check_iree_runtime(
            vm_module,
            model_metadata,
            params.observation_space,
            params.action_space,
            state_shapes=params.state_shapes,
        )
        check_iree_matches_sb3(model, vm_module, model_metadata, params)


class TestExportScript:

    @pytest.mark.parametrize(
        "export_params",
        [
            # Testing for merged dict action spaces
            ExportParams("ppo", dict_obs_space, box_action_space),
            ExportParams("sac", dict_obs_space, box_action_space),
            ExportParams("td3", dict_obs_space, box_action_space),
            ExportParams("ddpg", dict_obs_space, box_action_space),
            ExportParams("a2c", dict_obs_space, box_action_space),
            ExportParams("dqn", dict_obs_space, discrete_action_space),
        ],
        ids=str,
    )
    def test_algorithms(self, tmp_path, export_params):
        params: ExportParams = export_params
        env = make_vec_env(params.get_env_class(), 2, seed=1)
        model: BaseAlgorithm = params.make_algo()(params.policy_type, env, verbose=1)
        model.save(tmp_path / "test.zip")
        export_iree_app(
            [
                f"{tmp_path}/test.zip",
                f"{tmp_path}/test.vmfb",
                "--algorithm",
                params.algo_name.upper(),
            ],
            result_action="return_value",
            exit_on_error=False,
        )

        vm_module, model_metadata = check_iree_model_loads(tmp_path / "test.vmfb")
        check_iree_metadata(
            model_metadata,
            params.observation_space,
            params.action_space,
            state_shapes=params.state_shapes,
            metadata=params.metadata,
        )
        check_iree_runtime(
            vm_module,
            model_metadata,
            params.observation_space,
            params.action_space,
            state_shapes=params.state_shapes,
        )
        check_iree_matches_sb3(model, vm_module, model_metadata, params)

    @pytest.mark.parametrize(
        "export_params",
        [
            ExportParams("ppo", dict_obs_space, box_dict_action_space),
            ExportParams("ppo", dict_obs_space, discrete_dict_action_space),
            ExportParams("ppo", dict_obs_space, binary_dict_action_space),
            ExportParams("ppo", dict_obs_space, multi_discrete_dict_action_space),
        ],
        ids=str,
    )
    def test_merged_dict_action_space(self, tmp_path, export_params):
        params: ExportParams = export_params

        env = make_vec_env(params.get_env_class(), 2, seed=1)
        env = VecMergeDictActionWrapper(env)

        model: BaseAlgorithm = params.make_algo()(params.policy_type, env, verbose=1)
        setattr(model, "__unmerged_action_space", env.unmerged_action_space)
        model.save(tmp_path / "test.zip")

        export_iree_app(
            [
                f"{tmp_path}/test.zip",
                f"{tmp_path}/test.vmfb",
                "--algorithm",
                params.algo_name.upper(),
            ],
            result_action="return_value",
            exit_on_error=False,
        )
        vm_module, model_metadata = check_iree_model_loads(tmp_path / "test.vmfb")
        check_iree_metadata(
            model_metadata,
            params.observation_space,
            params.action_space,
            state_shapes=params.state_shapes,
            metadata=params.metadata,
        )
        check_iree_runtime(
            vm_module,
            model_metadata,
            params.observation_space,
            params.action_space,
            state_shapes=params.state_shapes,
        )
        check_iree_matches_sb3(model, vm_module, model_metadata, params)

    @pytest.mark.parametrize(
        "export_params",
        [
            ExportParams("ppo", dict_obs_space, box_dict_action_space),
        ],
        ids=str,
    )
    def test_merge_with_no_hidden_state(self, tmp_path, export_params):
        params: ExportParams = export_params
        env = make_vec_env(params.get_env_class(), 2, seed=1)
        env = VecMergeDictActionWrapper(env)
        model: BaseAlgorithm = params.make_algo()(params.policy_type, env, verbose=1)
        model.save(tmp_path / "test.zip")
        export_iree_app(
            [
                f"{tmp_path}/test.zip",
                f"{tmp_path}/test.vmfb",
                "--algorithm",
                params.algo_name.upper(),
            ],
            result_action="return_value",
            exit_on_error=False,
        )
        # Check that the action space is the same as the merged action space, since the original shape wasn't provided
        vm_module, model_metadata = check_iree_model_loads(tmp_path / "test.vmfb")
        check_iree_metadata(
            model_metadata,
            params.observation_space,
            env.action_space,
            state_shapes=params.state_shapes,
            metadata=params.metadata,
        )
        check_iree_runtime(
            vm_module,
            model_metadata,
            params.observation_space,
            env.action_space,
            state_shapes=params.state_shapes,
        )
        check_iree_matches_sb3(
            model, vm_module, model_metadata, params, action_space=env.action_space
        )
