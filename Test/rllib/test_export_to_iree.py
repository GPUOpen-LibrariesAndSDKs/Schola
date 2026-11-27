# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for exporting RLlib policies to IREE (.vmfb).

Mirrors ``test_export_to_onnx.py``. Cases, environments, fixtures, and helpers
shared by both live in ``export_test_common.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import pytest

pytest.importorskip("iree.turbine")
pytest.importorskip("iree.runtime")

from ray.rllib import Policy
from ray.rllib.algorithms.algorithm import Algorithm

from schola.rllib.export import export_iree_from_policy
from schola.scripts.rllib.rllib_to_iree import export_iree_app
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
    box_obs_space,
    dict_obs_space,
    discrete_obs_space,
    multi_discrete_obs_space,
    offset_box_action_space,
)
from Test.rllib.export_test_common import (
    ExportParams,
    dict_action_space,
    initial_state,
    make_algo,
    make_algo_config,
    record_new_api_actions,
    record_old_api_actions,
    rl_module_export_cases,
)


def evaluate_iree_policy(
    vm_module,
    model_metadata,
    env: gym.Env,
    action_space: gym.Space,
    state_shapes: Dict[str, Tuple[int, ...]] | None = None,
    n_eval_episodes: int = 1,
) -> None:
    """Step ``env`` with an exported IREE policy, carrying recurrent state when present."""
    state = initial_state(state_shapes or {})
    episodes_completed = 0
    observation, _ = env.reset()
    while episodes_completed < n_eval_episodes:
        output_map = run_iree_model(
            vm_module,
            model_metadata,
            batch_observation(observation, env.observation_space) | state,
        )
        for name in state:
            state[name] = output_map["state_out_" + name[len("state_in_") :]]
        action = action_from_outputs(output_map, action_space)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            episodes_completed += 1
            if episodes_completed < n_eval_episodes:
                observation, _ = env.reset()
                state = initial_state(state_shapes or {})


def check_iree_matches_rllib(
    algo: Algorithm,
    vm_module,
    model_metadata,
    action_space: gym.Space,
    state_shapes: Dict[str, Tuple[int, ...]] | None = None,
    episode_length: int = 32,
) -> None:
    """Fixed-seed episodes should make the exported policy match RLlib."""
    assert algo.config is not None, "Unexpected Error, config was None during test"
    if algo.config.enable_env_runner_and_connector_v2:
        rllib_actions, env_cls = record_new_api_actions(
            algo, action_space, episode_length
        )
    else:
        rllib_actions, env_cls = record_old_api_actions(algo, episode_length)

    iree_env = env_cls()
    iree_env.record_actions = True
    iree_env.episode_length = episode_length
    evaluate_iree_policy(
        vm_module,
        model_metadata,
        iree_env,
        action_space,
        state_shapes=state_shapes,
    )
    assert len(iree_env.recorded_actions) == episode_length
    assert_recorded_actions_match(
        rllib_actions,
        iree_env.recorded_actions,
        action_space,
    )


def check_iree_export(
    vmfb_path,
    alg_params: ExportParams,
) -> tuple[Any, Any]:
    """Load ``vmfb_path`` and run the metadata and runtime checks for ``alg_params``."""
    vm_module, model_metadata = check_iree_model_loads(vmfb_path)
    check_iree_metadata(
        model_metadata,
        alg_params.observation_space,
        alg_params.action_space,
        state_shapes=alg_params.state_shapes,
        metadata=alg_params.metadata,
    )
    check_iree_runtime(
        vm_module,
        model_metadata,
        alg_params.observation_space,
        alg_params.action_space,
        state_shapes=alg_params.state_shapes,
    )
    return vm_module, model_metadata


@pytest.mark.xdist_group(name="ray-cluster")
class TestExportFunctions:
    # Test exporting RLlib policies to IREE
    @pytest.mark.parametrize(
        "alg_params",
        [
            # Check different observation spaces with box action space
            ExportParams("ppo", box_obs_space, box_action_space, new_stack=False),
            ExportParams("ppo", discrete_obs_space, box_action_space, new_stack=False),
            ExportParams("ppo", dict_obs_space, dict_action_space, new_stack=False),
            ExportParams(
                "ppo", multi_discrete_obs_space, box_action_space, new_stack=False
            ),
            ExportParams("sac", box_obs_space, box_action_space, new_stack=False),
            ExportParams(
                "sac",
                box_obs_space,
                offset_box_action_space,
                new_stack=False,
                custom_id="SAC(Box, BoxWithOffset, old)",
            ),
        ],
        ids=str,
    )
    def test_export_rllib_policy_to_iree(
        self, tmp_path, make_algo, alg_params, ray_cluster
    ):
        """Test exporting RLlib policies to IREE format."""
        algo = make_algo(alg_params)
        policy: Policy = algo.get_policy("default_policy")
        vmfb_file = tmp_path / "test_policy" / "model.vmfb"

        export_iree_from_policy(
            policy, vmfb_file, observation_space=alg_params.observation_space
        )

        vm_module, model_metadata = check_iree_export(vmfb_file, alg_params)
        check_iree_matches_rllib(
            algo,
            vm_module,
            model_metadata,
            alg_params.action_space,
            state_shapes=alg_params.state_shapes,
        )

    # test exporting just the RLModule to IREE (direct creation without Ray cluster)
    @pytest.mark.parametrize("alg_params", rl_module_export_cases, ids=str)
    def test_export_rllib_rl_module_to_iree(
        self, tmp_path, make_algo, alg_params, ray_cluster
    ):
        """Test exporting RLlib RLModule to IREE format (no Ray cluster spawning)."""
        algo = make_algo(alg_params)
        vmfb_file = tmp_path / "model.vmfb"
        export_iree_from_policy(
            algo.get_module(), vmfb_file, observation_space=alg_params.observation_space
        )
        vm_module, model_metadata = check_iree_export(vmfb_file, alg_params)
        check_iree_matches_rllib(
            algo,
            vm_module,
            model_metadata,
            alg_params.action_space,
            state_shapes=alg_params.state_shapes,
        )

    # test exporting using the entire algorithm object
    @pytest.mark.parametrize(
        "alg_params",
        [
            # Check different action spaces with dict observation space
            # Test with LSTM enabled
            ExportParams("ppo", box_obs_space, box_action_space, use_lstm=True),
            ExportParams("sac", dict_obs_space, box_action_space),
            ExportParams(
                "sac",
                dict_obs_space,
                offset_box_action_space,
                custom_id="SAC(Dict, BoxWithOffset)",
            ),
        ],
        ids=str,
    )
    def test_export_rllib_algorithm_to_iree(
        self, tmp_path, make_algo, alg_params, ray_cluster
    ):
        """Test exporting RLlib algorithms to IREE format."""
        algo = make_algo(alg_params)
        vmfb_file = tmp_path / "model.vmfb"
        export_iree_from_policy(algo, vmfb_file)

        vm_module, model_metadata = check_iree_export(vmfb_file, alg_params)
        check_iree_matches_rllib(
            algo,
            vm_module,
            model_metadata,
            alg_params.action_space,
            state_shapes=alg_params.state_shapes,
        )

    @pytest.mark.parametrize(
        "alg_params",
        [
            ExportParams("ppo", box_obs_space, box_action_space, num_agents=2),
            ExportParams("ppo", dict_obs_space, dict_action_space, num_agents=2),
            ExportParams(
                "ppo", box_obs_space, box_action_space, num_agents=2, num_policies=1
            ),
        ],
        ids=str,
    )
    def test_export_rllib_multiagent_to_iree(
        self, tmp_path, make_algo, alg_params, ray_cluster
    ):
        """Test exporting RLlib multiagent policies to IREE format."""
        algo = make_algo(alg_params)
        export_iree_from_policy(algo, tmp_path)

        for i in range(alg_params.num_policies):
            check_iree_export(tmp_path / f"policy_{i}.vmfb", alg_params)


@pytest.mark.xdist_group(name="ray-cluster")
class TestExportScript:
    """End-to-end tests for the rllib-to-iree CLI script."""

    @pytest.mark.parametrize(
        "alg_params",
        [
            ExportParams("ppo", box_obs_space, box_action_space),
            ExportParams("ppo", dict_obs_space, dict_action_space, use_lstm=True),
            ExportParams("sac", box_obs_space, box_action_space),
            ExportParams(
                "sac",
                box_obs_space,
                offset_box_action_space,
                custom_id="SAC(Box, BoxWithOffset)",
            ),
        ],
        ids=str,
    )
    def test_rllib_to_iree_script(
        self, tmp_path, make_algo, alg_params, ray_cluster
    ):
        """Test the rllib-to-iree script: save checkpoint, run script, verify IREE output."""
        algo: Algorithm = make_algo(alg_params)
        checkpoint_path = tmp_path / "checkpoint"
        export_path = tmp_path / "model.vmfb"
        algo.save(checkpoint_path)
        export_iree_app(
            [str(checkpoint_path), str(export_path)],
            result_action="return_value",
            exit_on_error=False,
        )
        vm_module, model_metadata = check_iree_export(export_path, alg_params)
        check_iree_matches_rllib(
            algo,
            vm_module,
            model_metadata,
            alg_params.action_space,
            state_shapes=alg_params.state_shapes,
        )


class TestExportIreeFromPolicyEdgeCases:
    """Tests for edge cases in export_iree_from_policy"""

    def test_export_fails_with_unsupported_type(self, tmp_path):
        """Test that export raises TypeError for unsupported argument types"""
        path = tmp_path / "output.vmfb"

        # Try with an integer (unsupported type)
        with pytest.raises(
            TypeError, match="Cannot export IREE from Policy/Checkpoint"
        ):
            export_iree_from_policy(123, path)

    def test_export_fails_with_list(self, tmp_path):
        """Test that export raises TypeError for list argument"""
        path = tmp_path / "output.vmfb"

        with pytest.raises(
            TypeError, match="Cannot export IREE from Policy/Checkpoint"
        ):
            export_iree_from_policy([1, 2, 3], path)

    def test_export_fails_with_none(self, tmp_path):
        """Test that export raises TypeError for None argument"""
        path = tmp_path / "output.vmfb"

        with pytest.raises(
            TypeError, match="Cannot export IREE from Policy/Checkpoint"
        ):
            export_iree_from_policy(None, path)
