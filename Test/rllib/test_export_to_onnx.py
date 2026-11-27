# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for exporting RLlib policies to ONNX"""

from typing import Dict, Tuple
import pytest
import gymnasium as gym
import numpy as np
import onnx
from ray.rllib import Policy
from ray.rllib.algorithms.algorithm import Algorithm
from schola.rllib.export import export_onnx_from_policy
from schola.scripts.rllib.rllib_to_onnx import export_onnx_app
from Test.conftest import (
    check_onnx_metadata,
    check_onnx_model_loads,
    check_onnx_runtime,
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


def evaluate_onnx_policy(
    model: onnx.ModelProto,
    env: gym.Env,
    action_space: gym.Space,
    state_shapes: Dict[str, Tuple[int, ...]] | None = None,
    n_eval_episodes: int = 1,
) -> None:
    """Step ``env`` with an exported ONNX policy, carrying recurrent state when present."""
    import onnxruntime as ort

    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    output_names = [output.name for output in model.graph.output]
    state = initial_state(state_shapes or {})
    episodes_completed = 0
    observation, _ = env.reset()
    while episodes_completed < n_eval_episodes:
        outputs = session.run(
            None,
            batch_observation(observation, env.observation_space) | state,
        )
        output_map = dict(zip(output_names, outputs))
        for name in state:
            state[name] = np.asarray(output_map["state_out_" + name[len("state_in_") :]])
        action = action_from_outputs(output_map, action_space)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            episodes_completed += 1
            if episodes_completed < n_eval_episodes:
                observation, _ = env.reset()
                state = initial_state(state_shapes or {})


def check_onnx_matches_rllib(
    algo: Algorithm,
    model: onnx.ModelProto,
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

    onnx_env = env_cls()
    onnx_env.record_actions = True
    onnx_env.episode_length = episode_length
    evaluate_onnx_policy(
        model,
        onnx_env,
        action_space,
        state_shapes=state_shapes,
    )
    assert len(onnx_env.recorded_actions) == episode_length
    assert_recorded_actions_match(
        rllib_actions,
        onnx_env.recorded_actions,
        action_space,
    )


@pytest.mark.xdist_group(name="ray-cluster")
class TestExportFunctions:
    # Test exporting RLlib policies to ONNX
    @pytest.mark.parametrize(
        "alg_params",
        [
            # Check different observation spaces with box action space
            ExportParams("ppo", box_obs_space, box_action_space, new_stack=False),
            # TestExportParams("ppo", box_obs_space, box_action_space, use_attention=True, new_stack=False),
            # TestExportParams("ppo", dict_obs_space, dict_action_space, use_attention=True, new_stack=False),
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
    def test_export_rllib_policy_to_onnx(
        self, tmp_path, make_algo, alg_params, ray_cluster
    ):
        """Test exporting RLlib policies to ONNX format."""

        # Get the policy
        algo = make_algo(alg_params)
        policy: Policy = algo.get_policy("default_policy")
        onnx_file = tmp_path / "test_policy" / "model.onnx"

        # Export to ONNX
        export_onnx_from_policy(
            policy, onnx_file, observation_space=alg_params.observation_space
        )

        # Verify the ONNX file was created
        onnx_model = check_onnx_model_loads(onnx_file)
        check_onnx_metadata(
            onnx_model,
            alg_params.observation_space,
            alg_params.action_space,
            state_shapes=alg_params.state_shapes,
            metadata=alg_params.metadata,
        )
        check_onnx_runtime(
            onnx_model,
            alg_params.observation_space,
            alg_params.action_space,
            state_shapes=alg_params.state_shapes,
        )
        check_onnx_matches_rllib(
            algo,
            onnx_model,
            alg_params.action_space,
            state_shapes=alg_params.state_shapes,
        )

    # test exporting just the RLModule to ONNX (direct creation without Ray cluster)
    @pytest.mark.parametrize("alg_params", rl_module_export_cases, ids=str)
    def test_export_rllib_rl_module_to_onnx(
        self, tmp_path, make_algo, alg_params, ray_cluster
    ):
        """Test exporting RLlib RLModule to ONNX format (no Ray cluster spawning)."""
        algo = make_algo(alg_params)
        onnx_file = tmp_path / "model.onnx"
        # Export to ONNX - uses the Algorithm export path which calls get_module()
        export_onnx_from_policy(
            algo.get_module(), onnx_file, observation_space=alg_params.observation_space
        )
        # TODO handle exporting vision models
        # Verify the ONNX file was created
        onnx_model = check_onnx_model_loads(onnx_file)
        check_onnx_metadata(
            onnx_model,
            alg_params.observation_space,
            alg_params.action_space,
            state_shapes=alg_params.state_shapes,
            metadata=alg_params.metadata,
        )
        check_onnx_runtime(
            onnx_model,
            alg_params.observation_space,
            alg_params.action_space,
            state_shapes=alg_params.state_shapes,
        )
        check_onnx_matches_rllib(
            algo,
            onnx_model,
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
    def test_export_rllib_algorithm_to_onnx(
        self, tmp_path, make_algo, alg_params, ray_cluster
    ):
        """Test exporting RLlib algorithms to ONNX format from checkpoint using tune.run()."""
        algo = make_algo(alg_params)
        onnx_file = tmp_path / "model.onnx"
        # Export the RLModule from the algorithm object
        export_onnx_from_policy(algo, onnx_file)

        # Verify the ONNX file was created
        onnx_model = check_onnx_model_loads(onnx_file)
        check_onnx_metadata(
            onnx_model,
            alg_params.observation_space,
            alg_params.action_space,
            state_shapes=alg_params.state_shapes,
            metadata=alg_params.metadata,
        )
        check_onnx_runtime(
            onnx_model,
            alg_params.observation_space,
            alg_params.action_space,
            state_shapes=alg_params.state_shapes,
        )
        check_onnx_matches_rllib(
            algo,
            onnx_model,
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
    def test_export_rllib_multiagent_to_onnx(
        self, tmp_path, make_algo, alg_params, ray_cluster
    ):
        """Test exporting RLlib multiagent policies to ONNX format."""
        algo = make_algo(alg_params)
        export_onnx_from_policy(algo, tmp_path)

        for i in range(alg_params.num_policies):
            onnx_model = check_onnx_model_loads(tmp_path / f"policy_{i}.onnx")
            check_onnx_metadata(
                onnx_model,
                alg_params.observation_space,
                alg_params.action_space,
                state_shapes=alg_params.state_shapes,
                metadata=alg_params.metadata,
            )
            check_onnx_runtime(
                onnx_model,
                alg_params.observation_space,
                alg_params.action_space,
                state_shapes=alg_params.state_shapes,
            )


@pytest.mark.xdist_group(name="ray-cluster")
class TestExportScript:
    """End-to-end tests for the rllib-to-onnx CLI script."""

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
    def test_rllib_to_onnx_script(
        self, tmp_path, make_algo, alg_params, ray_cluster
    ):
        """Test the rllib-to-onnx script: save checkpoint, run script, verify ONNX output."""
        algo: Algorithm = make_algo(alg_params)
        checkpoint_path = tmp_path / "checkpoint"
        export_path = tmp_path / "model.onnx"
        algo.save(checkpoint_path)
        export_onnx_app(
            [str(checkpoint_path), str(export_path)],
            result_action="return_value",
            exit_on_error=False,
        )
        onnx_model = check_onnx_model_loads(export_path)
        check_onnx_metadata(
            onnx_model,
            alg_params.observation_space,
            alg_params.action_space,
            state_shapes=alg_params.state_shapes,
            metadata=alg_params.metadata,
        )
        check_onnx_runtime(
            onnx_model,
            alg_params.observation_space,
            alg_params.action_space,
            state_shapes=alg_params.state_shapes,
        )
        check_onnx_matches_rllib(
            algo,
            onnx_model,
            alg_params.action_space,
            state_shapes=alg_params.state_shapes,
        )


class TestExportOnnxFromPolicyEdgeCases:
    """Tests for edge cases in export_onnx_from_policy"""

    def test_export_fails_with_unsupported_type(self, tmp_path):
        """Test that export raises TypeError for unsupported argument types"""
        path = tmp_path / "output.onnx"

        # Try with an integer (unsupported type)
        with pytest.raises(
            TypeError, match="Cannot export ONNX from Policy/Checkpoint"
        ):
            export_onnx_from_policy(123, path)

    def test_export_fails_with_list(self, tmp_path):
        """Test that export raises TypeError for list argument"""
        path = tmp_path / "output.onnx"

        with pytest.raises(
            TypeError, match="Cannot export ONNX from Policy/Checkpoint"
        ):
            export_onnx_from_policy([1, 2, 3], path)

    def test_export_fails_with_none(self, tmp_path):
        """Test that export raises TypeError for None argument"""
        path = tmp_path / "output.onnx"

        with pytest.raises(
            TypeError, match="Cannot export ONNX from Policy/Checkpoint"
        ):
            export_onnx_from_policy(None, path)
