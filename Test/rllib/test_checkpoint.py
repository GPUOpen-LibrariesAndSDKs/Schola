# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Unit tests for RLlib checkpoint family detection and warm-start planning."""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest
from ray.rllib.algorithms.bc import BC
from ray.rllib.algorithms.bc.torch.default_bc_torch_rl_module import (
    DefaultBCTorchRLModule,
)
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.utils.checkpoints import Checkpointable

from schola.rllib.checkpoint import (
    algorithm_class_from_checkpoint,
    algorithm_family,
    assert_warm_start_compatible,
    load_rl_module_from_algorithm_checkpoint,
    plan_resume_from_checkpoint,
    resume_mode_for_checkpoint,
    rl_module_dir_from_algorithm_checkpoint,
)
from schola.rllib.policy_mapping import schola_algorithm_subclass
from schola.scripts.rllib.training import ResourcePlan, TrainingPlan, run_training


def _write_ctor_pickle(checkpoint_dir, cls):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ctor_path = checkpoint_dir / Checkpointable.CLASS_AND_CTOR_ARGS_FILE_NAME
    with ctor_path.open("wb") as ctor_file:
        pickle.dump({"class": cls, "ctor_args_and_kwargs": ((), {})}, ctor_file)
    return checkpoint_dir


def test_algorithm_family_unwraps_schola_ppo():
    assert algorithm_family(PPO) == "PPO"
    assert algorithm_family(schola_algorithm_subclass(PPO)) == "PPO"
    assert algorithm_family(BC) == "BC"


def test_resume_mode_matches_family(tmp_path):
    ppo_dir = _write_ctor_pickle(tmp_path / "ppo", PPO)
    bc_dir = _write_ctor_pickle(tmp_path / "bc", BC)
    schola_ppo = schola_algorithm_subclass(PPO)

    assert resume_mode_for_checkpoint(ppo_dir, schola_ppo) == "restore"
    assert resume_mode_for_checkpoint(bc_dir, schola_ppo) == "warm_start"
    assert resume_mode_for_checkpoint(bc_dir, BC) == "restore"
    assert algorithm_class_from_checkpoint(bc_dir) is BC


def test_plan_resume_restore_does_not_require_rl_module(tmp_path):
    ppo_dir = _write_ctor_pickle(tmp_path / "ppo", PPO)
    restore, warm_start = plan_resume_from_checkpoint(
        ppo_dir,
        schola_algorithm_subclass(PPO),
        PPOConfig().framework("torch"),
        ["default_policy"],
    )
    assert restore == ppo_dir
    assert warm_start is None


def test_plan_resume_warm_start_requires_module_dir(tmp_path):
    bc_dir = _write_ctor_pickle(tmp_path / "bc", BC)
    with pytest.raises(FileNotFoundError, match="No RLModule checkpoint directory"):
        plan_resume_from_checkpoint(
            bc_dir,
            schola_algorithm_subclass(PPO),
            PPOConfig().framework("torch"),
            ["default_policy"],
        )


def test_assert_warm_start_allows_bc_into_ppo():
    loaded = {"default_policy": object.__new__(DefaultBCTorchRLModule)}
    assert_warm_start_compatible(
        loaded, PPOConfig().framework("torch"), ["default_policy"]
    )


def test_assert_warm_start_allows_matching_ppo_module():
    loaded = {"default_policy": object.__new__(DefaultPPOTorchRLModule)}
    assert_warm_start_compatible(
        loaded, PPOConfig().framework("torch"), ["default_policy"]
    )


def test_assert_warm_start_rejects_bc_into_sac():
    loaded = {"default_policy": object.__new__(DefaultBCTorchRLModule)}
    with pytest.raises(ValueError, match="Cannot warm-start"):
        assert_warm_start_compatible(
            loaded, SACConfig().framework("torch"), ["default_policy"]
        )


def test_assert_warm_start_rejects_unknown_module_ids():
    loaded = {"default_policy": object.__new__(DefaultBCTorchRLModule)}
    with pytest.raises(ValueError, match="not among the live policies"):
        assert_warm_start_compatible(loaded, PPOConfig().framework("torch"), ["Tagger"])


def test_run_training_uses_plan_restore_and_warm_start_callback(mocker, tmp_path):
    captured = {}

    def fake_tune_run(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return mocker.Mock()

    mocker.patch("ray.init")
    mocker.patch("ray.shutdown")
    mocker.patch("ray.tune.run", side_effect=fake_tune_run)
    mocker.patch("ray.air.CheckpointConfig", return_value=object())

    args = SimpleNamespace(
        resource_settings=SimpleNamespace(using_cluster=True, num_cpus=1, num_gpus=0),
        checkpoint_settings=SimpleNamespace(
            enable_checkpoints=False,
            save_freq=0,
            save_final_policy=False,
            export_onnx=False,
            should_persist=False,
            storage_path=None,
        ),
        logging_settings=SimpleNamespace(rllib_verbosity=0),
        resume_settings=SimpleNamespace(resume_from=tmp_path / "ignored"),
    )
    config = PPOConfig().framework("torch")
    ckpt = tmp_path / "bc_ckpt"
    ckpt.mkdir()
    plan = TrainingPlan(
        config=config,
        trainable=PPO,
        stop={"num_env_steps_sampled_lifetime": 1},
        resource_plan=ResourcePlan.online(args),
        label="test",
        restore=None,
        warm_start_rl_module_dir=ckpt,
        warm_start_policy_ids=("default_policy",),
    )
    run_training(args, plan)

    assert captured["restore"] is None
    assert captured["config"].callbacks_on_algorithm_init is not None


def test_run_training_passes_plan_restore_to_tune(mocker, tmp_path):
    captured = {}

    def fake_tune_run(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return mocker.Mock()

    mocker.patch("ray.init")
    mocker.patch("ray.shutdown")
    mocker.patch("ray.tune.run", side_effect=fake_tune_run)
    mocker.patch("ray.air.CheckpointConfig", return_value=object())

    restore_dir = tmp_path / "checkpoint_000000"
    restore_dir.mkdir()
    args = SimpleNamespace(
        resource_settings=SimpleNamespace(using_cluster=True, num_cpus=1, num_gpus=0),
        checkpoint_settings=SimpleNamespace(
            enable_checkpoints=False,
            save_freq=0,
            save_final_policy=False,
            export_onnx=False,
            should_persist=False,
            storage_path=None,
        ),
        logging_settings=SimpleNamespace(rllib_verbosity=0),
        resume_settings=SimpleNamespace(resume_from=None),
    )
    plan = TrainingPlan(
        config=PPOConfig().framework("torch"),
        trainable=PPO,
        stop={"num_env_steps_sampled_lifetime": 1},
        resource_plan=ResourcePlan.online(args),
        label="test",
        restore=restore_dir,
    )
    run_training(args, plan)
    assert captured["restore"] == str(restore_dir)


def test_rl_module_dir_missing(tmp_path):
    checkpoint = tmp_path / "checkpoint_000000"
    checkpoint.mkdir()
    with pytest.raises(FileNotFoundError, match="No RLModule checkpoint directory"):
        rl_module_dir_from_algorithm_checkpoint(checkpoint)


def test_load_rl_module_reports_missing_module_id(monkeypatch, tmp_path):
    import schola.rllib.checkpoint as checkpoint_mod

    monkeypatch.setattr(
        checkpoint_mod,
        "load_multi_rl_module_from_algorithm_checkpoint",
        lambda checkpoint: {"other_policy": object()},
    )
    with pytest.raises(FileNotFoundError, match="default_policy"):
        load_rl_module_from_algorithm_checkpoint(tmp_path, module_id="default_policy")
