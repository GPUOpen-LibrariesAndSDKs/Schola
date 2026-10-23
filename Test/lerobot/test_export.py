# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import onnx_ir as ir
import pytest
import torch as th
from gymnasium.spaces import Box, Dict
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.processor.normalize_processor import (
    NormalizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot_env_schola.export import LeRobotACTScholaModule
from lerobot_env_schola.export.act import action_chunking_metadata
from lerobot_env_schola.export.export import check_observations_cover_features


@pytest.fixture
def make_normalizers():
    def _make(
        *,
        features: dict[str, FeatureType],
        norm_map: dict[FeatureType, NormalizationMode],
        stats: dict[str, dict[str, th.Tensor]],
    ) -> tuple[NormalizerProcessorStep, UnnormalizerProcessorStep]:
        policy_features = {
            key: PolicyFeature(
                type=feature_type, shape=tuple(next(iter(stats[key].values())).shape)
            )
            for key, feature_type in features.items()
        }
        action_features = {
            key: feature
            for key, feature in policy_features.items()
            if feature.type == FeatureType.ACTION
        }
        return (
            NormalizerProcessorStep(
                features=policy_features, norm_map=norm_map, stats=stats
            ),
            UnnormalizerProcessorStep(
                features=action_features, norm_map=norm_map, stats=stats
            ),
        )

    return _make


@pytest.fixture
def make_act_module():
    def _make(
        *,
        state_dim: int,
        action_dim: int,
        chunk: th.Tensor | None = None,
        observation_space: Dict | None = None,
        normalizers: (
            tuple[NormalizerProcessorStep, UnnormalizerProcessorStep] | None
        ) = None,
    ) -> tuple[LeRobotACTScholaModule, MagicMock]:
        if observation_space is None:
            observation_space = Dict(
                {"observation.state": Box(-1, 1, shape=(state_dim,), dtype=np.float32)}
            )
        action_space = Dict(
            {"action": Box(-1, 1, shape=(action_dim,), dtype=np.float32)}
        )
        policy = MagicMock()
        policy.eval.return_value = policy
        if chunk is not None:
            policy.predict_action_chunk.return_value = chunk
        normalizer, unnormalizer = normalizers or (None, None)
        return (
            LeRobotACTScholaModule(
                policy,
                observation_space,
                action_space,
                normalizer=normalizer,
                unnormalizer=unnormalizer,
            ),
            policy,
        )

    return _make


def test_act_module_returns_full_chunk(make_act_module):
    chunk = th.arange(24, dtype=th.float32).reshape(2, 4, 3)
    model, policy = make_act_module(state_dim=2, action_dim=3, chunk=chunk)
    assert not model.is_stateful
    (out,) = model.forward(th.zeros(2, 2))
    th.testing.assert_close(out, chunk)
    batch = policy.predict_action_chunk.call_args.args[0]
    assert list(batch) == ["observation.state"]


def test_act_module_maps_model_input_names_to_lerobot_batch_keys(make_act_module):
    observation_space = Dict(
        {
            "state": Box(-1, 1, shape=(2,), dtype=np.float32),
            "images.wrist": Box(-1, 1, shape=(3, 4, 4), dtype=np.float32),
        }
    )
    model, policy = make_act_module(
        state_dim=2,
        action_dim=3,
        chunk=th.zeros(1, 4, 3),
        observation_space=observation_space,
    )

    inputs = {
        "state": th.zeros(1, 2),
        "images.wrist": th.zeros(1, 3, 4, 4),
    }
    model.forward(*(inputs[key] for key in model.input_obs_keys))

    batch = policy.predict_action_chunk.call_args.args[0]
    assert set(batch) == {
        "observation.state",
        "observation.images.wrist",
    }


def test_act_module_applies_lerobot_mean_std(make_act_module, make_normalizers):
    chunk = th.zeros(1, 4, 3)
    state_mean = th.tensor([10.0, 20.0])
    state_std = th.tensor([2.0, 4.0])
    env_mean = th.tensor([0.5])
    env_std = th.tensor([0.25])
    action_mean = th.tensor([1.0, 2.0, 3.0])
    action_std = th.tensor([4.0, 5.0, 6.0])
    observation_space = Dict(
        {
            "observation.state": Box(-1, 1, shape=(2,), dtype=np.float32),
            "observation.environment_state": Box(-1, 1, shape=(1,), dtype=np.float32),
        }
    )
    model, policy = make_act_module(
        state_dim=2,
        action_dim=3,
        chunk=chunk,
        observation_space=observation_space,
        normalizers=make_normalizers(
            features={
                "observation.state": FeatureType.STATE,
                "observation.environment_state": FeatureType.ENV,
                "action": FeatureType.ACTION,
            },
            norm_map={
                FeatureType.STATE: NormalizationMode.MEAN_STD,
                FeatureType.ENV: NormalizationMode.MEAN_STD,
                FeatureType.ACTION: NormalizationMode.MEAN_STD,
            },
            stats={
                "observation.state": {"mean": state_mean, "std": state_std},
                "observation.environment_state": {"mean": env_mean, "std": env_std},
                "action": {"mean": action_mean, "std": action_std},
            },
        ),
    )
    state = th.tensor([[14.0, 12.0]])
    env = th.tensor([[1.0]])
    values = {
        "observation.state": state,
        "observation.environment_state": env,
    }
    (action,) = model.forward(*(values[key] for key in model.input_obs_keys))
    seen = policy.predict_action_chunk.call_args.args[0]
    th.testing.assert_close(
        seen["observation.state"],
        (state - state_mean) / (state_std + 1e-8),
    )
    th.testing.assert_close(
        seen["observation.environment_state"],
        (env - env_mean) / (env_std + 1e-8),
    )
    th.testing.assert_close(action, action_mean.view(1, 1, 3).expand(1, 4, 3))


@pytest.mark.parametrize(
    ("mode", "stats"),
    [
        (
            NormalizationMode.MIN_MAX,
            {"min": th.tensor([2.0]), "max": th.tensor([6.0])},
        ),
        (
            NormalizationMode.QUANTILES,
            {"q01": th.tensor([2.0]), "q99": th.tensor([6.0])},
        ),
        (
            NormalizationMode.QUANTILE10,
            {"q10": th.tensor([2.0]), "q90": th.tensor([6.0])},
        ),
    ],
)
def test_act_module_applies_lerobot_range_normalization(
    make_act_module, make_normalizers, mode, stats
):
    model, policy = make_act_module(
        state_dim=1,
        action_dim=1,
        chunk=th.zeros(1, 1, 1),
        normalizers=make_normalizers(
            features={
                "observation.state": FeatureType.STATE,
                "action": FeatureType.ACTION,
            },
            norm_map={FeatureType.STATE: mode, FeatureType.ACTION: mode},
            stats={"observation.state": stats, "action": stats},
        ),
    )

    (action,) = model.forward(th.tensor([[4.0]]))
    seen = policy.predict_action_chunk.call_args.args[0]
    th.testing.assert_close(seen["observation.state"], th.zeros(1, 1))
    th.testing.assert_close(action, th.tensor([[[4.0]]]))


def test_act_module_scales_uint8_images_to_unit_range(make_act_module):
    observation_space = Dict(
        {
            "state": Box(-1, 1, shape=(2,), dtype=np.float32),
            "images.wrist": Box(0, 255, shape=(3, 2, 2), dtype=np.uint8),
        }
    )
    model, policy = make_act_module(
        state_dim=2,
        action_dim=3,
        chunk=th.zeros(1, 4, 3),
        observation_space=observation_space,
    )
    inputs = {
        "state": th.tensor([[3.0, 4.0]]),
        "images.wrist": th.full((1, 3, 2, 2), 255, dtype=th.uint8),
    }
    model.forward(*(inputs[key] for key in model.input_obs_keys))

    batch = policy.predict_action_chunk.call_args.args[0]
    th.testing.assert_close(batch["observation.images.wrist"], th.ones(1, 3, 2, 2))
    th.testing.assert_close(batch["observation.state"], inputs["state"])


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param(
            ACTConfig(),
            {
                "schola.action_chunking.chunk_mode": "queue",
                "schola.action_chunking.num_action_steps": "100",
            },
            id="default-queue",
        ),
        pytest.param(
            ACTConfig(chunk_size=50, n_action_steps=20),
            {
                "schola.action_chunking.chunk_mode": "queue",
                "schola.action_chunking.num_action_steps": "20",
            },
            id="partial-queue",
        ),
        pytest.param(
            ACTConfig(n_action_steps=1, temporal_ensemble_coeff=0.01),
            {
                "schola.action_chunking.chunk_mode": "temporal_ensemble",
                "schola.action_chunking.temporal_ensemble_coeff": "0.01",
            },
            id="temporal-ensemble",
        ),
    ],
)
def test_action_chunking_metadata_matches_act_config(config, expected):
    assert action_chunking_metadata(config) == expected


def test_act_export_writes_action_chunking_metadata(make_act_module, monkeypatch):
    # A real th.onnx.export imports this as a side effect; the stub below does not
    import onnx_ir.passes.common  # noqa: F401

    model, policy = make_act_module(state_dim=2, action_dim=3)
    policy.config = ACTConfig(chunk_size=50, n_action_steps=20)
    empty_model = ir.Model(ir.Graph([], [], nodes=[]), ir_version=10)
    monkeypatch.setattr(
        th.onnx, "export", lambda *args, **kwargs: SimpleNamespace(model=empty_model)
    )

    onnx_program = model.export_onnx_program()

    assert onnx_program.model.metadata_props == action_chunking_metadata(
        policy.config
    )


POLICY_INPUT_FEATURES = {
    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
    "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 4, 4)),
}


def test_check_observations_cover_features_accepts_matching_yaml():
    observation_space = Dict(
        {
            "state": Box(-1, 1, shape=(2,), dtype=np.float32),
            "images.wrist": Box(0, 255, shape=(3, 4, 4), dtype=np.uint8),
        }
    )
    check_observations_cover_features(observation_space, POLICY_INPUT_FEATURES)


@pytest.mark.parametrize(
    "observation_space",
    [
        pytest.param(
            Dict({"obs": Box(-1, 1, shape=(2,), dtype=np.float32)}), id="unnamed"
        ),
        pytest.param(
            Dict(
                {
                    "stat": Box(-1, 1, shape=(2,), dtype=np.float32),
                    "images.wrist": Box(0, 255, shape=(3, 4, 4), dtype=np.uint8),
                }
            ),
            id="typo",
        ),
    ],
)
def test_check_observations_cover_features_lists_missing_features(observation_space):
    with pytest.raises(ValueError, match=r"missing the policy's input features") as err:
        check_observations_cover_features(observation_space, POLICY_INPUT_FEATURES)
    assert "'state'" in str(err.value)
    assert "obs'" not in str(err.value)
