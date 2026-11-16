# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch as th
from gymnasium.spaces import Box, Dict
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor.normalize_processor import (
    NormalizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot_env_schola.export import LeRobotACTScholaModule
from lerobot_env_schola.export.action_processors import (
    ACTProcessor,
    TemporalEnsembleProcessor,
)
from lerobot_env_schola.export.export import check_observations_cover_features
from Test.conftest import (
    ExpectedActionChunking,
    check_onnx_metadata,
    check_onnx_model_loads,
    check_onnx_runtime,
)


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
        config: ACTConfig | None = None,
        action_processor: str | None = None,
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
        if config is None:
            chunk_size = chunk.shape[1] if chunk is not None else 4
            config = ACTConfig(chunk_size=chunk_size, n_action_steps=chunk_size)
        policy.config = config
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
                action_processor=action_processor,
            ),
            policy,
        )

    return _make


def test_act_module_returns_full_chunk(make_act_module):
    chunk = th.arange(24, dtype=th.float32).reshape(2, 4, 3)
    model, policy = make_act_module(state_dim=2, action_dim=3, chunk=chunk)
    assert not model.is_stateful
    out, _ = model.forward(th.zeros(2, 2))
    th.testing.assert_close(out, chunk)
    batch = policy.predict_action_chunk.call_args.args[0]
    assert list(batch) == ["observation.state"]


def test_act_module_returns_n_action_steps_of_chunk(make_act_module):
    chunk = th.arange(12, dtype=th.float32).reshape(1, 4, 3)
    model, _ = make_act_module(
        state_dim=2,
        action_dim=3,
        chunk=chunk,
        config=ACTConfig(chunk_size=4, n_action_steps=2),
    )
    assert isinstance(model.action_processor, ACTProcessor)
    assert model.output_names == ["action", "action_processor.ACTActionProcessor"]
    out, _ = model.forward(th.zeros(1, 2))
    th.testing.assert_close(out, chunk[:, :2])


def test_temporal_ensemble_returns_full_chunk_and_coeff(make_act_module):
    chunk = th.arange(12, dtype=th.float32).reshape(1, 4, 3)
    model, _ = make_act_module(
        state_dim=2,
        action_dim=3,
        chunk=chunk,
        config=ACTConfig(chunk_size=4, n_action_steps=1, temporal_ensemble_coeff=0.5),
    )
    assert isinstance(model.action_processor, TemporalEnsembleProcessor)
    assert model.output_names == [
        "action",
        "temporal_ensemble_coeff",
        "action_processor.TemporalEnsembleActionProcessor",
    ]
    out, coeff, _ = model.forward(th.zeros(1, 2))
    th.testing.assert_close(out, chunk)
    th.testing.assert_close(coeff, th.tensor([0.5]))


def test_act_module_rejects_outputs_not_matching_extra_output_names(
    make_act_module, monkeypatch
):
    chunk = th.arange(12, dtype=th.float32).reshape(1, 4, 3)
    model, _ = make_act_module(state_dim=2, action_dim=3, chunk=chunk)
    monkeypatch.setattr(ACTProcessor, "forward", lambda self, chunk: (chunk, chunk))
    with pytest.raises(ValueError, match="ACTProcessor returned 2 outputs, expected"):
        model.forward(th.zeros(1, 2))


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
    action, _ = model.forward(*(values[key] for key in model.input_obs_keys))
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

    action, _ = model.forward(th.tensor([[4.0]]))
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


ACT_EXPORT_OBSERVATION_SPACE = Dict(
    {
        "state": Box(-1, 1, shape=(2,), dtype=np.float32),
        "environment_state": Box(-1, 1, shape=(3,), dtype=np.float32),
    }
)
ACT_EXPORT_ACTION_SPACE = Dict(
    {"action": Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)}
)


def make_tiny_act_policy(**config_kwargs) -> ACTPolicy:
    """A small untrained ACT policy over ``ACT_EXPORT_*_SPACE`` that exports in seconds."""
    config = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
            "observation.environment_state": PolicyFeature(
                type=FeatureType.ENV, shape=(3,)
            ),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(3,))
        },
        chunk_size=4,
        dim_model=16,
        n_heads=2,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=False,
        device="cpu",
        **config_kwargs,
    )
    return ACTPolicy(config)


@pytest.mark.parametrize(
    ("config_kwargs", "action_processor", "expected_chunking"),
    [
        pytest.param(
            {"n_action_steps": 2},
            None,
            ExpectedActionChunking(processor="ACTActionProcessor", chunk_size=2),
            id="act",
        ),
        pytest.param(
            {"n_action_steps": 1, "temporal_ensemble_coeff": 0.01},
            None,
            ExpectedActionChunking(
                processor="TemporalEnsembleActionProcessor",
                chunk_size=4,
                extra_outputs={
                    "temporal_ensemble_coeff": np.array([0.01], dtype=np.float32)
                },
            ),
            id="temporal-ensemble",
        ),
        pytest.param(
            {"n_action_steps": 1, "temporal_ensemble_coeff": 0.01},
            "act",
            ExpectedActionChunking(processor="ACTActionProcessor", chunk_size=1),
            id="act-named-over-ensemble-config",
        ),
    ],
)
def test_act_export_passes_onnx_checks(
    tmp_path, config_kwargs, action_processor, expected_chunking
):
    model = LeRobotACTScholaModule(
        make_tiny_act_policy(**config_kwargs),
        ACT_EXPORT_OBSERVATION_SPACE,
        ACT_EXPORT_ACTION_SPACE,
        action_processor=action_processor,
    )
    model.save_as_onnx(tmp_path / "act.onnx")

    onnx_model = check_onnx_model_loads(tmp_path / "act.onnx")
    check_onnx_metadata(
        onnx_model,
        ACT_EXPORT_OBSERVATION_SPACE,
        ACT_EXPORT_ACTION_SPACE,
        action_chunking=expected_chunking,
        batch_size=1,
    )
    check_onnx_runtime(
        onnx_model,
        ACT_EXPORT_OBSERVATION_SPACE,
        ACT_EXPORT_ACTION_SPACE,
        num_samples=4,
        action_chunking=expected_chunking,
        batch_size=1,
    )
    assert not onnx_model.metadata_props


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
