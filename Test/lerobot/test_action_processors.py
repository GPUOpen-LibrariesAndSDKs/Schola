# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

import pytest
import torch as th
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot_env_schola.export import action_processors
from lerobot_env_schola.export.action_processors import (
    ActionProcessor,
    ACTProcessor,
    TemporalEnsembleProcessor,
    make_action_processor,
    register_action_processor,
)


@pytest.fixture
def make_act_config():
    """Build an ACT config with a small chunk.

    ``n_action_steps`` defaults to the whole chunk, or to 1 with a ``temporal_ensemble_coeff``,
    which LeRobot requires.
    """

    def _make(
        *,
        chunk_size: int = 4,
        n_action_steps: int | None = None,
        temporal_ensemble_coeff: float | None = None,
    ) -> ACTConfig:
        if n_action_steps is None:
            n_action_steps = 1 if temporal_ensemble_coeff is not None else chunk_size
        return ACTConfig(
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            temporal_ensemble_coeff=temporal_ensemble_coeff,
        )

    return _make


@pytest.fixture
def make_chunk():
    """Build a [1, chunk_size, action_dim] chunk for ``config`` whose values are all different."""

    def _make(config: ACTConfig, action_dim: int = 3) -> th.Tensor:
        return th.arange(config.chunk_size * action_dim, dtype=th.float32).reshape(
            1, config.chunk_size, action_dim
        )

    return _make


@pytest.fixture
def registry(monkeypatch) -> dict[str, type[ActionProcessor]]:
    """A copy of the processor registry that processors registered in the test are added to."""
    registry = dict(action_processors.ACTION_PROCESSORS)
    monkeypatch.setattr(action_processors, "ACTION_PROCESSORS", registry)
    return registry


def test_default_processor_without_temporal_ensemble_coeff_is_act(make_act_config):
    assert type(make_action_processor(make_act_config())) is ACTProcessor


def test_default_processor_with_temporal_ensemble_coeff_is_temporal_ensemble(
    make_act_config,
):
    config = make_act_config(temporal_ensemble_coeff=0.01)
    assert type(make_action_processor(config)) is TemporalEnsembleProcessor


def test_named_processor_overrides_default(make_act_config):
    config = make_act_config(temporal_ensemble_coeff=0.01)
    assert type(make_action_processor(config, "act")) is ACTProcessor


def test_default_processor_must_be_unique(make_act_config, registry):
    @register_action_processor
    class AlwaysDefaultProcessor(ActionProcessor):
        name = "always_default"

        @classmethod
        def is_default_for(cls, config):
            return True

    with pytest.raises(ValueError, match=r"got \['act', 'always_default'\]"):
        make_action_processor(make_act_config())


def test_unknown_processor_name_lists_registered_names(make_act_config, registry):
    with pytest.raises(ValueError, match="Unknown action processor 'missing'") as err:
        make_action_processor(make_act_config(), "missing")
    for name in registry:
        assert repr(name) in str(err.value)


def test_temporal_ensemble_needs_coeff(make_act_config):
    with pytest.raises(ValueError, match="temporal_ensemble_coeff"):
        make_action_processor(make_act_config(), "temporal_ensemble")


@pytest.mark.parametrize(
    "coeff",
    [pytest.param(float("nan"), id="nan"), pytest.param(-100.0, id="weights-overflow")],
)
def test_temporal_ensemble_rejects_coeff_without_finite_weights(make_act_config, coeff):
    config = make_act_config(chunk_size=2, temporal_ensemble_coeff=coeff)
    with pytest.raises(ValueError, match="does not give finite weights over a chunk of 2"):
        TemporalEnsembleProcessor(config)


def test_temporal_ensemble_accepts_negative_coeff_with_finite_weights(make_act_config):
    TemporalEnsembleProcessor(make_act_config(chunk_size=2, temporal_ensemble_coeff=-1.0))


def test_act_outputs_n_action_steps(make_act_config, make_chunk):
    config = make_act_config(n_action_steps=2)
    processor = ACTProcessor(config)
    chunk = make_chunk(config)
    (actions,) = processor(chunk)
    th.testing.assert_close(actions, chunk[:, :2])


def test_temporal_ensemble_outputs_chunk_and_coeff(make_act_config, make_chunk):
    config = make_act_config(temporal_ensemble_coeff=0.5)
    processor = TemporalEnsembleProcessor(config)
    chunk = make_chunk(config)
    actions, coeff = processor(chunk)
    th.testing.assert_close(actions, chunk)
    th.testing.assert_close(coeff, th.tensor([0.5]))
    assert processor.extra_output_names == ("temporal_ensemble_coeff",)


@pytest.mark.parametrize(
    ("config_kwargs", "expected"),
    [
        pytest.param({}, "action_processor.ACTActionProcessor", id="act"),
        pytest.param(
            {"temporal_ensemble_coeff": 0.01},
            "action_processor.TemporalEnsembleActionProcessor",
            id="ensemble",
        ),
    ],
)
def test_marker_output_names_processor(make_act_config, config_kwargs, expected):
    processor = make_action_processor(make_act_config(**config_kwargs))
    assert processor.marker_output_name == expected
    assert "marker" in dict(processor.named_buffers())


def test_registered_processor_can_be_named(make_act_config, registry):
    @register_action_processor
    class FirstActionProcessor(ActionProcessor):
        name = "first_action"

        def forward(self, chunk):
            return (chunk[:, :1],)

    assert registry["first_action"] is FirstActionProcessor
    assert type(make_action_processor(make_act_config(), "first_action")) is FirstActionProcessor
    with pytest.raises(ValueError, match="already registered"):
        register_action_processor(FirstActionProcessor)
