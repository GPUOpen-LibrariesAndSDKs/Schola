# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

import numpy as np
import pytest
import torch

from lerobot.envs.utils import preprocess_observation
from lerobot_env_schola.processors import ScholaProcessorStep


@pytest.fixture
def make_processor():
    def _make(
        *policy_keys: str, uint8_image_keys: tuple[str, ...] = ()
    ) -> ScholaProcessorStep:
        return ScholaProcessorStep(
            policy_keys=policy_keys,
            uint8_image_keys=uint8_image_keys,
        )

    return _make


def test_processor_preserves_policy_shaped_state(make_processor):
    processor = make_processor("observation.state")

    processed = processor.observation(
        {
            "observation.state": torch.tensor([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]]),
            "task": ["reach", "reach"],
        }
    )

    assert set(processed) == {"observation.state", "task"}
    assert torch.equal(
        processed["observation.state"],
        torch.tensor([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]]),
    )
    assert processed["task"] == ["reach", "reach"]


def test_processor_normalizes_uint8_image_source(make_processor):
    processor = make_processor(
        "observation.images.front",
        uint8_image_keys=("observation.images.front",),
    )
    image = torch.tensor([[[[0, 255]], [[128, 64]], [[255, 0]]]])

    processed = processor.observation({"observation.images.front": image})

    assert processed["observation.images.front"].dtype == torch.float32
    assert torch.allclose(
        processed["observation.images.front"],
        image.float() / 255,
    )


def test_lerobot_prefixes_policy_shaped_gym_keys_before_processing(make_processor):
    processor = make_processor(
        "observation.state",
        "observation.images.front",
        uint8_image_keys=("observation.images.front",),
    )
    tensors = preprocess_observation(
        {
            "state": np.ones((2, 3), dtype=np.float32),
            "images.front": np.full((2, 3, 4, 5), 255, dtype=np.uint8),
        }
    )

    processed = processor.observation(tensors)

    assert set(processed) == {"observation.state", "observation.images.front"}
    assert torch.all(processed["observation.images.front"] == 1)


def test_processor_rejects_missing_source(make_processor):
    processor = make_processor("observation.state")

    with pytest.raises(KeyError, match="observation.state"):
        processor.observation({})
