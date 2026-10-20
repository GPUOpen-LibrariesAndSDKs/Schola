"""Failure-focused tests for request and ONNX session contracts."""

from dataclasses import dataclass
from typing import Sequence, cast

import numpy as np
import pytest

from export_shared_policy import INPUT_NAME, OUTPUT_NAMES
from run_shared_policy_demo import (
    FloatArray,
    HeadName,
    TWO_NPC_HEADS,
    TWO_NPC_OBSERVATIONS,
    run_shared_policy_batch,
)


@dataclass(frozen=True)
class FakeValueInfo:
    name: str


class NonExecutingSession:
    """Expose configurable metadata and fail if inference is attempted."""

    def __init__(
        self,
        input_names: Sequence[str] = (INPUT_NAME,),
        output_names: Sequence[str] = OUTPUT_NAMES,
    ) -> None:
        self._inputs = [FakeValueInfo(name) for name in input_names]
        self._outputs = [FakeValueInfo(name) for name in output_names]
        self.run_call_count = 0

    def get_inputs(self) -> list[FakeValueInfo]:
        return self._inputs

    def get_outputs(self) -> list[FakeValueInfo]:
        return self._outputs

    def run(
        self,
        output_names: Sequence[str],
        input_feed: dict[str, FloatArray],
    ) -> list[FloatArray]:
        self.run_call_count += 1
        raise AssertionError("invalid requests must not execute the model")


def test_rejects_incorrect_observation_width_before_inference() -> None:
    session = NonExecutingSession()
    observations = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match=r"shape \[batch, 4\]"):
        run_shared_policy_batch(session, observations, TWO_NPC_HEADS)

    assert session.run_call_count == 0


def test_rejects_non_float32_observations_before_inference() -> None:
    session = NonExecutingSession()
    observations = np.zeros((2, 4), dtype=np.float64)

    with pytest.raises(TypeError, match="must use float32"):
        run_shared_policy_batch(session, observations, TWO_NPC_HEADS)

    assert session.run_call_count == 0


def test_rejects_mismatched_routing_length_before_inference() -> None:
    session = NonExecutingSession()

    with pytest.raises(ValueError, match="one entry per observation"):
        run_shared_policy_batch(session, TWO_NPC_OBSERVATIONS, ("combat",))

    assert session.run_call_count == 0


def test_rejects_unknown_head_before_inference() -> None:
    session = NonExecutingSession()
    selected_heads = cast(tuple[HeadName, ...], ("combat", "stealth"))

    with pytest.raises(ValueError, match="unknown head names"):
        run_shared_policy_batch(session, TWO_NPC_OBSERVATIONS, selected_heads)

    assert session.run_call_count == 0


@pytest.mark.parametrize(
    "output_names",
    [
        ("combat_actions",),
        ("combat_actions", "healing_actions"),
    ],
    ids=["missing-output", "renamed-output"],
)
def test_rejects_missing_or_renamed_outputs_before_inference(
    output_names: tuple[str, ...],
) -> None:
    session = NonExecutingSession(output_names=output_names)

    with pytest.raises(ValueError, match="expected ONNX outputs"):
        run_shared_policy_batch(session, TWO_NPC_OBSERVATIONS, TWO_NPC_HEADS)

    assert session.run_call_count == 0
