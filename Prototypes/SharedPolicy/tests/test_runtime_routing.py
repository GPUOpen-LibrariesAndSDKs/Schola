"""Tests for one-call ONNX Runtime execution and per-row head routing."""

from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort

from export_shared_policy import export_shared_policy
from run_shared_policy_demo import (
    FloatArray,
    TWO_NPC_HEADS,
    TWO_NPC_OBSERVATIONS,
    run_shared_policy_batch,
)


class CountingSession:
    """Delegate to a real session while counting model-execution calls."""

    def __init__(self, session: ort.InferenceSession) -> None:
        self._session = session
        self.run_call_count = 0

    def run(
        self,
        output_names: Sequence[str],
        input_feed: dict[str, FloatArray],
    ) -> list[FloatArray]:
        self.run_call_count += 1
        return self._session.run(output_names, input_feed)


def test_two_npcs_use_one_runtime_call_and_preserve_routing_order(
    tmp_path: Path,
) -> None:
    model_path = export_shared_policy(tmp_path / "shared_policy.onnx")
    counting_session = CountingSession(
        ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
    )

    result = run_shared_policy_batch(
        counting_session,
        TWO_NPC_OBSERVATIONS,
        TWO_NPC_HEADS,
    )

    assert counting_session.run_call_count == 1
    assert result.combat_actions.shape == (2, 2)
    assert result.support_actions.shape == (2, 2)
    np.testing.assert_allclose(
        result.combat_actions,
        np.array([[4.5, 5.5], [9.5, -2.5]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        result.support_actions,
        np.array([[6.0, -2.0], [4.5, -1.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        result.routed_actions,
        np.array([[4.5, 5.5], [4.5, -1.0]], dtype=np.float32),
    )
