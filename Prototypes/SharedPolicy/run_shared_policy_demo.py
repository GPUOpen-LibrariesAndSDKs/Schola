"""Run one two-NPC ONNX inference call and route each row to its role."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from export_shared_policy import INPUT_NAME, OUTPUT_NAMES, export_shared_policy

HeadName = Literal["combat", "support"]
FloatArray = npt.NDArray[np.float32]

TWO_NPC_OBSERVATIONS = np.array(
    [
        [1.0, 2.0, 3.0, 4.0],
        [4.0, 1.0, 2.0, 0.0],
    ],
    dtype=np.float32,
)
TWO_NPC_HEADS: tuple[HeadName, ...] = ("combat", "support")


class RuntimeSession(Protocol):
    """Small interface containing the ONNX Runtime operation the demo needs."""

    def run(
        self,
        output_names: Sequence[str],
        input_feed: dict[str, FloatArray],
    ) -> list[FloatArray]: ...


@dataclass(frozen=True)
class SharedPolicyBatchResult:
    """Both model outputs plus the per-row actions selected after inference."""

    combat_actions: FloatArray
    support_actions: FloatArray
    routed_actions: FloatArray


def run_shared_policy_batch(
    session: RuntimeSession,
    observations: FloatArray,
    selected_heads: Sequence[HeadName],
) -> SharedPolicyBatchResult:
    """Execute one batched model call, then route each row to its chosen head."""

    output_values = session.run(
        list(OUTPUT_NAMES),
        {INPUT_NAME: observations},
    )
    combat_actions, support_actions = output_values
    actions_by_head = {
        "combat": combat_actions,
        "support": support_actions,
    }
    routed_actions = np.stack(
        [
            actions_by_head[head][row_index]
            for row_index, head in enumerate(selected_heads)
        ]
    )

    return SharedPolicyBatchResult(
        combat_actions=combat_actions,
        support_actions=support_actions,
        routed_actions=routed_actions,
    )


def main() -> None:
    """Export, execute, route, and print the meeting demonstration."""

    model_path = export_shared_policy()
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    result = run_shared_policy_batch(
        session,
        TWO_NPC_OBSERVATIONS,
        TWO_NPC_HEADS,
    )

    print(f"observations shape: {TWO_NPC_OBSERVATIONS.shape}")
    print(f"combat_actions shape: {result.combat_actions.shape}")
    print(f"support_actions shape: {result.support_actions.shape}")
    print(f"NPC 0 -> combat: {result.routed_actions[0].tolist()}")
    print(f"NPC 1 -> support: {result.routed_actions[1].tolist()}")
    print("ONNX Runtime calls for the batch: 1")
    print("Limit: this does not prove Unreal NNE RunSync integration.")


if __name__ == "__main__":
    main()
