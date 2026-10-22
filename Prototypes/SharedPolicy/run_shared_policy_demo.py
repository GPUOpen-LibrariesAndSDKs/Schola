"""Run one two-NPC ONNX inference call and route each row to its role."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from export_shared_policy import INPUT_NAME, OUTPUT_NAMES, export_shared_policy
from shared_policy import SharedPolicyReferenceModel

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

    def get_inputs(self) -> Sequence[RuntimeValueInfo]: ...

    def get_outputs(self) -> Sequence[RuntimeValueInfo]: ...

    def run(
        self,
        output_names: Sequence[str],
        input_feed: dict[str, FloatArray],
    ) -> list[FloatArray]: ...


class RuntimeValueInfo(Protocol):
    """Named input or output metadata exposed by an inference session."""

    name: str


@dataclass(frozen=True)
class SharedPolicyBatchResult:
    """Both model outputs plus the per-row actions selected after inference."""

    combat_actions: FloatArray
    support_actions: FloatArray
    routed_actions: FloatArray


def _validate_request(
    observations: FloatArray,
    selected_heads: Sequence[HeadName],
) -> None:
    """Reject malformed tensors and routing requests before model execution."""

    if not isinstance(observations, np.ndarray):
        raise TypeError("observations must be a NumPy array")
    if observations.dtype != np.float32:
        raise TypeError(
            f"observations must use float32; received {observations.dtype}"
        )
    if (
        observations.ndim != 2
        or observations.shape[1] != SharedPolicyReferenceModel.OBSERVATION_SIZE
    ):
        raise ValueError(
            "observations must have shape [batch, 4]; "
            f"received {observations.shape}"
        )
    if observations.shape[0] == 0:
        raise ValueError("observations must contain at least one row")
    if len(selected_heads) != observations.shape[0]:
        raise ValueError("selected_heads must contain one entry per observation")

    invalid_heads = [
        head for head in selected_heads if head not in ("combat", "support")
    ]
    if invalid_heads:
        raise ValueError(f"unknown head names: {invalid_heads}")


def _validate_session_contract(session: RuntimeSession) -> None:
    """Require the exact public tensor names promised by this prototype."""

    input_names = [value.name for value in session.get_inputs()]
    output_names = [value.name for value in session.get_outputs()]
    if input_names != [INPUT_NAME]:
        raise ValueError(
            f"expected ONNX inputs {[INPUT_NAME]}; received {input_names}"
        )
    if output_names != list(OUTPUT_NAMES):
        raise ValueError(
            f"expected ONNX outputs {list(OUTPUT_NAMES)}; received {output_names}"
        )


def run_shared_policy_batch(
    session: RuntimeSession,
    observations: FloatArray,
    selected_heads: Sequence[HeadName],
) -> SharedPolicyBatchResult:
    """Execute one batched model call, then route each row to its chosen head."""

    _validate_request(observations, selected_heads)
    _validate_session_contract(session)
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
