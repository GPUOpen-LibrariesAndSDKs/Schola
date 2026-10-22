"""Tests for the exported Shared Policy ONNX graph contract."""

from pathlib import Path

import onnx
from onnx import TensorProto

from export_shared_policy import (
    BATCH_DIMENSION_NAME,
    INPUT_NAME,
    OUTPUT_NAMES,
    _tensor_shape,
    export_shared_policy,
)


def test_exported_model_has_named_dynamic_tensor_contract(
    tmp_path: Path,
) -> None:
    output_path = export_shared_policy(tmp_path / "shared_policy.onnx")
    model = onnx.load(output_path)

    onnx.checker.check_model(model)
    assert [value.name for value in model.graph.input] == [INPUT_NAME]
    assert [value.name for value in model.graph.output] == list(OUTPUT_NAMES)

    observations = model.graph.input[0]
    combat_actions, support_actions = model.graph.output

    assert observations.type.tensor_type.elem_type == TensorProto.FLOAT
    assert combat_actions.type.tensor_type.elem_type == TensorProto.FLOAT
    assert support_actions.type.tensor_type.elem_type == TensorProto.FLOAT
    assert _tensor_shape(observations) == (BATCH_DIMENSION_NAME, 4)
    assert _tensor_shape(combat_actions) == (BATCH_DIMENSION_NAME, 2)
    assert _tensor_shape(support_actions) == (BATCH_DIMENSION_NAME, 2)
