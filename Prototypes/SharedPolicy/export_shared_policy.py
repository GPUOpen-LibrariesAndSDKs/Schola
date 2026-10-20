"""Export and validate the Shared Policy reference model as ONNX."""

from __future__ import annotations

import logging
from pathlib import Path

import onnx
import torch

from shared_policy import SharedPolicyReferenceModel

INPUT_NAME = "observations"
OUTPUT_NAMES = ("combat_actions", "support_actions")
BATCH_DIMENSION_NAME = "batch_size"
OPSET_VERSION = 21
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("shared_policy.onnx")
_REGISTRATION_LOGGER_NAME = "torch.onnx._internal.exporter._registration"


def export_shared_policy(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Export one shared-backbone model with a dynamic batch dimension."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    model = SharedPolicyReferenceModel().eval()
    example_observations = torch.zeros(
        (2, model.OBSERVATION_SIZE),
        dtype=torch.float32,
    )
    batch_dimension = torch.export.Dim(BATCH_DIMENSION_NAME)

    registration_logger = logging.getLogger(_REGISTRATION_LOGGER_NAME)
    previous_log_level = registration_logger.level
    registration_logger.setLevel(logging.ERROR)
    try:
        with torch.no_grad():
            onnx_program = torch.onnx.export(
                model,
                args=(example_observations,),
                input_names=(INPUT_NAME,),
                output_names=OUTPUT_NAMES,
                opset_version=OPSET_VERSION,
                dynamic_shapes=({0: batch_dimension},),
                dynamo=True,
                report=False,
                optimize=True,
                verbose=False,
            )
    finally:
        registration_logger.setLevel(previous_log_level)

    if onnx_program is None:
        raise RuntimeError("PyTorch did not return an ONNX program")

    onnx_program.save(destination, external_data=False)
    onnx.checker.check_model(onnx.load(destination))
    return destination


def _tensor_shape(value_info: onnx.ValueInfoProto) -> tuple[str | int, ...]:
    """Convert ONNX dimension metadata into readable Python values."""

    dimensions: list[str | int] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.dim_param:
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append(dimension.dim_value)
    return tuple(dimensions)


def main() -> None:
    """Generate the model and print its externally visible tensor contract."""

    output_path = export_shared_policy()
    model = onnx.load(output_path)

    print(f"Exported and validated: {output_path}")
    for value_info in (*model.graph.input, *model.graph.output):
        print(f"{value_info.name}: {_tensor_shape(value_info)}")


if __name__ == "__main__":
    main()
