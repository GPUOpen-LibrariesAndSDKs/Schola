"""Export equivalent A–E benchmark architectures to ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import Tensor, nn

from benchmarks.models import (
    DEFAULT_CONFIG_PATH,
    BackboneModel,
    BenchmarkConfig,
    CompleteRoleModel,
    ConditionalPolicyModel,
    MaskedPolicyModel,
    RoleHeadModel,
    SharedPolicyBenchmarkModel,
    build_seeded_policy,
    load_benchmark_config,
)


OPSET_VERSION = 21
ALL_ARCHITECTURES = ("A", "B", "C", "D", "E")
BATCH_DIMENSION_NAME = "batch_size"
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("artifacts")
_REGISTRATION_LOGGER_NAME = "torch.onnx._internal.exporter._registration"


@dataclass(frozen=True)
class ExportSpec:
    """Everything needed to create and describe one ONNX artifact."""

    name: str
    architecture: str
    model: nn.Module
    example_inputs: tuple[Tensor, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    components: tuple[str, ...]


def _parameter_hash(module: nn.Module) -> str:
    """Hash names, shapes, dtypes, and bytes for every model parameter."""

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _component_hashes(
    source: SharedPolicyBenchmarkModel,
) -> dict[str, str]:
    """Identify the exact seeded backbone and heads reused by every export."""

    hashes = {"backbone": _parameter_hash(source.backbone)}
    hashes.update(
        {
            f"{role}_head": _parameter_hash(source.heads[role])
            for role in source.roles
        }
    )
    return hashes


def _build_export_specs(
    source: SharedPolicyBenchmarkModel,
    architectures: Sequence[str] = ALL_ARCHITECTURES,
) -> tuple[ExportSpec, ...]:
    """Arrange one seeded policy into the selected A–E models."""

    unsupported = sorted(set(architectures) - set(ALL_ARCHITECTURES))
    if unsupported:
        raise ValueError(f"unsupported architectures: {unsupported}")
    dimensions = source.dimensions
    observations = torch.zeros(
        (2, dimensions.observation),
        dtype=torch.float32,
    )
    role_masks = torch.zeros(
        (2, len(source.roles)),
        dtype=torch.float32,
    )
    for row_index in range(role_masks.shape[0]):
        role_masks[row_index, row_index % len(source.roles)] = 1.0
    features = torch.zeros(
        (2, dimensions.backbone),
        dtype=torch.float32,
    )
    all_components = ("backbone",) + tuple(
        f"{role}_head" for role in source.roles
    )

    specs: list[ExportSpec] = []
    if "A" in architectures:
        specs.append(
            ExportSpec(
                name="A_all_heads",
                architecture="A",
                model=source,
                example_inputs=(observations,),
                input_names=("observations",),
                output_names=tuple(f"{role}_actions" for role in source.roles),
                components=all_components,
            )
        )
    if "B" in architectures:
        specs.append(
            ExportSpec(
                name="B_masked",
                architecture="B",
                model=MaskedPolicyModel(source).eval(),
                example_inputs=(observations, role_masks),
                input_names=("observations", "role_masks"),
                output_names=("selected_actions",),
                components=all_components,
            )
        )
    if "D" in architectures:
        specs.append(
            ExportSpec(
                name="D_conditional",
                architecture="D",
                model=ConditionalPolicyModel(source).eval(),
                example_inputs=(
                    observations,
                    torch.tensor(True, dtype=torch.bool),
                ),
                input_names=("observations", "use_first_expert"),
                output_names=("actions",),
                components=all_components,
            )
        )

    if "C" in architectures:
        for role in source.roles:
            specs.append(
                ExportSpec(
                    name=f"C_{role}",
                    architecture="C",
                    model=CompleteRoleModel(source, role).eval(),
                    example_inputs=(observations,),
                    input_names=("observations",),
                    output_names=("actions",),
                    components=("backbone", f"{role}_head"),
                )
            )

    if "E" in architectures:
        specs.append(
            ExportSpec(
                name="E_backbone",
                architecture="E",
                model=BackboneModel(source).eval(),
                example_inputs=(observations,),
                input_names=("observations",),
                output_names=("features",),
                components=("backbone",),
            )
        )
        for role in source.roles:
            specs.append(
                ExportSpec(
                    name=f"E_{role}_head",
                    architecture="E",
                    model=RoleHeadModel(source, role).eval(),
                    example_inputs=(features,),
                    input_names=("features",),
                    output_names=("actions",),
                    components=(f"{role}_head",),
                )
            )
    return tuple(specs)


def _tensor_shape(value_info: onnx.ValueInfoProto) -> list[str | int]:
    """Convert ONNX shape metadata into JSON-friendly values."""

    shape: list[str | int] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        shape.append(
            dimension.dim_param
            if dimension.dim_param
            else dimension.dim_value
        )
    return shape


def _tensor_contract(value_info: onnx.ValueInfoProto) -> dict[str, object]:
    """Describe one public ONNX input or output."""

    element_type = value_info.type.tensor_type.elem_type
    return {
        "name": value_info.name,
        "dtype": np.dtype(
            onnx.helper.tensor_dtype_to_np_dtype(element_type)
        ).name,
        "shape": _tensor_shape(value_info),
    }


def _export_one(
    spec: ExportSpec,
    destination: Path,
    component_hashes: dict[str, str],
) -> dict[str, object]:
    """Export, check, load, and smoke-run one model before describing it."""

    batch_dimension = torch.export.Dim(BATCH_DIMENSION_NAME)
    dynamic_shapes = tuple(
        {0: batch_dimension} if value.ndim > 0 else None
        for value in spec.example_inputs
    )

    registration_logger = logging.getLogger(_REGISTRATION_LOGGER_NAME)
    previous_log_level = registration_logger.level
    registration_logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings(), torch.no_grad():
            # PyTorch warns when two inputs intentionally share one symbolic
            # batch dimension. That is B's required contract, not a failure.
            warnings.filterwarnings(
                "ignore",
                message=r"# The axis name: batch_size will not be used.*",
                category=UserWarning,
            )
            onnx_program = torch.onnx.export(
                spec.model,
                args=spec.example_inputs,
                input_names=spec.input_names,
                output_names=spec.output_names,
                opset_version=OPSET_VERSION,
                dynamic_shapes=dynamic_shapes,
                dynamo=True,
                report=False,
                optimize=True,
                verbose=False,
            )
    finally:
        registration_logger.setLevel(previous_log_level)

    if onnx_program is None:
        raise RuntimeError(f"PyTorch did not return an ONNX program for {spec.name}")

    onnx_program.save(destination, external_data=False)
    model = onnx.load(destination)
    onnx.checker.check_model(model)

    session = ort.InferenceSession(
        str(destination),
        providers=["CPUExecutionProvider"],
    )
    session.run(
        list(spec.output_names),
        {
            name: value.detach().cpu().numpy()
            for name, value in zip(
                spec.input_names,
                spec.example_inputs,
                strict=True,
            )
        },
    )

    return {
        "architecture": spec.architecture,
        "file": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "parameter_hashes": {
            name: component_hashes[name] for name in spec.components
        },
        "ir_version": model.ir_version,
        "opsets": {
            imported.domain or "ai.onnx": imported.version
            for imported in model.opset_import
        },
        "inputs": [_tensor_contract(value) for value in model.graph.input],
        "outputs": [_tensor_contract(value) for value in model.graph.output],
    }


def export_profile(
    config: BenchmarkConfig,
    profile_name: str,
    output_dir: str | Path,
    architectures: Sequence[str] = ALL_ARCHITECTURES,
) -> dict[str, object]:
    """Export the selected architectures for one configured profile."""

    source = build_seeded_policy(config, profile_name)
    profile_dir = Path(output_dir) / profile_name
    profile_dir.mkdir(parents=True, exist_ok=True)
    component_hashes = _component_hashes(source)

    artifacts: dict[str, object] = {}
    for spec in _build_export_specs(source, architectures):
        destination = profile_dir / f"{spec.name}.onnx"
        artifacts[spec.name] = _export_one(
            spec,
            destination,
            component_hashes,
        )

    dimensions = source.dimensions
    return {
        "dimensions": {
            "observation": dimensions.observation,
            "backbone": dimensions.backbone,
            "head": dimensions.head,
            "action": dimensions.action,
            "roles": list(dimensions.roles),
        },
        "source_parameter_hashes": component_hashes,
        "artifacts": artifacts,
    }


def export_models(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    profiles: Sequence[str] | None = None,
) -> Path:
    """Export selected profiles and write one manifest describing the result."""

    config = load_benchmark_config(config_path)
    selected_profiles = tuple(
        config.profiles.keys() if profiles is None else profiles
    )
    if not selected_profiles:
        raise ValueError("at least one profile must be selected")
    if len(set(selected_profiles)) != len(selected_profiles):
        raise ValueError("selected profiles must be unique")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 1,
        "seed": config.seed,
        "opset": OPSET_VERSION,
        "profiles": {
            profile_name: export_profile(config, profile_name, destination)
            for profile_name in selected_profiles
        },
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export shared-policy architecture benchmarks to ONNX.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--profile",
        action="append",
        dest="profiles",
        help="Profile to export; repeat the option to select multiple profiles.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the exporter from the command line."""

    args = _parse_args()
    manifest_path = export_models(
        config_path=args.config,
        output_dir=args.output_dir,
        profiles=args.profiles,
    )
    print(f"Exported and validated models: {manifest_path}")


if __name__ == "__main__":
    main()
