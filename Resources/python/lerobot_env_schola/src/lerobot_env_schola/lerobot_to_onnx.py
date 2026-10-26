# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

from cyclopts import App, types

export_onnx_app = App(
    name="lerobot-to-onnx",
    help="Convert a LeRobot policy to ONNX for Unreal Engine",
)


@export_onnx_app.default
def export(
    policy_pretrained_path: str,
    output_path: types.File,
    spaces_path: types.ExistingFile,
):
    """
    Convert a pretrained ACT policy to ONNX.

    Parameters
    ----------
    policy_pretrained_path : str
        Local pretrained directory or Hugging Face Hub id.
    output_path : path
        Destination ``.onnx`` file.
    spaces_path : path
        YAML file describing ONNX observation and action tensors.
    """
    from lerobot_env_schola.export import convert_pretrained_to_onnx

    convert_pretrained_to_onnx(
        pretrained_path=str(policy_pretrained_path),
        export_path=str(output_path),
        spaces_path=str(spaces_path),
    )


if __name__ == "__main__":
    export_onnx_app()
