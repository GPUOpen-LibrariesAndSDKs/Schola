#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

import hashlib
import json
import platform
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

GODOT_ROOT = Path(__file__).resolve().parents[1]
LOCK_FILE = GODOT_ROOT / "dependencies.lock.json"
INSTALL_DIRECTORY = GODOT_ROOT / "thirdparty" / "onnxruntime"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as archive:
        for block in iter(lambda: archive.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def host_platform():
    system = {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}.get(
        platform.system()
    )
    machine = {
        "AMD64": "x86_64",
        "x86_64": "x86_64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }.get(platform.machine())
    if system is None or machine is None:
        raise RuntimeError(
            f"Unsupported host: {platform.system()} {platform.machine()}"
        )
    return f"{system}-{machine}"


def extract(archive_path, destination):
    if archive_path.name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            if any(".." in Path(name).parts for name in archive.namelist()):
                raise RuntimeError("ONNX Runtime archive contains an unsafe path")
            archive.extractall(destination)
    else:
        with tarfile.open(archive_path) as archive:
            # The data filter rejects unsafe paths and keeps the library symlinks.
            archive.extractall(destination, filter="data")


def install_onnxruntime(archive_path, archive):
    expected_hash = archive["sha256"]
    actual_hash = sha256(archive_path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"ONNX Runtime archive checksum mismatch: expected {expected_hash}, got {actual_hash}"
        )

    with tempfile.TemporaryDirectory(dir=GODOT_ROOT) as temporary_directory:
        staged_directory = Path(temporary_directory)
        extract(archive_path, staged_directory)
        # Each archive holds one top-level directory, such as onnxruntime-win-x64-1.30.0/.
        roots = [path for path in staged_directory.iterdir() if path.is_dir()]
        if (
            len(roots) != 1
            or not (roots[0] / "include" / "onnxruntime_c_api.h").is_file()
        ):
            raise RuntimeError("ONNX Runtime archive does not have the expected layout")

        if INSTALL_DIRECTORY.exists():
            shutil.rmtree(INSTALL_DIRECTORY)
        INSTALL_DIRECTORY.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(roots[0], INSTALL_DIRECTORY)


def main():
    dependency = json.loads(LOCK_FILE.read_text(encoding="utf-8"))["onnxruntime"]
    host = host_platform()
    archive = dependency["archives"].get(host)
    if archive is None:
        raise RuntimeError(f"No pinned ONNX Runtime archive for {host}")

    with tempfile.TemporaryDirectory() as temporary_directory:
        archive_path = Path(temporary_directory) / Path(archive["url"]).name
        urllib.request.urlretrieve(archive["url"], archive_path)
        install_onnxruntime(archive_path, archive)
    print(f"Installed ONNX Runtime {dependency['version']} to {INSTALL_DIRECTORY}")


if __name__ == "__main__":
    main()
