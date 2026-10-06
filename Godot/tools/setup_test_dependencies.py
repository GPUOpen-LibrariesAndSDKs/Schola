#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


GODOT_ROOT = Path(__file__).resolve().parents[1]
LOCK_FILE = GODOT_ROOT / "dependencies.lock.json"
INSTALL_DIRECTORY = GODOT_ROOT / "addons" / "gdUnit4"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as archive:
        for block in iter(lambda: archive.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install_gdunit4(archive_path, dependency):
    expected_hash = dependency["sha256"]
    actual_hash = sha256(archive_path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"GdUnit4 archive checksum mismatch: expected {expected_hash}, got {actual_hash}"
        )

    archive_prefix = dependency["archive_path"].rstrip("/") + "/"
    with tempfile.TemporaryDirectory(dir=GODOT_ROOT) as temporary_directory:
        staged_directory = Path(temporary_directory) / "gdUnit4"
        with zipfile.ZipFile(archive_path) as archive:
            members = [
                member
                for member in archive.infolist()
                if member.filename.startswith(archive_prefix)
                and member.filename != archive_prefix
            ]
            if not members:
                raise RuntimeError("GdUnit4 archive does not contain the expected add-on")

            for member in members:
                relative_path = Path(member.filename.removeprefix(archive_prefix))
                if ".." in relative_path.parts:
                    raise RuntimeError("GdUnit4 archive contains an unsafe path")
                destination = staged_directory / relative_path
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)

        if INSTALL_DIRECTORY.exists():
            shutil.rmtree(INSTALL_DIRECTORY)
        shutil.move(staged_directory, INSTALL_DIRECTORY)


def main():
    parser = argparse.ArgumentParser(description="Install pinned Godot test dependencies")
    parser.add_argument("--archive", type=Path, help="Use an existing GdUnit4 archive")
    arguments = parser.parse_args()

    dependency = json.loads(LOCK_FILE.read_text(encoding="utf-8"))["gdunit4"]
    if arguments.archive:
        install_gdunit4(arguments.archive.resolve(), dependency)
        return

    with tempfile.TemporaryDirectory() as temporary_directory:
        archive_path = Path(temporary_directory) / "gdunit4.zip"
        urllib.request.urlretrieve(dependency["url"], archive_path)
        install_gdunit4(archive_path, dependency)


if __name__ == "__main__":
    main()
