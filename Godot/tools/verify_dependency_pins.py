#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

import json
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOCK_FILE = REPOSITORY_ROOT / "Godot" / "dependencies.lock.json"
SUBMODULE_DEPENDENCIES = ("godot_cpp", "catch2")


def submodule_commit(path):
    result = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT / path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main():
    dependencies = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    mismatches = []

    for name in SUBMODULE_DEPENDENCIES:
        dependency = dependencies[name]
        expected = dependency["commit"]
        actual = submodule_commit(dependency["path"])
        if actual != expected:
            mismatches.append(f"{name}: lock has {expected}, submodule is {actual}")

    if mismatches:
        raise RuntimeError("Dependency pin mismatch:\n" + "\n".join(mismatches))

    print("Dependency lock matches pinned submodules.")


if __name__ == "__main__":
    main()
