# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

import os
from pathlib import Path
import shutil
import subprocess
import tempfile


repository = Path(__file__).resolve().parents[1]
environment = os.environ.copy()
environment["PATH"] = os.pathsep.join(
    [str(repository / ".venv/bin"), str(repository / ".venv/Scripts"), environment["PATH"]]
)

with tempfile.TemporaryDirectory(prefix="schola-hook-test-") as directory:
    root = Path(directory)

    def git(*arguments, check=True):
        return subprocess.run(
            ["git", *arguments], cwd=root, env=environment,
            capture_output=True, text=True, check=check,
        )

    git("init", "--quiet")
    git("config", "user.name", "Hook Test")
    git("config", "user.email", "hook-test@example.invalid")
    git("config", "commit.gpgsign", "false")
    git("config", "core.hooksPath", str(repository / ".githooks"))
    (root / "Godot").mkdir()
    shutil.copyfile(repository / "Godot/.clang-format", root / "Godot/.clang-format")
    git("add", "Godot/.clang-format")
    git("commit", "--quiet", "-m", "Non-C++ changes pass")

    source = root / "Godot/example with spaces.cpp"
    good = "int answer = 42;\n"
    bad = "int    answer=42;\n"

    # A clean working copy must not conceal malformed staged contents.
    source.write_text(bad)
    git("add", "Godot/example with spaces.cpp")
    source.write_text(good)
    rejected = git("commit", "--quiet", "-m", "Must fail", check=False)
    assert rejected.returncode != 0, rejected
    assert "Formatting failed" in rejected.stderr, rejected.stderr

    # Unstaged edits must not affect a correctly formatted staged snapshot.
    git("add", "Godot/example with spaces.cpp")
    source.write_text(bad)
    git("commit", "--quiet", "-m", "Formatted staged contents pass")
    assert source.read_text() == bad
    assert git("show", "HEAD:Godot/example with spaces.cpp").stdout == good

    source.unlink()
    git("add", "-u")
    git("commit", "--quiet", "-m", "Deleted C++ files pass")

print("Pre-commit hook checks passed.")
