# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

from __future__ import annotations

import subprocess
import sys


def test_installed_plugin_is_discovered_by_lerobot():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from lerobot.envs.configs import EnvConfig; "
                "from lerobot.utils.import_utils import register_third_party_plugins; "
                "register_third_party_plugins(); "
                "assert EnvConfig.get_choice_class('schola').__name__ == 'ScholaEnvConfig'"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
