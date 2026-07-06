# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Entry point for the ``schola`` CLI.
"""

import sys
from cyclopts import App, Parameter, validators, group_extractors, Group
from schola.scripts.common.panel import console, error_console, print_error

from rich.traceback import install as install_rich_traceback

# Route uncaught tracebacks through the (stderr) error console, as recommended
# by the Cyclopts "Rich Formatted Exceptions" cookbook.
install_rich_traceback(console=error_console)

app = App(
    console=console,
    error_console=error_console,
    name="schola",
    help="CLI for Schola. Useful for training a model or invoking utilities",
)

# Note that because these scripts don't import their dependencies until they are invoked
# This means that the ImportError will only appear when the command is invoked
try:
    from schola.scripts.sb3 import sb3_app

    app.command(sb3_app, name="sb3")
except ImportError:
    error_console.print_exception()
    print_error(
        "Stable Baselines3 (SB3) is not installed. Install via:\n"
        "pip install 'stable_baselines3'\n"
        "Then re-run your schola command."
    )

try:
    from schola.scripts.rllib import rllib_app

    app.command(rllib_app, name="rllib")
except ImportError:
    error_console.print_exception()
    print_error(
        "Ray RLlib is not installed. Install via:\n"
        "pip install 'ray[rllib]'\n"
        "Then re-run your schola command."
    )

try:
    from schola.scripts.minari import minari_app

    app.command(minari_app, name="minari")
except ImportError:
    error_console.print_exception()
    print_error(
        "Minari is not installed. Install via:\n"
        "pip install 'minari'\n"
        "Then re-run your schola command."
    )


from schola.scripts.utils.compile_proto import app as compile_proto_app

app.command(compile_proto_app, name="compile-proto")

from schola.scripts.utils.build_docs import app as build_docs_app

app.command(build_docs_app, name="build-docs")

# app["compile-proto"].group = "Utilities"
app["--version"].group = "Utilities"
app["--help"].group = "Utilities"


# indirection function for CLI entry point
def main():
    """
    Console script entry point for the top-level ``schola`` CLI.

    Notes
    -----
    On unexpected exceptions, prints a Rich traceback and exits with status code ``1``.
    """
    try:
        app()
    except Exception:
        error_console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
