# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Compile Schola ``.proto`` sources to Python and C++ gRPC stubs.

The Cyclopts application ``app`` is exposed as the ``compile-proto`` subcommand
under the main ``schola`` CLI.
"""

from pathlib import Path
import subprocess
import os
import logging
from os.path import isfile, join
import re
from typing import Annotated, Any

from schola.scripts.common.console import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def get_files(folder):
    """
    List regular files in a directory (non-recursive).

    Parameters
    ----------
    folder : str or path-like
        Directory to scan.

    Returns
    -------
    list of str
        Basenames of files directly under ``folder``, sorted for stable ordering.
    """
    return sorted(
        file_name for file_name in os.listdir(folder) if isfile(join(folder, file_name))
    )


def get_proto_files(folder):
    """
    List ``*.proto`` files in a directory.

    Parameters
    ----------
    folder : str or path-like
        Directory containing ``.proto`` sources.

    Returns
    -------
    list of str
        Proto basenames suitable for ``protoc`` arguments.
    """
    return [
        file_name for file_name in get_files(folder) if file_name.endswith(".proto")
    ]


def remove_stale_generated_files(
    folder: Path, expected_files: set[str], suffixes: tuple[str, ...]
):
    """
    Delete generated artifacts in ``folder`` that are not in ``expected_files``.

    Parameters
    ----------
    folder : pathlib.Path
        Directory to scan (non-recursive).
    expected_files : set of str
        Basenames that should be kept.
    suffixes : tuple of str
        File suffixes to consider for deletion (e.g. ``".pb.h"``).
    """
    for file_name in get_files(folder):
        if file_name.endswith(suffixes) and file_name not in expected_files:
            stale_file = Path(folder) / file_name
            logger.info("Removing stale generated file %s", stale_file)
            stale_file.unlink()


def get_expected_generated_files(proto_files: list[str], add_type_stubs: bool):
    """
    Compute expected generated basenames for the current ``.proto`` set.

    Parameters
    ----------
    proto_files : list of str
        Proto basenames (e.g. ``["Foo.proto"]``).
    add_type_stubs : bool
        Whether ``.pyi`` stubs are emitted for Python.

    Returns
    -------
    tuple
        ``(cpp_private, cpp_public, python_files)`` sets of basenames for
        private C++, public headers, and Python outputs respectively.
    """
    proto_stems = [Path(file_name).stem for file_name in proto_files]
    cpp_private = set()
    cpp_public = set()
    python_files = set()

    for proto_stem in proto_stems:
        cpp_private.update(
            {
                f"{proto_stem}.pb.cc",
                f"{proto_stem}.grpc.pb.cc",
                f"{proto_stem}.pb.h",
                f"{proto_stem}.grpc.pb.h",
            }
        )
        cpp_public.update({f"{proto_stem}.pb.h", f"{proto_stem}.grpc.pb.h"})
        python_files.update({f"{proto_stem}_pb2.py", f"{proto_stem}_pb2_grpc.py"})
        if add_type_stubs:
            python_files.add(f"{proto_stem}_pb2.pyi")

    return cpp_private, cpp_public, python_files


def get_generated_cpp_file_types(folder):
    """
    Classify generated C++ protobuf / gRPC outputs in a folder.

    Parameters
    ----------
    folder : str or path-like
        Directory containing ``protoc`` outputs.

    Returns
    -------
    dict
        Keys ``"proto-header"``, ``"grpc-header"``, ``"proto-c"``, ``"grpc-c"`` mapping
        to filename lists.
    """
    files = get_files(folder)
    output = {
        "proto-header": [],
        "grpc-header": [],
        "proto-c": [],
        "grpc-c": [],
    }

    for file_name in files:

        if file_name.endswith(".grpc.pb.cc"):
            output["grpc-c"].append(file_name)
        elif file_name.endswith(".pb.cc"):
            output["proto-c"].append(file_name)
        elif file_name.endswith(".grpc.pb.h"):
            output["grpc-header"].append(file_name)
        elif file_name.endswith(".pb.h"):
            output["proto-header"].append(file_name)
    return output


def get_generated_python_file_types(folder):
    """
    Classify generated Python ``_pb2`` and ``_pb2_grpc`` modules.

    Parameters
    ----------
    folder : str or path-like
        Directory containing generated Python stubs.

    Returns
    -------
    dict
        Keys ``"proto"`` and ``"grpc"`` mapping to filename lists.
    """
    files = get_files(folder)
    output = {
        "proto": [],
        "grpc": [],
    }

    for file_name in files:
        if file_name.endswith("_pb2_grpc.py"):
            output["grpc"].append(file_name)
        elif file_name.endswith("_pb2.py"):
            output["proto"].append(file_name)

    return output


def fix_imports(folder):
    """
    Rewrite plain protobuf imports to ``schola.generated`` package paths.

    Parameters
    ----------
    folder : str or path-like
        Directory of generated ``*_pb2.py`` / ``*_pb2_grpc.py`` files to edit in place.
    """
    files = get_files(folder)
    import_pattern = "^import (.*) as (.*)"
    for file in files:
        with open(join(folder, file), "r+") as f:
            file_contents = f.readlines()
            for i, line in enumerate(file_contents):
                result = re.search(import_pattern, line)
                if result:
                    file_contents[i] = (
                        f"import schola.generated.{result.group(1)} as {result.group(2)}\n"
                    )
            f.seek(0)
            f.writelines(file_contents)
            f.truncate()


def disable_warnings(folder, file_paths, warnings):
    """
    Insert MSVC ``#pragma warning(disable: ...)`` lines into generated C++ sources.

    Parameters
    ----------
    folder : str or path-like
        Base directory containing ``file_paths``.
    file_paths : list of str
        Relative paths to C++ sources to modify.
    warnings : list of str
        MSVC warning numbers to disable (e.g. ``"4125"``).
    """
    for file_path in file_paths:
        with open(join(folder, file_path), "r+") as f:
            file_contents = f.readlines()
            # files have two lines of headers to start
            try:
                headers_start_index = next(
                    (
                        line_num
                        for line_num, line in enumerate(file_contents)
                        if line.startswith("#include ")
                    )
                )
            except StopIteration:
                logger.warning(
                    "Could not find header insertion point in %s; using default index",
                    file_path,
                )
                headers_start_index = 3
            for warning in warnings:
                file_contents.insert(
                    headers_start_index, f"#pragma warning (disable : {warning})\n"
                )
            f.seek(0)
            f.writelines(file_contents)
            f.truncate()


def add_api_macro(folder, file_paths, api_macro):
    """
    Prefix Unreal ``*_API`` macros on generated gRPC ``Service`` virtual methods.

    Parameters
    ----------
    folder : str or path-like
        Base directory containing ``file_paths``.
    file_paths : list of str
        Relative paths to generated ``.grpc.pb.h`` service headers.
    api_macro : str
        Macro token to inject (for example ``SCHOLAPROTOBUF_API``).
    """
    SERVICE_CLASS = "class Service : public ::grpc::Service {"
    END_OF_CLASS = "};"
    for file_path in file_paths:
        with open(join(folder, file_path), "r+") as f:
            file_contents = f.readlines()
            file_contents_cleaned = [
                line.rstrip().lstrip() for line in file_contents
            ]  # remove trailing whitespace
            # TODO make this work for more than one service per file
            if file_contents_cleaned.count(SERVICE_CLASS) == 1:
                logger.info("Found Service in %s. Adding Macros to methods.", file_path)

                # find start and end of service class
                service_class_start = file_contents_cleaned.index(SERVICE_CLASS)
                service_class_end = file_contents_cleaned.index(
                    END_OF_CLASS, service_class_start
                )

                # add the api macro to each virtual method in the service class, and the Constructor
                for i in range(service_class_start + 1, service_class_end):
                    # replace lines that start with whitespace followed by 'virtual'or 'Service() with the same line but with the api_macro added after the whitespace
                    file_contents[i] = re.sub(
                        r"^(\s+)(virtual .*|Service\(\);)",
                        r"\1" + api_macro + r" \2",
                        file_contents[i],
                    )

                f.seek(0)
                f.writelines(file_contents)
                f.truncate()
            elif file_contents_cleaned.count(SERVICE_CLASS) > 1:
                logger.warning(
                    "Multiple service classes found in %s; skipping API macro addition",
                    file_path,
                )


def ensure_ue_verify_macro_sandbox(file_contents: list[str]) -> bool:
    """
    UE defines ``verify`` as a macro; Abseil's btree containers use a member
    named ``verify``. Sandboxing restores correct parsing for protobuf/gRPC includes.

    Inserts ``#include "ScholaProtobufMacroGuardBegin.h"`` immediately after each
    ``THIRD_PARTY_INCLUDES_START`` line and ``#include "ScholaProtobufMacroGuardEnd.h"``
    immediately before each ``THIRD_PARTY_INCLUDES_END`` line when missing. The
    guard headers themselves perform the ``push_macro`` / ``undef`` / ``pop_macro``
    pair, keeping the per-file pollution to a single include line on each side.

    Returns
    -------
    bool
        True if ``file_contents`` was modified.
    """
    start_line = "THIRD_PARTY_INCLUDES_START\n"
    end_line = "THIRD_PARTY_INCLUDES_END\n"
    begin_include = '#include "ScholaProtobufMacroGuardBegin.h"\n'
    end_include = '#include "ScholaProtobufMacroGuardEnd.h"\n'

    modified = False
    i = 0
    while i < len(file_contents):
        line = file_contents[i]
        if line == start_line:
            already_sandboxed = (
                i + 1 < len(file_contents) and file_contents[i + 1] == begin_include
            )
            if not already_sandboxed:
                file_contents.insert(i + 1, begin_include)
                modified = True
                i += 2
                continue
        elif line == end_line:
            already_sandboxed = i > 0 and file_contents[i - 1] == end_include
            if not already_sandboxed:
                file_contents.insert(i, end_include)
                modified = True
                i += 2
                continue
        i += 1
    return modified


def add_third_party_include_guards(
    folder: Path,
    file_paths: list[str],
    include_prefix: str,
    port_def_include: str,
    port_undef_include: str,
):
    """
    Wrap generated protobuf / gRPC ``#include`` blocks for Unreal Engine.

    Inserts ``#include "HAL/Platform.h"`` and ``THIRD_PARTY_INCLUDES_START`` before
    the first third-party include matching ``include_prefix``, and
    ``THIRD_PARTY_INCLUDES_END`` immediately after the line matching
    ``port_undef_include`` (per-line ``strip()``). This keeps UE macros such as
    ``verify`` from breaking
    Abseil / protobuf headers when ``CoreMinimal`` was included earlier in the TU.
    It also preserves the nesting order of Unreal's warning stack relative to
    protobuf / gRPC ``port(s)_def.inc`` and ``port(s)_undef.inc`` pairs.

    Also inserts ``ScholaProtobufMacroGuardBegin.h`` / ``ScholaProtobufMacroGuardEnd.h``
    include pairs around each third-party region to sandbox the ``verify`` macro;
    see ``ensure_ue_verify_macro_sandbox``. Existing files that already have
    ``THIRD_PARTY_INCLUDES_START`` are updated in place when the sandbox is missing.

    Parameters
    ----------
    folder : pathlib.Path
        Base directory containing ``file_paths``.
    file_paths : list of str
        Basenames of generated ``.pb`` / ``.grpc.pb`` C++ sources or headers.
    include_prefix : str
        Prefix of the first third-party ``#include`` line to anchor the start guard
        (e.g. ``'#include \"google/protobuf/'`` or ``'#include <grpcpp/'``).
    port_def_include : str
        Exact ``#include`` line for ``port_def.inc`` / ``ports_def.inc``.
    port_undef_include : str
        Exact ``#include`` line for ``port_undef.inc`` / ``ports_undef.inc``.
    """
    prelude_lines = [
        '#include "HAL/Platform.h"\n',
        "\n",
        "THIRD_PARTY_INCLUDES_START\n",
        '#include "ScholaProtobufMacroGuardBegin.h"\n',
    ]
    start_marker = "THIRD_PARTY_INCLUDES_START\n"
    end_line = "THIRD_PARTY_INCLUDES_END\n"
    end_include_line = '#include "ScholaProtobufMacroGuardEnd.h"\n'

    for file_path in file_paths:
        file_abspath = Path(folder) / file_path
        with open(file_abspath, "r+", encoding="utf-8") as f:
            file_contents = f.readlines()
            modified = False

            if start_marker not in file_contents:
                try:
                    first_third_party_include = next(
                        index
                        for index, line in enumerate(file_contents)
                        if line.startswith(include_prefix)
                    )
                except StopIteration:
                    logger.warning(
                        "Could not find third-party include (%r) in %s",
                        include_prefix,
                        file_path,
                    )
                    continue

                try:
                    port_def_index = next(
                        index
                        for index, line in enumerate(file_contents)
                        if line.strip() == port_def_include
                    )
                except StopIteration:
                    logger.warning(
                        "Could not find %r in %s", port_def_include, file_path
                    )
                    continue

                try:
                    port_undef_index = next(
                        index
                        for index, line in enumerate(file_contents)
                        if line.strip() == port_undef_include
                    )
                except StopIteration:
                    logger.warning(
                        "Could not find %r in %s", port_undef_include, file_path
                    )
                    continue

                for line in reversed(prelude_lines):
                    file_contents.insert(first_third_party_include, line)

                inserted = len(prelude_lines)
                if port_def_index >= first_third_party_include:
                    port_def_index += inserted
                if port_undef_index >= first_third_party_include:
                    port_undef_index += inserted

                if port_undef_index <= port_def_index:
                    logger.warning(
                        "Expected %r after %r in %s; skipping",
                        port_undef_include,
                        port_def_include,
                        file_path,
                    )
                    continue

                file_contents.insert(port_undef_index + 1, end_include_line)
                file_contents.insert(port_undef_index + 2, end_line)
                modified = True

            if ensure_ue_verify_macro_sandbox(file_contents):
                modified = True

            if modified:
                f.seek(0)
                f.writelines(file_contents)
                f.truncate()


def move_files(src_folder: Path, files: list[str], target_folder: Path):
    """
    Move a list of files from ``src_folder`` into ``target_folder``.

    Parameters
    ----------
    src_folder : pathlib.Path
        Source directory.
    files : list of str
        Basenames to move.
    target_folder : pathlib.Path
        Destination directory (created if needed).

    Notes
    -----
    Skips missing sources, identical source/destination paths, and logs failures.
    """
    import shutil

    target_folder = Path(target_folder)
    try:
        target_folder.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error("Could not create target folder %s: %s", target_folder, e)
        return

    for entry in files:
        src = src_folder / entry
        dest = target_folder / entry

        # If entry is not an existing path
        if not src.exists():
            logger.warning("Source file %s does not exist; skipping", src.absolute())
        elif src.resolve() == dest.resolve():
            logger.warning("Source and destination identical for %s; skipping", src)
        else:
            try:
                shutil.move(str(src), str(dest))
            except Exception as e:
                logger.error("Failed to move %s -> %s: %s", src, dest, e)


def make_proto_files(
    protoc_path,
    proto_folder,
    python_folder,
    cpp_folder,
    proto_files: list[str],
    add_type_stubs=False,
    cpp_output_options="",
):
    """
    Run ``protoc`` for each ``.proto`` to emit Python and C++ stubs.

    Parameters
    ----------
    protoc_path : str or path-like
        ``protoc`` executable.
    proto_folder : str or path-like
        ``-I`` include root and location of ``.proto`` files.
    python_folder : str or path-like
        ``--python_out`` (and optional ``--pyi_out``) destination.
    cpp_folder : str or path-like
        ``--cpp_out`` destination.
    proto_files : list of str
        Proto basenames under ``proto_folder`` to compile.
    add_type_stubs : bool, optional
        If True, pass ``--pyi_out`` alongside ``--python_out``.
    cpp_output_options : str, optional
        Options forwarded inside ``--cpp_out`` before the colon separator.
    """
    args = []
    args += [f"-I={proto_folder}"]
    args += [f"--python_out={python_folder}"]
    args += [f"--cpp_out={cpp_output_options}:{cpp_folder}"]

    args += [f"--pyi_out={python_folder}"] if add_type_stubs else []

    for file in proto_files:
        subprocess.run([protoc_path] + args + [join(proto_folder, file)], check=True)


def make_grpc_files(
    protoc_path,
    proto_folder,
    plugin_path,
    target_folder,
    proto_files: list[str],
    output_options="",
):
    """
    Run ``protoc`` with the gRPC C++ plugin for each ``.proto`` service.

    Parameters
    ----------
    protoc_path : str or path-like
        ``protoc`` executable.
    proto_folder : str or path-like
        Include root containing ``.proto`` files.
    plugin_path : str or path-like
        ``protoc-gen-grpc`` plugin binary.
    target_folder : str or path-like
        ``--grpc_out`` destination.
    proto_files : list of str
        Proto basenames under ``proto_folder`` to compile.
    output_options : str, optional
        Options embedded in ``--grpc_out`` before the colon separator.
    """
    I_arg = f"-I={proto_folder}"
    grpc_out = f"--grpc_out={output_options}:{target_folder}"
    plugin = f"--plugin=protoc-gen-grpc={plugin_path}"

    for file in proto_files:
        subprocess.run(
            [protoc_path, I_arg, grpc_out, plugin, join(proto_folder, file)], check=True
        )


from cyclopts import App, Parameter

app = App(name="compile-proto", help="Compile Schola Protobuf files to python/c++!")


def default_warnings(value: Any):
    """
    Default MSVC warning numbers to silence in generated gRPC C++.

    Parameters
    ----------
    value : Any
        Unused; present for Cyclopts ``Parameter(show_default=...)`` integration.

    Returns
    -------
    list of str
        Warning ids ``4125`` and ``4800``.
    """
    return ["4073", "4125", "4800"]


@app.default
def main(
    plugin_folder: Path = Path("."),
    warnings_to_disable: Annotated[
        list[str] | None, Parameter(show_default=default_warnings)
    ] = None,
    add_type_stubs: bool = True,
):
    """
    Generate Python and C++ protobuf / gRPC sources for the Schola plugin.

    Removes stale generated outputs under ``ScholaProtobuf`` and
    ``schola/generated`` that no longer correspond to any ``Proto/*.proto``,
    then runs ``protoc`` and gRPC plugins, applies MSVC warning suppressions and
    Unreal ``SCHOLAPROTOBUF_API`` on generated services, wraps protobuf/gRPC includes
    with ``HAL/Platform.h`` and ``THIRD_PARTY_INCLUDES_*``, moves public headers, and
    rewrites Python imports to ``schola.generated``.

    Parameters
    ----------
    plugin_folder : pathlib.Path
        Plugin root containing ``Proto/`` and ``Resources/tools/protoc`` artifacts.
    warnings_to_disable : list of str, optional
        MSVC warning numbers to silence in generated C++ (defaults from
        ``default_warnings``).
    add_type_stubs : bool, default=True
        If True, pass ``--pyi_out`` so ``.pyi`` stubs are emitted next to ``_pb2.py``.
    """
    if warnings_to_disable is None:
        warnings_to_disable = default_warnings(None)

    module_name = "ScholaProtobuf"

    plugin_folder = plugin_folder
    proto_folder = plugin_folder / "Proto"
    tools_path = plugin_folder / "Resources" / "tools"
    protoc_path = tools_path / "protoc.exe"
    python_plugin_path = tools_path / "grpc_python_plugin.exe"
    cpp_plugin_path = tools_path / "grpc_cpp_plugin.exe"
    cpp_code_folder = plugin_folder / "Source" / module_name / "Private"
    cpp_header_folder = plugin_folder / "Source" / module_name / "Public"
    python_code_folder = plugin_folder / "Resources" / "python" / "schola" / "generated"
    proto_files = get_proto_files(proto_folder)

    short_dep_path = r"Schola\Resources\Build\windows_dependencies.bat"

    api_macro = module_name.upper() + "_API"

    # Check if protoc_path exists
    if not protoc_path.exists():
        raise FileNotFoundError(
            f"Protoc Path {protoc_path} does not exist. Please run {short_dep_path} to generate this. Please note that Linux is not supported."
        )

    # Check if plugin paths exist
    if not python_plugin_path.exists():
        raise FileNotFoundError(
            f"Python Plugin Path {python_plugin_path} does not exist. Please run {short_dep_path} to generate this. Please note that Linux is not supported."
        )

    if not cpp_plugin_path.exists():
        raise FileNotFoundError(
            f"C++ Plugin Path {cpp_plugin_path} does not exist. Please run {short_dep_path} to generate this. Please note that Linux is not supported."
        )

    expected_cpp_private, expected_cpp_public, expected_python = (
        get_expected_generated_files(proto_files, add_type_stubs)
    )
    remove_stale_generated_files(
        cpp_code_folder,
        expected_cpp_private,
        (".pb.cc", ".grpc.pb.cc", ".pb.h", ".grpc.pb.h"),
    )
    remove_stale_generated_files(
        cpp_header_folder,
        expected_cpp_public,
        (".pb.h", ".grpc.pb.h"),
    )
    remove_stale_generated_files(
        python_code_folder,
        expected_python,
        ("_pb2.py", "_pb2.pyi", "_pb2_grpc.py"),
    )

    # generate protobuf files defining serialization for the messages
    logger.info("Generating protobuf files")
    make_proto_files(
        protoc_path,
        proto_folder,
        python_code_folder,
        cpp_code_folder,
        proto_files,
        add_type_stubs,
        cpp_output_options=f"dllexport_decl={api_macro}",
    )

    # generate source for the various message services
    logger.info("Generating gRPC files")
    make_grpc_files(
        protoc_path, proto_folder, python_plugin_path, python_code_folder, proto_files
    )
    make_grpc_files(
        protoc_path, proto_folder, cpp_plugin_path, cpp_code_folder, proto_files
    )

    generated_cpp_files = get_generated_cpp_file_types(cpp_code_folder)
    generated_python_files = get_generated_python_file_types(python_code_folder)

    # need to disable safe to ignore warnings that would otherwise cause Unreal compilation errors
    logger.info("Removing Warnings from generated C++ files")
    disable_warnings(
        cpp_code_folder, generated_cpp_files["proto-c"], warnings_to_disable
    )
    logger.info(
        "Adding UHT API Macro to grpc C++ files %s (protobuf files get theirs from protoc already)",
        generated_cpp_files["grpc-header"],
    )
    add_api_macro(cpp_code_folder, generated_cpp_files["grpc-header"], api_macro)

    logger.info("Adding Unreal THIRD_PARTY_INCLUDES guards to generated protobuf C++")
    add_third_party_include_guards(
        cpp_code_folder,
        generated_cpp_files["proto-header"] + generated_cpp_files["proto-c"],
        '#include "google/protobuf/',
        '#include "google/protobuf/port_def.inc"',
        '#include "google/protobuf/port_undef.inc"',
    )
    add_third_party_include_guards(
        cpp_code_folder,
        generated_cpp_files["grpc-header"] + generated_cpp_files["grpc-c"],
        "#include <grpcpp/",
        "#include <grpcpp/ports_def.inc>",
        "#include <grpcpp/ports_undef.inc>",
    )

    # Note this will leave the generated_cpp_files dict out of date so any other operations need to be done first
    logger.info("Moving c++ headers to Public folder")
    move_files(cpp_code_folder, generated_cpp_files["proto-header"], cpp_header_folder)
    move_files(cpp_code_folder, generated_cpp_files["grpc-header"], cpp_header_folder)

    # generated code doesn't import correctly so we need to prepend Schola.generated._____
    fix_imports(python_code_folder)


if __name__ == "__main__":
    app()
