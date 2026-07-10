# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Marker-based merge for Fabrica-generated C++ regions."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from pydantic import BaseModel, Field
from schola.scripts.fabrica.paths import (
    include_path_from_header,
    parse_class_name_from_header,
)

FABRICA_INIT_START = "// <fabrica_generated_init>"
FABRICA_INIT_END = "// </fabrica_generated_init>"
FABRICA_REWARD_START = "// <fabrica_generated_reward>"
FABRICA_REWARD_END = "// </fabrica_generated_reward>"
FABRICA_DECL_START = "// <fabrica_generated_declarations>"
FABRICA_DECL_END = "// </fabrica_generated_declarations>"


_MISSING_ENV_MSG = (
    "Fabrica codegen requires env_header or both class_name and include_path."
)

logger = logging.getLogger(__name__)


class FabricaCodegenData(BaseModel):
    """Structured code generation data: function bodies merged into ``*.fabrica.gen.cpp``."""

    init_body: str = Field(
        description="Raw C++ statements for the FabricaGeneratedInit function body."
    )
    reward_body: str = Field(
        description="Raw C++ statements for the FabricaGeneratedRewardForAgent function body."
    )

@dataclass
class FabricaCodegenResult:
    path: Path
    changed: bool


@dataclass
class FabricaRunArtifactsCleanup:
    header_changed: bool = False
    gen_cpp_changed: bool = False

    @property
    def changed(self) -> bool:
        return self.header_changed or self.gen_cpp_changed


def cleanup_fabrica_run_artifacts(
    env_header_path: Optional[Path] = None,
    gen_path: Optional[Path] = None,
) -> FabricaRunArtifactsCleanup:
    """Remove temporary Fabrica header declarations and/or the generated C++ file.

    Each requested cleanup step runs independently so a failure in one does not
    skip the other. Omit a path to skip cleaning that artifact.
    """
    header_changed = False
    gen_cpp_changed = False
    if env_header_path is not None:
        try:
            header_changed = clean_env_header_declarations(env_header_path).changed
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not clean Fabrica declarations on %s: %s",
                env_header_path.resolve(),
                exc,
            )
    if gen_path is not None:
        try:
            gen_cpp_changed = clean_gen_cpp_file(gen_path).changed
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not clean Fabrica file at %s: %s",
                gen_path.resolve(),
                exc,
            )
    return FabricaRunArtifactsCleanup(
        header_changed=header_changed,
        gen_cpp_changed=gen_cpp_changed,
    )


def derive_generated_cpp_path(
    env_header: Path,
    code_gen_folder: Optional[Path] = None,
) -> Path:
    """Derive ``{Stem}.fabrica.gen.cpp`` from ``env_header`` and optional folder override."""
    header = env_header.resolve()
    gen_name = f"{header.stem}.fabrica.gen.cpp"

    if code_gen_folder is not None:
        return code_gen_folder.resolve() / gen_name

    parts = header.parts
    if "Public" in parts:
        pub_idx = parts.index("Public")
        dir_parts = parts[:pub_idx] + ("Private",) + parts[pub_idx + 1 : -1]
        return Path(*dir_parts) / gen_name

    return header.parent / gen_name


@dataclass
class CodegenEnv:
    env_header_path: Path
    class_name: str
    include_path: str
    code_gen_folder: Path
    generated_cpp_path: Path

    @classmethod
    def from_env_header_and_code_gen_folder(
        cls,
        env_header_path: Path,
        code_gen_folder: Optional[Path] = None,
    ) -> CodegenEnv:

        env_header_path = env_header_path.resolve()

        if code_gen_folder is not None:
            code_gen_folder = code_gen_folder.resolve()
            generated_cpp_path = (
                code_gen_folder / f"{env_header_path.stem}.fabrica.gen.cpp"
            )
        else:
            generated_cpp_path = derive_generated_cpp_path(env_header_path)
            code_gen_folder = generated_cpp_path.parent

        return CodegenEnv(
            env_header_path=env_header_path,
            code_gen_folder=code_gen_folder,
            generated_cpp_path=generated_cpp_path,
            class_name=parse_class_name_from_header(env_header_path),
            include_path=include_path_from_header(env_header_path),
        )


def _build_gen_cpp_scaffold(context: CodegenEnv) -> str:
    """Full compilable ``*.fabrica.gen.cpp`` with markers inside function bodies."""
    return (
        "// Copyright — generated / merged by Schola Fabrica.\n"
        f"// Fabrica hook bodies for {context.class_name}; do not edit markers by hand.\n\n"
        f'#include "{context.include_path}"\n'
        '#include "TrainingDataTypes/AgentState.h"\n'
        '#include "Points/BoxPoint.h"\n\n'
        f"void {context.class_name}::FabricaGeneratedInit()\n"
        "{\n"
        f"{FABRICA_INIT_START}\n"
        f"{FABRICA_INIT_END}\n"
        "}\n\n"
        f"void {context.class_name}::FabricaGeneratedRewardForAgent(\n"
        "\tconst FString& AgentId, FAgentState& OutState)\n"
        "{\n"
        f"{FABRICA_REWARD_START}\n"
        f"{FABRICA_REWARD_END}\n"
        "}\n"
    )


def _valid_gen_cpp_re(class_name: str) -> re.Pattern[str]:
    """Regex for a ``*.fabrica.gen.cpp`` with hook bodies between merge markers."""
    cn = re.escape(class_name)
    init_start = re.escape(FABRICA_INIT_START)
    init_end = re.escape(FABRICA_INIT_END)
    reward_start = re.escape(FABRICA_REWARD_START)
    reward_end = re.escape(FABRICA_REWARD_END)
    return re.compile(
        rf"(?s)^"
        rf".*?"
        rf"void\s+{cn}::FabricaGeneratedInit\s*\(\)\s*\{{\s*"
        rf"{init_start}\s*"
        rf".*?"
        rf"{init_end}\s*"
        rf"\}}\s*"
        rf"void\s+{cn}::FabricaGeneratedRewardForAgent\s*\(\s*"
        rf"const\s+FString&\s+AgentId,\s*FAgentState&\s+OutState\s*\)\s*\{{\s*"
        rf"{reward_start}\s*"
        rf".*?"
        rf"{reward_end}\s*"
        rf"\}}\s*"
        rf"$"
    )


def _is_valid_gen_cpp(content: str, class_name: str) -> bool:
    """True when ``content`` matches the expected wrapped gen-cpp layout."""
    return _valid_gen_cpp_re(class_name).search(content) is not None


def _read_regions_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        i0 = text.index(FABRICA_INIT_START) + len(FABRICA_INIT_START)
        i1 = text.index(FABRICA_INIT_END)
        init = text[i0:i1].strip()
    except ValueError:
        init = None
    try:
        r0 = text.index(FABRICA_REWARD_START) + len(FABRICA_REWARD_START)
        r1 = text.index(FABRICA_REWARD_END)
        reward = text[r0:r1].strip()
    except ValueError:
        reward = None
    return init, reward


def _prepare_gen_cpp_content(
    path: Path,
    *,
    create_if_missing: bool,
    context: CodegenEnv,
) -> Tuple[str, bool]:
    """Load or create gen-cpp content; invalid files are replaced with a fresh scaffold."""
    path = path.resolve()

    if not path.is_file():
        if not create_if_missing:
            raise FileNotFoundError(f"Fabrica gen cpp not found: {path}")
        return _build_gen_cpp_scaffold(context), True

    content = path.read_text(encoding="utf-8")
    if _is_valid_gen_cpp(content, context.class_name):
        return content, False
    return _build_gen_cpp_scaffold(context), True


def _replace_region(
    content: str, start_marker: str, end_marker: str, new_body: str
) -> Tuple[str, bool]:
    i0 = content.find(start_marker)
    i1 = content.find(end_marker)
    if i0 == -1 or i1 == -1 or i1 <= i0:
        raise ValueError(
            f"Invalid or missing Fabrica markers: {start_marker!r} / {end_marker!r}"
        )
    before = content[: i0 + len(start_marker)]
    after = content[i1:]
    inner = new_body.strip("\n")
    if not inner.endswith("\n"):
        inner += "\n"
    new_content = before + "\n" + inner + after
    return new_content, new_content != content


def read_regions(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Return init and reward bodies between Fabrica merge markers in ``path``."""
    text = path.read_text(encoding="utf-8")
    return _read_regions_from_text(text)


def write_gen_cpp(
    context: CodegenEnv,
    generated_code: FabricaCodegenData,
    *,
    create_if_missing: bool = True,
) -> FabricaCodegenResult:
    path = context.generated_cpp_path.resolve()
    content, prep_changed = _prepare_gen_cpp_content(
        path,
        create_if_missing=create_if_missing,
        context=context,
    )

    content, c1 = _replace_region(
        content, FABRICA_INIT_START, FABRICA_INIT_END, generated_code.init_body
    )
    content, c2 = _replace_region(
        content, FABRICA_REWARD_START, FABRICA_REWARD_END, generated_code.reward_body
    )
    changed = prep_changed or c1 or c2
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return FabricaCodegenResult(path=path, changed=changed)


def clean_gen_cpp_regions(
    context: CodegenEnv,
) -> FabricaCodegenResult:
    """Remove a within-iteration candidate from disk (restore parent regions or empty)."""
    return write_gen_cpp(
        context,
        FabricaCodegenData(init_body="", reward_body=""),
        create_if_missing=False,
    )


_FABRICA_INIT_DECL_RE = re.compile(r"\bvirtual\s+void\s+FabricaGeneratedInit\s*\(\)")
_FABRICA_REWARD_DECL_RE = re.compile(
    r"\bvirtual\s+void\s+FabricaGeneratedRewardForAgent\s*\(\s*"
    r"const\s+FString&\s+AgentId,\s*FAgentState&\s+OutState\s*\)"
)


def _build_env_header_declarations_block() -> str:
    return (
        "\n"
        f"\t{FABRICA_DECL_START}\n"
        "\tvirtual void FabricaGeneratedInit() override;\n"
        "\tvirtual void FabricaGeneratedRewardForAgent("
        "const FString& AgentId, FAgentState& OutState) override;\n"
        f"\t{FABRICA_DECL_END}\n"
    )


_GENERATED_BODY_RE = re.compile(r"GENERATED_BODY\s*\(\s*\)")


def _class_body_range(text: str, class_name: str) -> tuple[int, int]:
    """Return ``(body_start, body_end)`` indices for the class body (exclusive end)."""
    class_re = re.compile(
        rf"class\s+(?:\w+_API\s+)?{re.escape(class_name)}\s*:\s*public\s+AFabricaEnvironment\s*\{{",
        re.MULTILINE,
    )
    match = class_re.search(text)
    if not match:
        raise ValueError(
            f"No AFabricaEnvironment subclass {class_name!r} found in header."
        )
    body_start = match.end()
    depth = 1
    i = body_start
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return body_start, i
        i += 1
    raise ValueError(f"Unbalanced braces in class {class_name!r}.")


def _class_body_insertion_index(text: str, class_name: str) -> int:
    """Index after ``GENERATED_BODY()`` (before the blank line preceding markers)."""
    body_start, body_end = _class_body_range(text, class_name)
    class_body = text[body_start:body_end]
    gen_match = _GENERATED_BODY_RE.search(class_body)
    if not gen_match:
        raise ValueError(f"GENERATED_BODY() not found in class {class_name!r}.")

    insert_at = body_start + gen_match.end()
    while insert_at < body_end and text[insert_at] in " \t":
        insert_at += 1
    if insert_at < len(text) and text[insert_at] == "\r":
        insert_at += 1
    if insert_at < len(text) and text[insert_at] == "\n":
        insert_at += 1
    return insert_at


def _header_has_fabrica_hook_declarations(text: str) -> bool:
    return (
        _FABRICA_INIT_DECL_RE.search(text) is not None
        and _FABRICA_REWARD_DECL_RE.search(text) is not None
    )


def inject_env_header_declarations(
    context: CodegenEnv,
) -> FabricaCodegenResult:
    """Add temporary Fabrica hook declarations to the environment header."""
    path = context.env_header_path.resolve()
    text = path.read_text(encoding="utf-8")
    if FABRICA_DECL_START in text:
        return FabricaCodegenResult(path=path, changed=False)

    if _header_has_fabrica_hook_declarations(text):
        return FabricaCodegenResult(path=path, changed=False)

    insert_at = _class_body_insertion_index(text, context.class_name)
    block = _build_env_header_declarations_block()
    new_text = text[:insert_at] + block + text[insert_at:]
    path.write_text(new_text, encoding="utf-8")
    return FabricaCodegenResult(path=path, changed=True)


def clean_env_header_declarations(env_header: Path) -> FabricaCodegenResult:
    """Remove temporary Fabrica hook declarations injected by ``inject_env_header_declarations``."""
    path = env_header.resolve()
    if not path.is_file():
        return FabricaCodegenResult(path=path, changed=False)

    text = path.read_text(encoding="utf-8")
    marker_start = text.find(FABRICA_DECL_START)
    marker_end = text.find(FABRICA_DECL_END)
    if marker_start == -1 or marker_end == -1 or marker_end < marker_start:
        return FabricaCodegenResult(path=path, changed=False)

    start = text.rfind("\n", 0, marker_start) + 1
    if start > 0:
        prev_line_start = text.rfind("\n", 0, start - 1) + 1
        if text[prev_line_start:start].strip() == "":
            start = prev_line_start
    end = marker_end + len(FABRICA_DECL_END)
    end_line = text.find("\n", end)
    if end_line == -1:
        end = len(text)
    else:
        end = end_line + 1
    new_text = text[:start] + text[end:]
    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
        return FabricaCodegenResult(path=path, changed=True)
    return FabricaCodegenResult(path=path, changed=False)


def clean_gen_cpp_file(path: Path) -> FabricaCodegenResult:
    path = path.resolve()
    if not path.is_file():
        return FabricaCodegenResult(path=path, changed=False)
    path.unlink()
    return FabricaCodegenResult(path=path, changed=True)
