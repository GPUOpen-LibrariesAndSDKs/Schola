# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Static validation for Fabrica-generated C++ function bodies."""

from __future__ import annotations

import re
from typing import Tuple

from schola.scripts.fabrica.codegen import FabricaCodegenData

_PREPROCESSOR_DIRECTIVE_RE = re.compile(
    r"^\s*#\s*(?:include|pragma|define|ifdef|ifndef|if|elif|else|endif|undef|import|using|error|warning)\b",
    re.MULTILINE | re.IGNORECASE,
)

_DENYLIST_TOKENS: Tuple[str, ...] = (
    "system",
    "FPlatformProcess",
    "IFileManager",
    "FFileHelper",
    "FSocket",
    "popen",
    "LoadLibrary",
    "fopen",
    "freopen",
    "CreateProc",
    "ShellExecute",
    "WinExec",
    "dlopen",
)

_DENYLIST_TOKEN_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(token) for token in _DENYLIST_TOKENS) + r")\b"
)


class FabricaCodegenValidationError(ValueError):
    """Raised when generated C++ bodies fail static safety checks."""


def _find_preprocessor_directive(body: str) -> str | None:
    match = _PREPROCESSOR_DIRECTIVE_RE.search(body)
    if match is None:
        return None
    return match.group(0).strip()


def _find_denylisted_token(body: str) -> str | None:
    match = _DENYLIST_TOKEN_RE.search(body)
    if match is None:
        return None
    return match.group(0)


def validate_generated_body(body: str, *, region: str) -> None:
    """Validate one generated function body before merge into ``*.fabrica.gen.cpp``.

    Empty bodies are allowed (used when clearing candidate regions).
    """
    if not body.strip():
        return

    directive = _find_preprocessor_directive(body)
    if directive is not None:
        raise FabricaCodegenValidationError(
            f"{region} contains a disallowed preprocessor directive: {directive!r}. "
            "Generated Fabrica bodies must not use #include, #pragma, or other preprocessor directives."
        )

    token = _find_denylisted_token(body)
    if token is not None:
        raise FabricaCodegenValidationError(
            f"{region} contains disallowed token {token!r}. "
            "Generated Fabrica bodies must not invoke OS, file, socket, or process APIs."
        )


def validate_fabrica_codegen_data(generated_code: FabricaCodegenData) -> None:
    """Validate init and reward bodies from the reward Deep Agent."""
    validate_generated_body(generated_code.init_body, region="init_body")
    validate_generated_body(generated_code.reward_body, region="reward_body")
