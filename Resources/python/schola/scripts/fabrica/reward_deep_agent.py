# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Run the Fabrica reward Deep Agent via ``deepagents.create_deep_agent``."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from schola.scripts.fabrica.sample_summary import FabricaSampleSummary
from schola.scripts.fabrica.langchain_client import build_chat_model
from schola.scripts.fabrica.paths import (
    include_path_from_header,
    parse_class_name_from_header,
)
from schola.scripts.fabrica.prompts import (
    FABRICA_SNAPSHOT_EXCERPT_TEMPLATE,
    FABRICA_ENV_HEADER_TEMPLATE,
    FABRICA_INSTRUCTIONS_TEMPLATE,
    FABRICA_FEEDBACK_TEMPLATE,
    FABRICA_SYSTEM_PROMPT,
)
from schola.scripts.fabrica.settings import FabricaScriptSettings
from schola.scripts.fabrica import ue_project_tools
from schola.scripts.fabrica.codegen import FabricaCodegenData
logger = logging.getLogger(__name__)





def _make_ue_tools(settings: FabricaScriptSettings) -> List[Any]:
    roots = settings.resolved_code_roots
    ignored_globs = settings.code_ignore_globs

    from langchain_core.tools import tool

    @tool
    def ue_list_dir(rel_path: str) -> str:
        """List names in a directory under --code-roots (path relative to a code root, not including the root folder name; POSIX slashes)."""
        try:
            return ue_project_tools.ue_list_dir(rel_path, roots, ignored_globs)
        except Exception as exc:
            return ue_project_tools.format_sandbox_tool_error(
                operation="list", target=rel_path, exc=exc
            )

    @tool
    def ue_read_file(rel_path: str) -> str:
        """Read a UTF-8 text file under --code-roots (path relative to a code root, not including the root folder name)."""
        try:
            return ue_project_tools.ue_read_file(rel_path, roots, ignored_globs)
        except Exception as exc:
            return ue_project_tools.format_sandbox_tool_error(
                operation="read", target=rel_path, exc=exc
            )

    @tool
    def ue_grep(pattern: str, file_glob: str = "*.h") -> str:
        """Regex-search files under code roots (bounded to 200 hits)."""
        try:
            return ue_project_tools.ue_grep(pattern, roots, ignored_globs, glob=file_glob)
        except Exception as exc:
            return ue_project_tools.format_sandbox_tool_error(
                operation="grep", target=pattern, exc=exc
            )

    return [ue_list_dir, ue_read_file, ue_grep]


def run_reward_deep_agent(
    model: BaseChatModel,
    settings: FabricaScriptSettings,
    *,
    messages: Sequence[BaseMessage],
    recursion_limit: int,
) -> FabricaCodegenData:
    """Invoke the Deep Agent harness and return structured init/reward bodies.

    ``messages`` must be HumanMessage / AIMessage chunks only (no system); the
    system prompt is configured on the agent harness.

    Requires the ``deepagents`` package (e.g. ``pip install 'schola[fabrica]'``).
    """
    try:
        from deepagents import create_deep_agent
    except ImportError as exc:
        raise ImportError(
            "Fabrica's reward agent requires the `deepagents` package. "
            "Install with e.g. `pip install 'schola[fabrica]'` "
            "(deepagents is published for Python 3.11+ on PyPI)."
        ) from exc

    tools: List[Any] = _make_ue_tools(settings)
    agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=FABRICA_SYSTEM_PROMPT,
        response_format=FabricaCodegenData,
    )
    logger.info("Using deepagents.create_deep_agent harness.")

    return agent.invoke(
        {"messages": messages},
        config={"recursion_limit": recursion_limit},
    )["structured_response"]


def _task_description_text(settings: FabricaScriptSettings) -> str:
    return settings.paths_settings.task_description.read_text(
        encoding="utf-8", errors="replace"
    )


def _snapshot_excerpt_text(settings: FabricaScriptSettings) -> str:
    snap_path = settings.paths_settings.snapshot_json_path
    if (
        settings.editor_snapshot_settings.enabled
        and snap_path.exists()
        and snap_path.is_file()
    ):
        txt = Path(snap_path).read_text(encoding="utf-8", errors="replace")
        snap_excerpt = txt[:24_000] + ("\n<<truncated>>" if len(txt) > 24_000 else "")
        return snap_excerpt
    return "(none)"


def _env_header_path_text(settings: FabricaScriptSettings) -> str:
    header = settings.paths_settings.env_header.resolve()
    if header.is_file():
        return include_path_from_header(header)
    return str(header)


def _env_header_tool_path_text(settings: FabricaScriptSettings) -> str:
    """Path for ue_read_file / ue_list_dir (relative to a --code-roots entry, POSIX slashes)."""
    header = settings.paths_settings.env_header.resolve()
    if not header.is_file():
        return str(header)
    for root in settings.resolved_code_roots:
        try:
            return header.relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
    return header.as_posix()


def _env_header_excerpt_text(settings: FabricaScriptSettings) -> str:
    header = settings.paths_settings.env_header
    if Path(header).is_file():
        txt = Path(header).read_text(encoding="utf-8", errors="replace")
        return txt[:24_000] + ("\n<<truncated>>" if len(txt) > 24_000 else "")
    return "(none)"


def _fabrica_regions_text(
    init: str | None,
    reward: str | None,
    *,
    empty_message: str,
) -> str:
    """Format init/reward bodies for an assistant turn."""
    chunks: List[str] = []
    if init:
        chunks.append(f"FabricaGeneratedInit body:\n{init}")
    if reward:
        chunks.append(f"FabricaGeneratedRewardForAgent body:\n{reward}")
    if not chunks:
        return empty_message
    return "\n\n".join(chunks)


def _parent_fabrica_regions_text(
    parent_init_body: str | None,
    parent_reward_body: str | None,
) -> str:
    return _fabrica_regions_text(
        parent_init_body,
        parent_reward_body,
        empty_message=(
            "(No parent reward implementation was captured for this iteration; "
            "marker regions were empty or missing.)"
        ),
    )


def build_reward_agent_messages(
    settings: FabricaScriptSettings,
    feedback: Optional[FabricaSampleSummary],
) -> List[BaseMessage]:
    """Build LangChain messages for ``run_reward_deep_agent``.

    - Iteration 0: a single user message composed from ``FABRICA_SNAPSHOT_EXCERPT_TEMPLATE``, ``FABRICA_ENV_HEADER_TEMPLATE``, and ``FABRICA_INSTRUCTIONS_TEMPLATE``.
    - Later iterations: first user (same template), assistant (iteration parent C++),
      then user (``FABRICA_FEEDBACK_TEMPLATE`` with policy feedback).
    """

    first_user_str = ""
    if settings.editor_snapshot_settings.enabled:
        first_user_str += FABRICA_SNAPSHOT_EXCERPT_TEMPLATE.format(
            snapshot_excerpt=_snapshot_excerpt_text(settings),
        )

    first_user_str += FABRICA_ENV_HEADER_TEMPLATE.format(
        env_header_path=_env_header_path_text(settings),
        env_header_tool_path=_env_header_tool_path_text(settings),
        env_header_excerpt=_env_header_excerpt_text(settings),
    )

    first_user_str += FABRICA_INSTRUCTIONS_TEMPLATE.format(
        env_class_name=parse_class_name_from_header(settings.paths_settings.env_header),
        task_text=_task_description_text(settings),
    )

    if feedback is None or feedback.response is None:
        return [HumanMessage(content=first_user_str)]

    feedback_str = FABRICA_FEEDBACK_TEMPLATE.format(
        policy_feedback_interval=settings.loop_settings.policy_feedback_interval,
        feedback=feedback.to_string(
            episode_freq=settings.loop_settings.policy_feedback_interval
        ),
    )
    return [
        HumanMessage(content=first_user_str),
        AIMessage(
            content=_parent_fabrica_regions_text(
                feedback.response.init_body, feedback.response.reward_body
            )
        ),
        HumanMessage(content=feedback_str),
    ]
