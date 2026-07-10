# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Human-readable markdown dumps for Fabrica reward-agent debugging."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from schola.scripts.fabrica.prompts import FABRICA_SYSTEM_PROMPT
from schola.scripts.fabrica.episode_metrics_callback import FabricaEpisodeMetrics
from schola.scripts.fabrica.codegen import FabricaCodegenData
from schola.scripts.fabrica.sample_summary import FabricaSampleSummary

AGENT_DEBUG_MD_FILENAME = "agent_debug.md"


def _message_role_label(message: BaseMessage) -> str:
    if isinstance(message, HumanMessage):
        return "User"
    if isinstance(message, AIMessage):
        return "Assistant"
    if isinstance(message, SystemMessage):
        return "System"
    return type(message).__name__


def _message_content_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if text is not None:
                    parts.append(str(text))
                else:
                    parts.append(str(block))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _fence_code(text: str, lang: str = "") -> str:
    """Wrap text in a markdown fenced block (extra backticks if content contains fences)."""
    fence = "```"
    marker = f"{fence}{lang}"
    while marker in text or text.rstrip().endswith(fence):
        fence += "`"
        marker = f"{fence}{lang}"
    return f"{marker}\n{text}\n{fence}\n"


def _looks_like_code(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped:
        return False
    code_markers = (
        "FabricaGenerated",
        "#include",
        "const ",
        "float ",
        "void ",
        "return ",
        "//",
        "/*",
        "{",
    )
    return any(stripped.startswith(marker) for marker in code_markers)


def _indent_function_body(body: str, indent: str = "\t") -> str:
    lines = body.strip("\n").splitlines()
    if not lines:
        return ""
    return "\n".join(f"{indent}{line}" if line.strip() else "" for line in lines)


def _wrap_init_function(class_name: str, body: str) -> str:
    inner = _indent_function_body(body)
    if not inner:
        return f"void {class_name}::FabricaGeneratedInit()\n{{\n}}"
    return f"void {class_name}::FabricaGeneratedInit()\n{{\n{inner}\n}}"


def _wrap_reward_function(class_name: str, body: str) -> str:
    inner = _indent_function_body(body)
    signature = (
        f"void {class_name}::FabricaGeneratedRewardForAgent(\n"
        "\tconst FString& AgentId, FAgentState& OutState)"
    )
    if not inner:
        return f"{signature}\n{{\n}}"
    return f"{signature}\n{{\n{inner}\n}}"


def _meta_table(rows: Sequence[tuple[str, str]]) -> str:
    lines = ["| Field | Value |", "| --- | --- |"]
    for key, value in rows:
        escaped = value.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {key} | {escaped} |")
    return "\n".join(lines)


def _format_messages_section(messages: Sequence[BaseMessage]) -> str:
    lines = ["## Messages", ""]
    for i, message in enumerate(messages, start=1):
        role = _message_role_label(message)
        body = _message_content_text(message)
        char_count = len(body)
        lines.append(f"### Turn {i} — {role}")
        lines.append("")
        lines.append(f"*({char_count:,} characters)*")
        lines.append("")
        if _looks_like_code(body):
            lines.append(_fence_code(body, "cpp").rstrip())
        else:
            lines.append(body)
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_system_prompt_section(system_prompt: str) -> str:
    body = system_prompt.strip()
    lines = [
        "## System Prompt",
        "",
        f"*({len(body):,} characters)*",
        "",
        body,
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def _format_metrics_section(
    metrics: FabricaEpisodeMetrics | None, policy_feedback_interval: int
) -> str:
    lines = ["## Metrics", ""]
    if metrics is None:
        lines.extend(["*(SB3 training was not run or failed.)*", ""])
        return "\n".join(lines)

    aggregate = metrics.mean().to_dict()
    rows: list[tuple[str, str]] = [
        ("Episodes", str(aggregate.get("num_episodes", 0))),
    ]
    for key in ("task_success", "episode_return", "episode_length"):
        value = aggregate.get(key)
        rows.append(
            (key.replace("_", " ").title(), "—" if value is None else str(value))
        )
    reward_components = aggregate.get("reward_components") or {}
    if reward_components:
        for comp_key, comp_value in sorted(reward_components.items()):
            rows.append((comp_key, str(comp_value)))
    lines.append(_meta_table(rows))
    lines.append("")

    if metrics.episodes:
        lines.extend(["### Per-episode", ""])
        episode_rows: list[tuple[str, str]] = []
        for ep in metrics.episodes[::policy_feedback_interval]:
            episode_rows.append((f"Env {ep.env_index}", str(ep.to_dict())))
        lines.append(_meta_table(episode_rows))
        lines.append("")

    lines.extend(
        [
            "### Policy feedback (for next iteration)",
            "",
            metrics.to_string(policy_feedback_interval),
            "",
        ]
    )
    return "\n".join(lines)


def _format_result_section(result: FabricaCodegenData, *, env_class_name: str) -> str:
    init_body = result.init_body or ""
    reward_body = result.reward_body or ""
    lines = [
        "## Result",
        "",
        "### FabricaGeneratedInit",
        "",
        _fence_code(_wrap_init_function(env_class_name, init_body), "cpp").rstrip(),
        "",
        "---",
        "",
        "### FabricaGeneratedRewardForAgent",
        "",
        _fence_code(_wrap_reward_function(env_class_name, reward_body), "cpp").rstrip(),
    ]
    return "\n".join(lines)


def format_agent_debug_markdown(
    summary: FabricaSampleSummary,
    *,
    iteration: int,
    sample_index: int,
    system_prompt: str = FABRICA_SYSTEM_PROMPT,
    env_class_name: str,
    policy_feedback_interval: int = 1,
) -> str:
    """Render a full :class:`FabricaSampleSummary` as a markdown debug document."""
    messages = list(summary.messages or [])
    result = summary.response
    if result is None:
        raise ValueError(
            "FabricaSampleSummary.response is required for agent debug markdown"
        )

    init_body = result.init_body or ""
    reward_body = result.reward_body or ""
    system_text = system_prompt.strip()
    meta_rows: list[tuple[str, str]] = [
        ("Iteration", str(iteration)),
        ("Sample", str(sample_index)),
        ("Environment class", env_class_name),
        ("System prompt chars", f"{len(system_text):,}"),
        ("Turns", str(len(messages))),
        ("Init body chars", f"{len(init_body):,}"),
        ("Reward body chars", f"{len(reward_body):,}"),
    ]
    aggregate = (
        summary.metrics.mean().to_dict() if summary.metrics is not None else None
    )
    if aggregate is not None:
        meta_rows.append(("Episodes", str(aggregate.get("num_episodes", 0))))
        task_success = aggregate.get("task_success")
        if task_success is not None:
            meta_rows.append(("Mean task success", str(task_success)))

    lines = [
        "# Fabrica Reward Agent",
        "",
        _meta_table(meta_rows),
        "",
        _format_system_prompt_section(system_prompt),
        _format_messages_section(messages),
        "",
        "---",
        "",
        _format_result_section(result, env_class_name=env_class_name),
        "",
        "---",
        "",
        _format_metrics_section(summary.metrics, policy_feedback_interval),
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_agent_debug_markdown(
    sample_dir: Any,
    summary: FabricaSampleSummary,
    *,
    iteration: int,
    sample_index: int,
    system_prompt: str = FABRICA_SYSTEM_PROMPT,
    env_class_name: str,
    policy_feedback_interval: int = 1,
) -> None:
    """Write ``agent_debug.md`` for a full sample summary under ``sample_dir``."""
    out = Path(sample_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / AGENT_DEBUG_MD_FILENAME).write_text(
        format_agent_debug_markdown(
            summary,
            iteration=iteration,
            sample_index=sample_index,
            system_prompt=system_prompt,
            env_class_name=env_class_name,
            policy_feedback_interval=policy_feedback_interval,
        ),
        encoding="utf-8",
    )
