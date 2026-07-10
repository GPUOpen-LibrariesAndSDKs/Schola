# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Construct LangChain chat models for Fabrica."""

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

from schola.scripts.fabrica.settings import FabricaLLMSettings

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


def build_chat_model(settings: FabricaLLMSettings) -> "BaseChatModel":
    """Return a LangChain ``BaseChatModel`` (lazy import)."""

    model_kwargs: dict[str, Any] = {
        "temperature": settings.temperature,
        "timeout": settings.timeout_s,
    }

    if settings.base_url is not None:
        model_kwargs["base_url"] = settings.base_url

    headers = {k: v for k, v in settings.headers.items() if v}
    if headers:
        model_kwargs["default_headers"] = headers

    if settings.max_tokens is not None:
        model_kwargs["max_tokens"] = settings.max_tokens

    if not settings.verify_ssl:
        import httpx

        # These will be managed by the LangChain client, so we don't need to close them
        model_kwargs["http_client"] = httpx.Client(verify=False)
        model_kwargs["http_async_client"] = httpx.AsyncClient(verify=False)

    if settings.api_key is not None:
        model_kwargs["api_key"] = settings.api_key

    from langchain.chat_models import init_chat_model

    return init_chat_model(settings.model, **model_kwargs)
