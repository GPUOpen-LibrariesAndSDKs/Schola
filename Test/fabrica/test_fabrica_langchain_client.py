# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

from unittest.mock import MagicMock, patch

from schola.scripts.fabrica.langchain_client import build_chat_model
from schola.scripts.fabrica.settings import FabricaLLMSettings


def test_build_chat_model_openai_compatible_passes_headers() -> None:
    settings = FabricaLLMSettings(
        api_key="dummy",
        base_url="https://llm-api.amd.com",
        model="swe-gpt35-turbo-exp1",
        headers={
            "Ocp-Apim-Subscription-Key": "secret-sub-key",
            "user": "tester",
        },
        temperature=0.0,
        max_tokens=3000,
    )
    mock_init = MagicMock()
    with patch("langchain.chat_models.init_chat_model", mock_init):
        build_chat_model(settings)

    mock_init.assert_called_once()
    args, kwargs = mock_init.call_args
    assert args[0] == "swe-gpt35-turbo-exp1"
    assert kwargs["default_headers"] == {
        "Ocp-Apim-Subscription-Key": "secret-sub-key",
        "user": "tester",
    }
    assert kwargs["base_url"] == "https://llm-api.amd.com"
    assert kwargs["api_key"] == "dummy"
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_tokens"] == 3000


def test_build_chat_model_openai_passes_headers() -> None:
    settings = FabricaLLMSettings(
        model="gpt-4o-mini",
        headers={"X-Custom": "1"},
    )
    mock_init = MagicMock()
    with patch("langchain.chat_models.init_chat_model", mock_init):
        build_chat_model(settings)

    mock_init.assert_called_once()
    args, kwargs = mock_init.call_args
    assert args[0] == "gpt-4o-mini"
    assert kwargs["default_headers"] == {"X-Custom": "1"}
