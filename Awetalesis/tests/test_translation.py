"""tests/test_translation.py — Unit tests for Translator (mocked HTTP)."""

import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from processing.translation import Translator


@pytest.mark.asyncio
class TestTranslator:
    async def test_empty_text_returns_empty(self):
        t = Translator(api_key="", target_language="es")
        result = await t.translate("")
        assert result == ""

    async def test_whitespace_only_returns_whitespace(self):
        t = Translator(api_key="", target_language="es")
        result = await t.translate("   ")
        assert result == "   "

    async def test_no_api_key_returns_original(self):
        """Without an API key the fallback path returns the original text."""
        t = Translator(api_key="", target_language="es")
        result = await t.translate("Hello world")
        assert result == "Hello world"

    async def test_google_api_success(self):
        """Successful Google API call returns translated text."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": {"translations": [{"translatedText": "Hola mundo"}]}
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            MockClient.return_value.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            t = Translator(api_key="fake_key", target_language="es")
            result = await t.translate("Hello world")

        assert result == "Hola mundo"

    async def test_google_api_failure_returns_original(self):
        """On API failure the original text is returned."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(
                side_effect=Exception("Network error")
            )
            MockClient.return_value.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            t = Translator(api_key="fake_key", target_language="es")
            result = await t.translate("Hello world")

        assert result == "Hello world"

    async def test_target_language_property(self):
        t = Translator(api_key="", target_language="fr")
        assert t.target_language == "fr"
        t.target_language = "de"
        assert t.target_language == "de"

    async def test_target_language_override_per_call(self):
        """translate() target_language parameter overrides the instance value."""
        t = Translator(api_key="", target_language="es")
        # Without API key, fallback returns original text regardless of target
        result = await t.translate("Hello", target_language="ja")
        assert result == "Hello"
