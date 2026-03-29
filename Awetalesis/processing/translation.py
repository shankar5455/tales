"""
processing/translation.py — Text translation.

Primary:  Google Cloud Translation REST API.
Fallback: return the original text unchanged (system must not crash).
"""

import asyncio
from typing import Optional

import httpx

import config
from utils.logger import get_logger

logger = get_logger(__name__)

_GOOGLE_TRANSLATE_URL = (
    "https://translation.googleapis.com/language/translate/v2"
)


class Translator:
    """Translate text between languages.

    Args:
        api_key: Google Cloud API key.
        target_language: Default BCP-47 target language code.
    """

    def __init__(
        self,
        api_key: str = config.GOOGLE_API_KEY,
        target_language: str = config.TARGET_LANGUAGE,
    ) -> None:
        self._api_key = api_key
        self._target_language = target_language

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def target_language(self) -> str:
        """Currently configured target language code."""
        return self._target_language

    @target_language.setter
    def target_language(self, value: str) -> None:
        self._target_language = value

    async def translate(
        self,
        text: str,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
    ) -> str:
        """Translate *text* to the target language.

        Args:
            text: Source text to translate.
            source_language: BCP-47 source language code.  Auto-detected by
                             Google if omitted.
            target_language: Override the instance-level target language.

        Returns:
            Translated text string.  Returns *text* unchanged on failure.
        """
        if not text.strip():
            return text

        target = target_language or self._target_language

        if self._api_key:
            try:
                result = await self._translate_google(text, source_language, target)
                if result:
                    return result
            except Exception as exc:
                logger.warning(
                    "Google Translate failed, returning original text: %s", exc
                )

        # Fallback: return original
        logger.info("Translation fallback: returning original text.")
        return text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _translate_google(
        self,
        text: str,
        source: Optional[str],
        target: str,
    ) -> Optional[str]:
        """POST to the Google Cloud Translation v2 endpoint."""
        payload: dict = {"q": text, "target": target, "format": "text"}
        if source:
            payload["source"] = source

        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                _GOOGLE_TRANSLATE_URL,
                params={"key": self._api_key},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        translations = (
            data.get("data", {}).get("translations", [])
        )
        if translations:
            translated = translations[0].get("translatedText", "")
            logger.debug("Google Translate: %s → %s", text[:40], translated[:40])
            return translated or None
        return None
