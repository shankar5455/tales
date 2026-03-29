"""
processing/lid.py — Language Identification.

Uses the Google Cloud Translation API ``detectLanguage`` endpoint as the
primary detector, with a lightweight *langdetect* fallback when the API is
unavailable or returns no result.
"""

import asyncio
from typing import Optional

import httpx

import config
from utils.logger import get_logger

logger = get_logger(__name__)

_GOOGLE_DETECT_URL = (
    "https://translation.googleapis.com/language/translate/v2/detect"
)


class LanguageIdentifier:
    """Detect the spoken/written language of a text snippet.

    Primary: Google Cloud Translation ``detectLanguage``.
    Fallback: *langdetect* library.

    Args:
        api_key: Google Cloud API key.  Uses :data:`config.GOOGLE_API_KEY`
                 when not provided explicitly.
    """

    def __init__(self, api_key: str = config.GOOGLE_API_KEY) -> None:
        self._api_key = api_key

    async def detect(self, text: str) -> str:
        """Identify the language of *text*.

        Args:
            text: The text whose language should be detected.

        Returns:
            BCP-47 language code string (e.g. ``"en"``, ``"fr"``).
            Falls back to :data:`config.SOURCE_LANGUAGE` on failure.
        """
        if not text.strip():
            return config.SOURCE_LANGUAGE

        # Try Google API first
        if self._api_key:
            try:
                result = await self._detect_google(text)
                if result:
                    return result
            except Exception as exc:
                logger.warning("Google LID failed, using fallback: %s", exc)

        # Fallback: langdetect
        return self._detect_langdetect(text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _detect_google(self, text: str) -> Optional[str]:
        """Call the Google detectLanguage endpoint."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                _GOOGLE_DETECT_URL,
                params={"key": self._api_key},
                json={"q": text},
            )
            resp.raise_for_status()
            data = resp.json()
            detections = (
                data.get("data", {}).get("detections", [[]])[0]
            )
            if detections:
                lang = detections[0].get("language", "")
                confidence = detections[0].get("confidence", 0.0)
                logger.debug("Google LID: %s (confidence=%.2f)", lang, confidence)
                return lang or None
        return None

    @staticmethod
    def _detect_langdetect(text: str) -> str:
        """Lightweight CPU fallback using *langdetect*."""
        try:
            from langdetect import detect  # type: ignore

            lang = detect(text)
            logger.debug("langdetect LID: %s", lang)
            return lang
        except Exception as exc:
            logger.warning("langdetect fallback failed: %s", exc)
            return config.SOURCE_LANGUAGE
