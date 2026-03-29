"""
processing/tts.py — Text-to-Speech synthesis.

Primary:  ElevenLabs REST API (high-quality, low-latency).
Fallback: gTTS (Google Text-to-Speech, free tier, returns MP3 bytes).

Both paths return raw audio bytes (MP3) that the pipeline can stream to
connected WebSocket clients.
"""

import asyncio
import io
from typing import Optional

import httpx

import config
from utils.logger import get_logger

logger = get_logger(__name__)

_ELEVENLABS_TTS_URL = (
    "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
)


class TTS:
    """Convert text to speech audio bytes.

    Args:
        api_key: ElevenLabs API key.
        voice_id: ElevenLabs voice identifier.
        target_language: Language code used for gTTS fallback.
    """

    def __init__(
        self,
        api_key: str = config.ELEVENLABS_API_KEY,
        voice_id: str = config.ELEVENLABS_VOICE_ID,
        target_language: str = config.TARGET_LANGUAGE,
    ) -> None:
        self._api_key = api_key
        self._voice_id = voice_id
        self._target_language = target_language

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def target_language(self) -> str:
        """Language code used for the gTTS fallback."""
        return self._target_language

    @target_language.setter
    def target_language(self, value: str) -> None:
        self._target_language = value

    async def synthesize(self, text: str) -> bytes:
        """Convert *text* to audio (MP3 bytes).

        Tries ElevenLabs first; falls back to gTTS.

        Args:
            text: Text to synthesize.

        Returns:
            MP3 audio bytes.  Returns empty bytes only if both providers fail.
        """
        if not text.strip():
            return b""

        if self._api_key:
            try:
                audio = await self._elevenlabs(text)
                if audio:
                    return audio
            except Exception as exc:
                logger.warning("ElevenLabs TTS failed, falling back to gTTS: %s", exc)

        return await self._gtts(text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _elevenlabs(self, text: str) -> Optional[bytes]:
        """Call the ElevenLabs text-to-speech endpoint."""
        url = _ELEVENLABS_TTS_URL.format(voice_id=self._voice_id)
        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            audio = resp.content
            if audio:
                logger.debug("ElevenLabs TTS: %d bytes", len(audio))
                return audio
        return None

    async def _gtts(self, text: str) -> bytes:
        """Generate speech with gTTS (runs in a thread pool)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run_gtts, text)

    def _run_gtts(self, text: str) -> bytes:
        """Synchronous gTTS synthesis."""
        try:
            from gtts import gTTS  # type: ignore

            lang = self._target_language[:2]
            tts = gTTS(text=text, lang=lang, slow=False)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            audio = buf.getvalue()
            logger.debug("gTTS fallback: %d bytes", len(audio))
            return audio
        except Exception as exc:
            logger.error("gTTS synthesis failed: %s", exc)
            return b""
