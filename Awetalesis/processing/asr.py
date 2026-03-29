"""
processing/asr.py — Automatic Speech Recognition.

Primary:  Google Cloud Speech-to-Text REST API.
Fallback: faster-whisper (tiny model, CPU-only).

The module exposes a single :class:`ASR` class whose ``transcribe`` coroutine
accepts raw PCM audio and returns a text string.
"""

import asyncio
import base64
import io
import wave
from typing import Optional

import httpx
import numpy as np

import config
from utils.logger import get_logger

logger = get_logger(__name__)

_GOOGLE_STT_URL = (
    "https://speech.googleapis.com/v1/speech:recognize"
)


def _pcm_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode a 1-D int16 PCM array as an in-memory WAV file."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


class ASR:
    """Transcribe speech audio to text.

    Args:
        api_key: Google Cloud API key.
        sample_rate: PCM sample rate in Hz.
        language_code: BCP-47 language code for recognition hint.
        whisper_model_size: Size of the Whisper fallback model.
    """

    def __init__(
        self,
        api_key: str = config.GOOGLE_API_KEY,
        sample_rate: int = config.SAMPLE_RATE,
        language_code: str = config.SOURCE_LANGUAGE,
        whisper_model_size: str = config.WHISPER_MODEL_SIZE,
    ) -> None:
        self._api_key = api_key
        self._sample_rate = sample_rate
        self._language_code = language_code
        self._whisper_model_size = whisper_model_size
        self._whisper_model = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def transcribe(self, audio: np.ndarray) -> str:
        """Convert *audio* to text.

        Tries Google Speech-to-Text first; falls back to Whisper on failure.

        Args:
            audio: 1-D int16 PCM array.

        Returns:
            Transcribed text string (empty string if nothing detected).
        """
        if len(audio) == 0:
            return ""

        if self._api_key:
            try:
                text = await self._transcribe_google(audio)
                if text:
                    return text
            except Exception as exc:
                logger.warning("Google STT failed, trying Whisper: %s", exc)

        return await self._transcribe_whisper(audio)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _transcribe_google(self, audio: np.ndarray) -> Optional[str]:
        """Send audio to the Google Cloud Speech REST endpoint."""
        wav_bytes = _pcm_to_wav_bytes(audio, self._sample_rate)
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")

        payload = {
            "config": {
                "encoding": "LINEAR16",
                "sampleRateHertz": self._sample_rate,
                "languageCode": self._language_code,
                "enableAutomaticPunctuation": True,
            },
            "audio": {"content": audio_b64},
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                _GOOGLE_STT_URL,
                params={"key": self._api_key},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        if results:
            transcript = (
                results[0]
                .get("alternatives", [{}])[0]
                .get("transcript", "")
            )
            logger.debug("Google STT: %s", transcript)
            return transcript.strip() or None
        return None

    async def _transcribe_whisper(self, audio: np.ndarray) -> str:
        """Transcribe using the faster-whisper tiny model (CPU only)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run_whisper, audio)

    def _run_whisper(self, audio: np.ndarray) -> str:
        """Synchronous Whisper inference (runs in a thread pool)."""
        try:
            if self._whisper_model is None:
                from faster_whisper import WhisperModel  # type: ignore

                logger.info(
                    "Loading Whisper model '%s' (CPU)…", self._whisper_model_size
                )
                self._whisper_model = WhisperModel(
                    self._whisper_model_size,
                    device="cpu",
                    compute_type="int8",
                )

            float_audio = audio.astype(np.float32) / 32768.0
            segments, _ = self._whisper_model.transcribe(
                float_audio,
                language=self._language_code[:2],
                beam_size=1,
            )
            text = " ".join(seg.text for seg in segments).strip()
            logger.debug("Whisper STT: %s", text)
            return text
        except Exception as exc:
            logger.error("Whisper transcription failed: %s", exc)
            return ""
