"""
pipeline/pipeline.py — Orchestrates the full Speech-to-Speech Translation pipeline.

Stage order:
    Microphone → AudioStream → AudioBuffer
    → NoiseSuppressor → VAD → ASR → LanguageIdentifier → Translator → TTS
    → bytes broadcast to WebSocket clients
"""

import asyncio
from typing import Callable, Optional

import numpy as np

import config
from audio.buffer import AudioBuffer
from audio.stream import AudioStream
from processing.asr import ASR
from processing.lid import LanguageIdentifier
from processing.noise_suppression import NoiseSuppressor
from processing.translation import Translator
from processing.tts import TTS
from processing.vad import VAD
from utils.logger import get_logger

logger = get_logger(__name__)

# Type alias for the callback that receives translated audio bytes
AudioCallback = Callable[[bytes, str, str], None]


class S2STPipeline:
    """End-to-end Speech-to-Speech Translation pipeline.

    Args:
        on_result: Async or sync callable invoked with ``(audio_bytes, original_text,
                   translated_text)`` after each utterance is processed.
    """

    def __init__(self, on_result: Optional[AudioCallback] = None) -> None:
        self._on_result = on_result

        self._buffer = AudioBuffer(max_frames=500)
        self._stream = AudioStream(self._buffer)
        self._noise_suppressor = NoiseSuppressor()
        self._vad = VAD()
        self._asr = ASR()
        self._lid = LanguageIdentifier()
        self._translator = Translator()
        self._tts = TTS()

        self._running: bool = False
        self._process_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start audio capture and the async processing loop."""
        if self._running:
            logger.warning("Pipeline already running.")
            return

        self._running = True
        self._vad.reset()
        self._buffer.clear()
        self._stream.start()
        self._process_task = asyncio.create_task(self._processing_loop())
        logger.info("S2ST Pipeline started.")

    async def stop(self) -> None:
        """Stop the pipeline gracefully."""
        if not self._running:
            return

        self._running = False
        self._stream.stop()

        if self._process_task is not None:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
            self._process_task = None

        logger.info("S2ST Pipeline stopped.")

    def set_target_language(self, language: str) -> None:
        """Update the translation and TTS target language at runtime.

        Args:
            language: BCP-47 language code (e.g. ``"fr"``, ``"de"``).
        """
        self._translator.target_language = language
        self._tts.target_language = language
        logger.info("Target language updated to '%s'.", language)

    @property
    def is_running(self) -> bool:
        """``True`` while the pipeline is active."""
        return self._running

    # ------------------------------------------------------------------
    # Internal processing loop
    # ------------------------------------------------------------------

    async def _processing_loop(self) -> None:
        """Continuously drain the audio buffer and process speech frames."""
        accumulated: list = []
        frame_samples = int(config.SAMPLE_RATE * config.CHUNK_MS / 1000)

        while self._running:
            frame = self._buffer.get()
            if frame is None:
                await asyncio.sleep(0.005)
                continue

            # Accumulate until we have exactly the required frame size
            accumulated.append(frame)
            total_samples = sum(len(f) for f in accumulated)

            if total_samples < frame_samples:
                continue

            # Build a frame of the exact expected length
            combined = np.concatenate(accumulated)
            vad_frame = combined[:frame_samples]
            leftover = combined[frame_samples:]
            accumulated = [leftover] if len(leftover) else []

            # Noise suppression
            clean_frame = self._noise_suppressor.suppress(vad_frame)

            # VAD — process_frame yields complete utterances
            for utterance in self._vad.process_frame(clean_frame):
                asyncio.create_task(self._handle_utterance(utterance))

    async def _handle_utterance(self, audio: np.ndarray) -> None:
        """Run ASR → LID → Translation → TTS for a single utterance."""
        try:
            # ASR
            original_text = await self._asr.transcribe(audio)
            if not original_text.strip():
                logger.debug("Empty transcription, skipping utterance.")
                return

            logger.info("ASR: %s", original_text)

            # Language identification
            detected_lang = await self._lid.detect(original_text)
            logger.info("LID: %s", detected_lang)

            # Translation
            translated_text = await self._translator.translate(
                original_text, source_language=detected_lang
            )
            logger.info(
                "Translation (%s→%s): %s",
                detected_lang,
                self._translator.target_language,
                translated_text,
            )

            # TTS
            audio_bytes = await self._tts.synthesize(translated_text)

            # Deliver result
            if self._on_result is not None:
                if asyncio.iscoroutinefunction(self._on_result):
                    await self._on_result(audio_bytes, original_text, translated_text)
                else:
                    self._on_result(audio_bytes, original_text, translated_text)

        except Exception as exc:
            logger.error("Pipeline error in utterance handling: %s", exc)
