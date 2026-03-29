"""
audio/stream.py — Microphone capture and audio streaming.

Opens a sounddevice input stream that continuously feeds PCM frames into
an :class:`~audio.buffer.AudioBuffer`.  The stream runs in a background
thread managed by sounddevice so the main event loop is never blocked.
"""

import asyncio
from typing import Optional

import numpy as np

import config
from audio.buffer import AudioBuffer
from utils.logger import get_logger

logger = get_logger(__name__)


class AudioStream:
    """Wraps a *sounddevice* input stream and writes frames into a buffer.

    Args:
        buffer: Shared :class:`~audio.buffer.AudioBuffer` instance.
        sample_rate: PCM sample rate in Hz (default from :mod:`config`).
        channels: Number of input channels (default from :mod:`config`).
        chunk_ms: Frame duration in milliseconds (default from :mod:`config`).
    """

    def __init__(
        self,
        buffer: AudioBuffer,
        sample_rate: int = config.SAMPLE_RATE,
        channels: int = config.CHANNELS,
        chunk_ms: int = config.CHUNK_MS,
    ) -> None:
        self._buffer = buffer
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_ms = chunk_ms
        self._stream: Optional[object] = None  # sd.InputStream, lazily assigned
        self._running: bool = False

        self._blocksize: int = int(sample_rate * chunk_ms / 1000)

    # ------------------------------------------------------------------
    # Internal callback
    # ------------------------------------------------------------------

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time,  # noqa: ANN001
        status,  # sd.CallbackFlags — imported lazily; use Any at class level
    ) -> None:
        """Called by sounddevice for each captured audio block."""
        if status:
            logger.warning("AudioStream callback status: %s", status)

        # Convert to int16 mono
        mono = indata[:, 0].copy()
        pcm = (mono * 32767).astype(np.int16)
        self._buffer.put(pcm)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open and start the microphone input stream."""
        if self._running:
            logger.warning("AudioStream already running.")
            return

        try:
            import sounddevice as sd  # lazy import — requires PortAudio at runtime
        except OSError as exc:
            raise RuntimeError(
                "PortAudio library not found. Install it with: "
                "sudo apt install portaudio19-dev (Linux) or "
                "brew install portaudio (macOS)"
            ) from exc

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="float32",
            blocksize=self._blocksize,
            callback=self._callback,
        )
        self._stream.start()
        self._running = True
        logger.info(
            "AudioStream started (rate=%d Hz, chunk=%d ms, blocksize=%d samples)",
            self._sample_rate,
            self._chunk_ms,
            self._blocksize,
        )

    def stop(self) -> None:
        """Stop and close the microphone input stream."""
        if not self._running:
            return
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._running = False
        logger.info("AudioStream stopped.")

    @property
    def is_running(self) -> bool:
        """``True`` while the stream is active."""
        return self._running
