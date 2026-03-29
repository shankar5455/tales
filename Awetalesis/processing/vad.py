"""
processing/vad.py — Voice Activity Detection using WebRTC VAD.

Segments a continuous PCM stream into speech utterances.  Frames are
classified as *speech* or *silence* and accumulated until a complete
utterance has been collected (silence gap exceeds the configured threshold).
"""

import collections
from typing import Generator, List

import numpy as np
import webrtcvad

import config
from utils.logger import get_logger

logger = get_logger(__name__)

# Supported frame durations (ms) for WebRTC VAD
_VALID_FRAME_MS = {10, 20, 30}


class VAD:
    """Voice Activity Detector wrapping the WebRTC VAD library.

    Args:
        sample_rate: PCM sample rate.  WebRTC VAD supports 8000, 16000,
                     32000, and 48000 Hz.
        aggressiveness: Filter aggressiveness 0 (least) – 3 (most).
        frame_ms: Duration of each input frame in milliseconds (10, 20, or 30).
        silence_threshold_ms: Milliseconds of silence required to consider an
                               utterance complete.
    """

    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE,
        aggressiveness: int = config.VAD_AGGRESSIVENESS,
        frame_ms: int = config.CHUNK_MS,
        silence_threshold_ms: int = config.VAD_SILENCE_THRESHOLD_MS,
    ) -> None:
        if frame_ms not in _VALID_FRAME_MS:
            raise ValueError(
                f"frame_ms must be one of {_VALID_FRAME_MS}, got {frame_ms}"
            )

        self._sample_rate = sample_rate
        self._frame_ms = frame_ms
        self._silence_threshold_ms = silence_threshold_ms
        self._frame_samples = int(sample_rate * frame_ms / 1000)

        self._vad = webrtcvad.Vad(aggressiveness)

        # Ring buffer of (is_speech, frame) pairs for padding
        padding_frames = silence_threshold_ms // frame_ms
        self._ring_buffer: collections.deque = collections.deque(
            maxlen=max(padding_frames, 1)
        )
        self._triggered: bool = False
        self._voiced_frames: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_speech(self, frame: np.ndarray) -> bool:
        """Return ``True`` if *frame* contains speech.

        Args:
            frame: 1-D int16 PCM array of exactly ``frame_samples`` samples.

        Returns:
            Boolean speech classification.
        """
        if len(frame) != self._frame_samples:
            logger.debug(
                "VAD frame length mismatch: expected %d, got %d",
                self._frame_samples,
                len(frame),
            )
            return False
        return self._vad.is_speech(frame.tobytes(), self._sample_rate)

    def process_frame(self, frame: np.ndarray) -> Generator[np.ndarray, None, None]:
        """Feed a single frame and yield complete speech segments.

        Uses a ring-buffer approach so that leading and trailing silence
        is trimmed while short pauses within speech are bridged.

        Args:
            frame: 1-D int16 PCM frame.

        Yields:
            Concatenated int16 arrays representing complete speech utterances.
        """
        speech = self.is_speech(frame)
        self._ring_buffer.append((frame, speech))

        if not self._triggered:
            num_voiced = sum(1 for _, s in self._ring_buffer if s)
            # Start collecting when >90 % of the ring buffer is voiced
            if num_voiced > 0.9 * self._ring_buffer.maxlen:
                self._triggered = True
                for f, _ in self._ring_buffer:
                    self._voiced_frames.append(f)
                self._ring_buffer.clear()
        else:
            self._voiced_frames.append(frame)
            num_unvoiced = sum(1 for _, s in self._ring_buffer if not s)
            # End utterance when ring buffer is mostly silence
            if num_unvoiced > 0.9 * self._ring_buffer.maxlen:
                self._triggered = False
                utterance = np.concatenate(self._voiced_frames)
                self._voiced_frames = []
                self._ring_buffer.clear()
                logger.debug("VAD: utterance collected (%d samples)", len(utterance))
                yield utterance

    def reset(self) -> None:
        """Reset internal state (call between sessions)."""
        self._triggered = False
        self._voiced_frames = []
        self._ring_buffer.clear()
