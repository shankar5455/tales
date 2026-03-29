"""tests/test_vad.py — Unit tests for the VAD module."""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from processing.vad import VAD


SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)


def _silence_frame() -> np.ndarray:
    return np.zeros(FRAME_SAMPLES, dtype=np.int16)


def _noise_frame(amplitude: int = 3000) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (rng.integers(-amplitude, amplitude, FRAME_SAMPLES)).astype(np.int16)


class TestVAD:
    def test_instantiation(self):
        vad = VAD(sample_rate=SAMPLE_RATE, frame_ms=FRAME_MS)
        assert vad is not None

    def test_invalid_frame_ms_raises(self):
        with pytest.raises(ValueError, match="frame_ms"):
            VAD(sample_rate=SAMPLE_RATE, frame_ms=15)

    def test_is_speech_silence(self):
        vad = VAD(sample_rate=SAMPLE_RATE, frame_ms=FRAME_MS)
        frame = _silence_frame()
        # Pure silence should not be classified as speech
        assert vad.is_speech(frame) is False

    def test_is_speech_wrong_length_returns_false(self):
        vad = VAD(sample_rate=SAMPLE_RATE, frame_ms=FRAME_MS)
        short_frame = np.zeros(10, dtype=np.int16)
        assert vad.is_speech(short_frame) is False

    def test_process_frame_silence_yields_nothing(self):
        vad = VAD(sample_rate=SAMPLE_RATE, frame_ms=FRAME_MS)
        frames = [_silence_frame() for _ in range(50)]
        results = []
        for f in frames:
            results.extend(list(vad.process_frame(f)))
        assert results == []

    def test_reset_clears_state(self):
        vad = VAD(sample_rate=SAMPLE_RATE, frame_ms=FRAME_MS)
        for _ in range(5):
            list(vad.process_frame(_noise_frame()))
        vad.reset()
        # After reset, internal state should be clear
        assert vad._triggered is False
        assert len(vad._voiced_frames) == 0
