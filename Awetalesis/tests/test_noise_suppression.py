"""tests/test_noise_suppression.py — Unit tests for NoiseSuppressor."""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from processing.noise_suppression import NoiseSuppressor


class TestNoiseSuppressor:
    def test_output_shape_matches_input(self):
        ns = NoiseSuppressor(sample_rate=16000)
        audio = np.random.randint(-1000, 1000, size=16000, dtype=np.int16)
        result = ns.suppress(audio)
        assert result.shape == audio.shape

    def test_output_dtype_is_int16(self):
        ns = NoiseSuppressor(sample_rate=16000)
        audio = np.random.randint(-1000, 1000, size=8000, dtype=np.int16)
        result = ns.suppress(audio)
        assert result.dtype == np.int16

    def test_empty_audio_returns_empty(self):
        ns = NoiseSuppressor(sample_rate=16000)
        audio = np.array([], dtype=np.int16)
        result = ns.suppress(audio)
        assert len(result) == 0

    def test_silence_remains_near_zero(self):
        ns = NoiseSuppressor(sample_rate=16000)
        silence = np.zeros(16000, dtype=np.int16)
        result = ns.suppress(silence)
        assert np.abs(result).max() < 100  # should remain very quiet

    def test_raises_on_2d_input(self):
        ns = NoiseSuppressor(sample_rate=16000)
        audio_2d = np.zeros((100, 2), dtype=np.int16)
        with pytest.raises(ValueError, match="1-D"):
            ns.suppress(audio_2d)

    def test_short_audio_does_not_crash(self):
        ns = NoiseSuppressor(sample_rate=16000)
        audio = np.array([100, -100, 50], dtype=np.int16)
        result = ns.suppress(audio)
        assert result is not None

    def test_values_in_int16_range(self):
        ns = NoiseSuppressor(sample_rate=16000)
        audio = np.random.randint(-32768, 32767, size=16000, dtype=np.int16)
        result = ns.suppress(audio)
        assert result.min() >= -32768
        assert result.max() <= 32767
