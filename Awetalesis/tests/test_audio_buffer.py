"""tests/test_audio_buffer.py — Unit tests for audio.buffer.AudioBuffer."""

import threading

import numpy as np
import pytest

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from audio.buffer import AudioBuffer


def _make_frame(size: int = 480) -> np.ndarray:
    return np.zeros(size, dtype=np.int16)


class TestAudioBuffer:
    def test_put_and_get(self):
        buf = AudioBuffer()
        frame = _make_frame()
        buf.put(frame)
        result = buf.get()
        assert result is not None
        np.testing.assert_array_equal(result, frame)

    def test_get_empty_returns_none(self):
        buf = AudioBuffer()
        assert buf.get() is None

    def test_len(self):
        buf = AudioBuffer()
        assert len(buf) == 0
        buf.put(_make_frame())
        buf.put(_make_frame())
        assert len(buf) == 2

    def test_is_empty(self):
        buf = AudioBuffer()
        assert buf.is_empty()
        buf.put(_make_frame())
        assert not buf.is_empty()

    def test_clear(self):
        buf = AudioBuffer()
        buf.put(_make_frame())
        buf.put(_make_frame())
        buf.clear()
        assert buf.is_empty()
        assert len(buf) == 0

    def test_get_all(self):
        buf = AudioBuffer()
        frames = [_make_frame(i + 100) for i in range(5)]
        for f in frames:
            buf.put(f)
        result = buf.get_all()
        assert len(result) == 5
        assert buf.is_empty()

    def test_max_frames_overflow(self):
        buf = AudioBuffer(max_frames=3)
        for i in range(5):
            buf.put(_make_frame())
        assert len(buf) == 3  # oldest frames dropped

    def test_thread_safety(self):
        """Multiple writers and a single reader should not cause errors."""
        buf = AudioBuffer(max_frames=1000)
        errors = []

        def writer():
            try:
                for _ in range(50):
                    buf.put(_make_frame())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(buf) <= 1000

    def test_fifo_order(self):
        buf = AudioBuffer()
        f1 = np.array([1, 2, 3], dtype=np.int16)
        f2 = np.array([4, 5, 6], dtype=np.int16)
        buf.put(f1)
        buf.put(f2)
        assert buf.get()[0] == 1
        assert buf.get()[0] == 4
