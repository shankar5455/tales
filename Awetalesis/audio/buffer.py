"""
audio/buffer.py — Thread-safe circular audio buffer.

Stores raw PCM frames (numpy int16 arrays) in a fixed-size deque so that
the recording thread and the processing thread can operate independently
without blocking each other.
"""

import threading
from collections import deque
from typing import Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class AudioBuffer:
    """Thread-safe circular buffer for raw PCM audio frames.

    Args:
        max_frames: Maximum number of audio chunks to keep in memory.
                    Older frames are silently dropped when the buffer is full.
    """

    def __init__(self, max_frames: int = 200) -> None:
        self._buffer: deque = deque(maxlen=max_frames)
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def put(self, frame: np.ndarray) -> None:
        """Append a PCM frame to the buffer.

        Args:
            frame: 1-D int16 numpy array representing a single audio chunk.
        """
        with self._lock:
            self._buffer.append(frame)

    def get(self) -> Optional[np.ndarray]:
        """Remove and return the oldest PCM frame, or *None* if empty.

        Returns:
            Oldest :class:`numpy.ndarray` frame, or ``None``.
        """
        with self._lock:
            if self._buffer:
                return self._buffer.popleft()
            return None

    def get_all(self) -> list:
        """Drain and return all frames currently in the buffer.

        Returns:
            List of :class:`numpy.ndarray` frames (may be empty).
        """
        with self._lock:
            frames = list(self._buffer)
            self._buffer.clear()
            return frames

    def clear(self) -> None:
        """Discard all frames."""
        with self._lock:
            self._buffer.clear()

    def is_empty(self) -> bool:
        """Return ``True`` if no frames are waiting."""
        with self._lock:
            return len(self._buffer) == 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)
