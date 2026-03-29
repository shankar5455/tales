"""
processing/noise_suppression.py — Lightweight spectral noise reduction.

Uses the *noisereduce* library (CPU-only, no torch dependency) to suppress
background noise from raw PCM audio before passing it to the VAD/ASR stages.
"""

import numpy as np
import noisereduce as nr

import config
from utils.logger import get_logger

logger = get_logger(__name__)


class NoiseSuppressor:
    """Applies spectral subtraction noise reduction to a PCM waveform.

    Args:
        sample_rate: Sample rate of the audio in Hz.
        prop_decrease: Proportion by which to reduce noise (0.0–1.0).
    """

    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE,
        prop_decrease: float = config.NOISE_REDUCE_PROP_DECREASE,
    ) -> None:
        self._sample_rate = sample_rate
        self._prop_decrease = prop_decrease

    def suppress(self, audio: np.ndarray) -> np.ndarray:
        """Reduce noise in *audio* and return the cleaned signal.

        The input is expected to be a 1-D int16 array.  The method works on
        a float32 copy internally and returns an int16 array of the same
        length.

        Args:
            audio: Raw PCM audio as a 1-D ``np.int16`` array.

        Returns:
            Noise-reduced PCM audio as a 1-D ``np.int16`` array.

        Raises:
            ValueError: If *audio* is not a 1-D array.
        """
        if audio.ndim != 1:
            raise ValueError(
                f"Expected 1-D audio array, got shape {audio.shape}"
            )

        if len(audio) == 0:
            return audio.copy()

        try:
            float_audio = audio.astype(np.float32) / 32768.0
            reduced = nr.reduce_noise(
                y=float_audio,
                sr=self._sample_rate,
                prop_decrease=self._prop_decrease,
                stationary=False,
            )
            return (reduced * 32767).astype(np.int16)
        except Exception as exc:  # pragma: no cover
            logger.warning("Noise suppression failed, returning original: %s", exc)
            return audio.copy()
