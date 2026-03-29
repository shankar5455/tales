"""
config.py — Central configuration for Awetalesis.

All settings are loaded from environment variables (via a .env file).
Defaults are provided so the system can start without all keys set.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))
CHANNELS: int = int(os.getenv("CHANNELS", "1"))
CHUNK_MS: int = int(os.getenv("CHUNK_MS", "30"))          # VAD frame length (ms)
BUFFER_SIZE_SEC: float = float(os.getenv("BUFFER_SIZE_SEC", "5"))

# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------
SOURCE_LANGUAGE: str = os.getenv("SOURCE_LANGUAGE", "en")
TARGET_LANGUAGE: str = os.getenv("TARGET_LANGUAGE", "es")

# ---------------------------------------------------------------------------
# Google APIs
# ---------------------------------------------------------------------------
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_PROJECT_ID: str = os.getenv("GOOGLE_PROJECT_ID", "")

# ---------------------------------------------------------------------------
# ElevenLabs TTS
# ---------------------------------------------------------------------------
ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

# ---------------------------------------------------------------------------
# Whisper fallback
# ---------------------------------------------------------------------------
WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "tiny")

# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------
VAD_AGGRESSIVENESS: int = int(os.getenv("VAD_AGGRESSIVENESS", "2"))  # 0–3
VAD_SILENCE_THRESHOLD_MS: int = int(os.getenv("VAD_SILENCE_THRESHOLD_MS", "400"))

# ---------------------------------------------------------------------------
# Noise suppression
# ---------------------------------------------------------------------------
NOISE_REDUCE_PROP_DECREASE: float = float(os.getenv("NOISE_REDUCE_PROP_DECREASE", "0.75"))

# ---------------------------------------------------------------------------
# API server
# ---------------------------------------------------------------------------
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
