# 🌐 Awetalesis

> **Speak once, understood everywhere.**

Awetalesis is a real-time, CPU-optimised, API-first **Speech-to-Speech Translation (S2ST)** system.  
It listens to microphone input, transcribes speech, translates the text, synthesises the translation, and streams the result to any connected client — all asynchronously.

---

## 📋 Table of Contents

- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [API Keys Setup](#api-keys-setup)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Running Tests](#running-tests)
- [Fallback Logic](#fallback-logic)

---

## Architecture

```
Microphone Input
      │
      ▼
AudioStream  ──►  AudioBuffer
                      │
                      ▼
               NoiseSuppressor
                      │
                      ▼
                     VAD  (WebRTC VAD)
                      │  (utterance detected)
                      ▼
                     ASR  ── Google STT  ──► Faster-Whisper (fallback)
                      │
                      ▼
          LanguageIdentifier  ── Google Detect ──► langdetect (fallback)
                      │
                      ▼
                 Translator  ── Google Translate ──► original text (fallback)
                      │
                      ▼
                     TTS  ── ElevenLabs ──► gTTS (fallback)
                      │
                      ▼
          WebSocket broadcast (audio + text)
```

---

## Requirements

| Requirement | Detail |
|---|---|
| Python | 3.10 or higher |
| CPU | Any modern CPU (no GPU required) |
| RAM | Minimum 2 GB free (8 GB system recommended) |
| Microphone | Any audio input device |
| OS | Linux / macOS / Windows |

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd Awetalesis

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy the environment template
cp .env.example .env
# Edit .env and add your API keys (see below)
```

> **Note (Linux):** `webrtcvad` requires `portaudio` headers.  
> Install with: `sudo apt install portaudio19-dev python3-dev`

---

## API Keys Setup

Edit `.env` with your credentials:

### Google Cloud

1. Go to [Google Cloud Console](https://console.cloud.google.com/).
2. Create a project and enable:
   - **Cloud Speech-to-Text API**
   - **Cloud Translation API**
3. Create an **API Key** under *APIs & Services → Credentials*.
4. Set `GOOGLE_API_KEY=<your key>` in `.env`.

### ElevenLabs (TTS)

1. Sign up at [elevenlabs.io](https://elevenlabs.io/).
2. Navigate to *Profile → API Key*.
3. Set `ELEVENLABS_API_KEY=<your key>` in `.env`.
4. Optionally set `ELEVENLABS_VOICE_ID` to a voice from your library.

---

## Usage

### Start the server

```bash
python main.py
# or
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### Open the browser UI

Navigate to [http://localhost:8000](http://localhost:8000)

### Start / stop translation

```bash
# Start
curl -X POST http://localhost:8000/start

# Stop
curl -X POST http://localhost:8000/stop
```

### Change target language at runtime

```bash
curl -X POST http://localhost:8000/config/target \
     -H "Content-Type: application/json" \
     -d '{"language": "fr"}'
```

---

## API Reference

### REST

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Minimal HTML control UI |
| `POST` | `/start` | Start the S2ST pipeline |
| `POST` | `/stop` | Stop the S2ST pipeline |
| `GET` | `/status` | `{"running": true\|false}` |
| `GET` | `/config` | Current (non-sensitive) configuration |
| `POST` | `/config/target` | Update target language `{"language": "de"}` |

### WebSocket

| Path | Description |
|------|-------------|
| `ws://host/ws/translate` | Streams JSON events with `original`, `translated`, and `audio_b64` (MP3 base64) fields |

#### WebSocket event format

```json
{
  "original":   "Hello, how are you?",
  "translated": "Hola, ¿cómo estás?",
  "audio_b64":  "<base64-encoded MP3 bytes>"
}
```

---

## Configuration

All settings live in `.env` (copied from `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | *(empty)* | Google Cloud API key |
| `ELEVENLABS_API_KEY` | *(empty)* | ElevenLabs API key |
| `ELEVENLABS_VOICE_ID` | `21m00Tcm4TlvDq8ikWAM` | ElevenLabs voice (Rachel) |
| `SOURCE_LANGUAGE` | `en` | Default source language |
| `TARGET_LANGUAGE` | `es` | Default target language |
| `SAMPLE_RATE` | `16000` | Audio sample rate (Hz) |
| `CHUNK_MS` | `30` | Audio frame size (ms) |
| `VAD_AGGRESSIVENESS` | `2` | WebRTC VAD level (0–3) |
| `VAD_SILENCE_THRESHOLD_MS` | `400` | Silence gap to end utterance |
| `NOISE_REDUCE_PROP_DECREASE` | `0.75` | Noise reduction strength |
| `WHISPER_MODEL_SIZE` | `tiny` | Whisper fallback model size |
| `API_HOST` | `0.0.0.0` | Server bind address |
| `API_PORT` | `8000` | Server port |

---

## Running Tests

```bash
# From the Awetalesis/ directory
pip install pytest pytest-asyncio
pytest tests/ -v
```

---

## Fallback Logic

| Stage | Primary | Fallback |
|-------|---------|---------|
| ASR | Google Speech-to-Text | faster-whisper tiny (CPU) |
| LID | Google Translate detect | langdetect |
| Translation | Google Translate | Return original text |
| TTS | ElevenLabs | gTTS |

The system **never crashes** on API failure — every stage has a graceful fallback or degraded-mode response.

---

## Project Structure

```
Awetalesis/
├── main.py              # Entry point
├── config.py            # Central configuration
├── requirements.txt
├── README.md
├── .env.example
│
├── audio/
│   ├── buffer.py        # Thread-safe circular buffer
│   └── stream.py        # Microphone capture
│
├── processing/
│   ├── noise_suppression.py
│   ├── vad.py           # WebRTC VAD
│   ├── lid.py           # Language identification
│   ├── asr.py           # Speech recognition
│   ├── translation.py   # Text translation
│   └── tts.py           # Text-to-speech
│
├── pipeline/
│   └── pipeline.py      # Async orchestration
│
├── api/
│   └── app.py           # FastAPI REST + WebSocket
│
├── utils/
│   └── logger.py        # Centralised logging
│
└── tests/
    ├── test_audio_buffer.py
    ├── test_config.py
    ├── test_noise_suppression.py
    ├── test_vad.py
    ├── test_translation.py
    └── test_api.py
```

---

## License

MIT © Awetalesis Contributors
