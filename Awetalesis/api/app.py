"""
api/app.py — FastAPI application exposing REST and WebSocket endpoints.

REST:
    GET  /           → minimal HTML UI
    POST /start      → start the S2ST pipeline
    POST /stop       → stop the pipeline
    GET  /status     → running state
    GET  /config     → current configuration
    POST /config/target → update target language

WebSocket:
    WS /ws/translate → live stream of translated audio + text events
"""

import asyncio
import base64
import json
from contextlib import asynccontextmanager
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import config
from pipeline.pipeline import S2STPipeline
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_pipeline: S2STPipeline = S2STPipeline()
_clients: Set[WebSocket] = set()


# ---------------------------------------------------------------------------
# WebSocket broadcast callback
# ---------------------------------------------------------------------------

async def _broadcast(audio_bytes: bytes, original: str, translated: str) -> None:
    """Send a result event to all connected WebSocket clients."""
    payload = json.dumps(
        {
            "original": original,
            "translated": translated,
            "audio_b64": base64.b64encode(audio_bytes).decode("utf-8"),
        }
    )
    disconnected = set()
    for ws in list(_clients):
        try:
            await ws.send_text(payload)
        except Exception:
            disconnected.add(ws)
    _clients.difference_update(disconnected)


# ---------------------------------------------------------------------------
# Application lifecycle (modern lifespan approach)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(application: FastAPI):
    global _pipeline
    _pipeline = S2STPipeline(on_result=_broadcast)
    logger.info("Awetalesis API ready.")
    yield
    await _pipeline.stop()


app = FastAPI(
    title="Awetalesis",
    description="Real-time Speech-to-Speech Translation API",
    version="1.0.0",
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, summary="Minimal UI")
async def root() -> str:
    """Return a minimal HTML control page."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <title>Awetalesis</title>
      <style>
        body { font-family: sans-serif; max-width: 600px; margin: 40px auto; }
        button { margin: 6px; padding: 8px 18px; cursor: pointer; }
        pre { background: #f4f4f4; padding: 12px; border-radius: 4px; }
      </style>
    </head>
    <body>
      <h1>🌐 Awetalesis</h1>
      <p><em>Speak once, understood everywhere.</em></p>
      <button onclick="fetch('/start',{method:'POST'})">▶ Start</button>
      <button onclick="fetch('/stop',{method:'POST'})">⏹ Stop</button>
      <h2>Translations</h2>
      <pre id="log">Connecting…</pre>
      <script>
        const log = document.getElementById('log');
        const ws = new WebSocket(`ws://${location.host}/ws/translate`);
        ws.onopen = () => { log.textContent = 'Connected. Waiting for speech…'; };
        ws.onmessage = e => {
          const d = JSON.parse(e.data);
          log.textContent = `[${d.original}]\n→ ${d.translated}\\n` + log.textContent;
        };
        ws.onerror = () => { log.textContent = 'WebSocket error'; };
      </script>
    </body>
    </html>
    """


@app.post("/start", summary="Start the pipeline")
async def start_pipeline() -> dict:
    """Start the S2ST pipeline."""
    await _pipeline.start()
    return {"status": "started"}


@app.post("/stop", summary="Stop the pipeline")
async def stop_pipeline() -> dict:
    """Stop the S2ST pipeline."""
    await _pipeline.stop()
    return {"status": "stopped"}


@app.get("/status", summary="Pipeline running state")
async def status() -> dict:
    """Return the current running state of the pipeline."""
    return {"running": _pipeline.is_running}


@app.get("/config", summary="Current configuration")
async def get_config() -> dict:
    """Return non-sensitive configuration values."""
    return {
        "sample_rate": config.SAMPLE_RATE,
        "channels": config.CHANNELS,
        "chunk_ms": config.CHUNK_MS,
        "source_language": config.SOURCE_LANGUAGE,
        "target_language": config.TARGET_LANGUAGE,
        "vad_aggressiveness": config.VAD_AGGRESSIVENESS,
        "whisper_model_size": config.WHISPER_MODEL_SIZE,
    }


@app.post("/config/target", summary="Update target language")
async def set_target_language(body: dict) -> dict:
    """Update the translation target language at runtime.

    Body: ``{"language": "fr"}``
    """
    language = body.get("language", "").strip()
    if not language:
        return {"error": "language field is required"}
    _pipeline.set_target_language(language)
    config.TARGET_LANGUAGE = language
    return {"target_language": language}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/translate")
async def ws_translate(websocket: WebSocket) -> None:
    """Live stream of translated audio + text events."""
    await websocket.accept()
    _clients.add(websocket)
    logger.info("WebSocket client connected (%d total).", len(_clients))
    try:
        while True:
            # Keep the connection alive; data is pushed via _broadcast
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("WebSocket error: %s", exc)
    finally:
        _clients.discard(websocket)
        logger.info("WebSocket client disconnected (%d total).", len(_clients))
