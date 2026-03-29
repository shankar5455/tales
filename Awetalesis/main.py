"""
main.py — Entry point for the Awetalesis S2ST server.

Run with:
    python main.py
or:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import uvicorn

import config
from api.app import app  # noqa: F401  (re-exported for uvicorn)

if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
        log_level="info",
    )
