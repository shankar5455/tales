"""tests/test_api.py — Integration tests for the FastAPI app."""

import sys
import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.app import app


@pytest.fixture(scope="module")
def client():
    """Provide a synchronous test client without starting the real pipeline."""
    with patch("api.app._pipeline") as mock_pipeline:
        mock_pipeline.is_running = False
        mock_pipeline.start = AsyncMock()
        mock_pipeline.stop = AsyncMock()
        mock_pipeline.set_target_language = AsyncMock()
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


class TestRootEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_returns_html(self, client):
        resp = client.get("/")
        assert "text/html" in resp.headers["content-type"]

    def test_contains_title(self, client):
        resp = client.get("/")
        assert "Awetalesis" in resp.text


class TestStatusEndpoint:
    def test_status_returns_200(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200

    def test_status_has_running_key(self, client):
        resp = client.get("/status")
        data = resp.json()
        assert "running" in data

    def test_status_running_is_bool(self, client):
        resp = client.get("/status")
        data = resp.json()
        assert isinstance(data["running"], bool)


class TestConfigEndpoint:
    def test_config_returns_200(self, client):
        resp = client.get("/config")
        assert resp.status_code == 200

    def test_config_has_sample_rate(self, client):
        resp = client.get("/config")
        data = resp.json()
        assert "sample_rate" in data

    def test_config_has_source_language(self, client):
        resp = client.get("/config")
        data = resp.json()
        assert "source_language" in data

    def test_config_has_target_language(self, client):
        resp = client.get("/config")
        data = resp.json()
        assert "target_language" in data


class TestConfigTargetEndpoint:
    def test_set_target_language(self, client):
        resp = client.post("/config/target", json={"language": "fr"})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("target_language") == "fr"

    def test_missing_language_returns_error(self, client):
        resp = client.post("/config/target", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data
