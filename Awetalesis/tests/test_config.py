"""tests/test_config.py — Unit tests for config.py."""

import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _reload_config(env_overrides: dict) -> object:
    """Reload config module with the given environment overrides."""
    original_env = {}
    for key, value in env_overrides.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    import config
    importlib.reload(config)

    # Restore
    for key, original in original_env.items():
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original

    return config


class TestConfig:
    def test_default_sample_rate(self):
        import config
        assert config.SAMPLE_RATE == 16000

    def test_default_channels(self):
        import config
        assert config.CHANNELS == 1

    def test_default_chunk_ms(self):
        import config
        assert config.CHUNK_MS == 30

    def test_default_source_language(self):
        import config
        assert config.SOURCE_LANGUAGE == "en"

    def test_default_target_language(self):
        import importlib
        import config
        importlib.reload(config)
        assert config.TARGET_LANGUAGE == "es"

    def test_default_vad_aggressiveness(self):
        import config
        assert config.VAD_AGGRESSIVENESS in (0, 1, 2, 3)

    def test_env_override_sample_rate(self):
        cfg = _reload_config({"SAMPLE_RATE": "8000"})
        assert cfg.SAMPLE_RATE == 8000

    def test_env_override_target_language(self):
        cfg = _reload_config({"TARGET_LANGUAGE": "de"})
        assert cfg.TARGET_LANGUAGE == "de"

    def test_env_override_api_port(self):
        cfg = _reload_config({"API_PORT": "9000"})
        assert cfg.API_PORT == 9000

    def test_api_host_default(self):
        import config
        assert config.API_HOST == "0.0.0.0"

    def test_noise_reduce_prop_decrease_range(self):
        import config
        assert 0.0 <= config.NOISE_REDUCE_PROP_DECREASE <= 1.0
