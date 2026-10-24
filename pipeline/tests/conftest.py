import importlib

import pytest

from pipeline import config


@pytest.fixture
def reload_config(monkeypatch):
    """Set env vars and reload config. Teardown reloads again so later tests see the real env."""

    def _reload(**env):
        for key, value in env.items():
            monkeypatch.setenv(key, str(value))
        return importlib.reload(config)

    yield _reload
    monkeypatch.undo()
    importlib.reload(config)
