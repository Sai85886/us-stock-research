"""Tests for LangSmith tracing configuration."""

from __future__ import annotations

import os

import pytest

from stock_research.config import Settings
from stock_research.observability.tracing import configure_tracing, tracing_status


def test_configure_tracing_disabled():
    settings = Settings()
    assert configure_tracing(settings, enabled=False) is False
    status = tracing_status()
    assert status["enabled"] is False


def test_configure_tracing_enabled_sets_env(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_test_key")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "demo-project")
    settings = Settings()

    assert configure_tracing(settings, enabled=True) is True
    assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
    assert os.environ.get("LANGSMITH_TRACING") == "true"
    assert os.environ.get("LANGCHAIN_API_KEY") == "lsv2_test_key"
    assert os.environ.get("LANGSMITH_PROJECT") == "demo-project"

    status = tracing_status()
    assert status["enabled"] is True
    assert status["project"] == "demo-project"
    assert status["has_api_key"] is True


def test_configure_tracing_requires_api_key_when_forced(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_API_KEY", "")
    monkeypatch.setenv("LANGSMITH_API_KEY", "")
    settings = Settings()
    # Ensure empty even if .env had a key during Settings init.
    object.__setattr__(settings, "langchain_api_key", "")
    with pytest.raises(ValueError, match="LANGCHAIN_API_KEY"):
        configure_tracing(settings, enabled=True)
