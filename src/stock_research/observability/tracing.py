"""LangSmith / LangChain tracing helpers."""

from __future__ import annotations

import os
from typing import Any

from stock_research.config import Settings


def configure_tracing(
    settings: Settings,
    *,
    enabled: bool | None = None,
) -> bool:
    """Apply tracing settings to the process environment.

    Returns True when tracing is active.
    """
    tracing_on = settings.langchain_tracing_v2 if enabled is None else enabled

    if not tracing_on:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGSMITH_TRACING_V2"] = "false"
        return False

    api_key = settings.langchain_api_key.strip()
    if not api_key:
        raise ValueError(
            "LANGCHAIN_API_KEY is required when tracing is enabled. "
            "Set it in .env or pass --no-trace."
        )

    project = settings.langchain_project.strip() or "us-stock-research"

    # Legacy LangChain env vars
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project

    # Current LangSmith env vars
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = project

    return True


def tracing_status() -> dict[str, Any]:
    """Return the current process tracing configuration."""
    enabled = os.environ.get("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes"}
    if not enabled:
        enabled = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() in {"1", "true", "yes"}
    return {
        "enabled": enabled,
        "project": os.environ.get("LANGSMITH_PROJECT")
        or os.environ.get("LANGCHAIN_PROJECT")
        or "",
        "has_api_key": bool(
            os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
        ),
    }
