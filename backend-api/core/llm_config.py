"""
core/llm_config.py
LLM configuration with environment-variable-first, request-fallback pattern.

Priority:
  1. OS environment variables (set in HuggingFace Spaces secrets)
  2. Per-request override from request body (llm_config field in payload)
  3. Hard defaults (aipipe proxy, gpt-4.1-mini)
"""
from __future__ import annotations

import os
from typing import Optional
from pydantic import BaseModel


# ── Hard defaults ──────────────────────────────────────────────────────────────
DEFAULT_BASE_URL = "https://aipipe.org/openrouter/v1"
DEFAULT_MODEL    = "openai/gpt-4.1-mini"
DEFAULT_TEMP     = 0.3


class LLMConfig(BaseModel):
    """
    Optional per-request LLM override.
    Frontend sends this block when server env vars are not set.
    """
    api_key:          Optional[str] = None
    base_url:         Optional[str] = None
    model:            Optional[str] = None
    temperature:      Optional[float] = None


def resolve_llm(override: LLMConfig | None = None):
    """
    Return a configured LangChain ChatOpenAI instance.
    Env vars take precedence over per-request override.
    """
    from langchain_openai import ChatOpenAI

    # Resolve each setting: env → override → default
    api_key  = os.getenv("OPENAI_API_KEY")  or (override.api_key  if override else None) or ""
    base_url = os.getenv("OPENAI_BASE_URL") or (override.base_url if override else None) or DEFAULT_BASE_URL
    model    = os.getenv("LLM_MODEL")       or (override.model    if override else None) or DEFAULT_MODEL
    temp_env = os.getenv("LLM_TEMPERATURE")

    temperature = (
        float(temp_env) if temp_env
        else (override.temperature if override and override.temperature is not None else DEFAULT_TEMP)
    )

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
    )


def llm_settings_available() -> dict:
    """Return which settings are resolved from env vs will require caller override."""
    return {
        "api_key_from_env":  bool(os.getenv("OPENAI_API_KEY")),
        "base_url_from_env": bool(os.getenv("OPENAI_BASE_URL")),
        "model_from_env":    bool(os.getenv("LLM_MODEL")),
        "effective_base_url": os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL,
        "effective_model":    os.getenv("LLM_MODEL") or DEFAULT_MODEL,
    }
