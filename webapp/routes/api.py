"""
routes/api.py
JSON API endpoints — settings management.
"""
import os
import json
from pathlib import Path

from flask import Blueprint, jsonify, request, session

import config

api_bp = Blueprint("api", __name__)


def _resolve_llm_settings() -> dict[str, str]:
    """Resolve effective settings shown in the UI and used at runtime."""
    env_api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    env_base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
    env_model = (os.getenv("LLM_MODEL") or "").strip()

    session_api_key = (session.get("api_key") or "").strip()
    session_base_url = (session.get("base_url") or "").strip()
    session_model = (session.get("model") or "").strip()

    if env_api_key:
        return {
            "api_key": env_api_key,
            "base_url": env_base_url or config.OPENAI_BASE_URL,
            "model": env_model or config.LLM_MODEL,
            "source": "env",
        }

    return {
        "api_key": session_api_key or config.OPENAI_API_KEY,
        "base_url": session_base_url or env_base_url or config.OPENAI_BASE_URL,
        "model": session_model or env_model or config.LLM_MODEL,
        "source": "session" if session_api_key or session_base_url or session_model else "config",
    }


@api_bp.route("/settings", methods=["GET"])
def get_settings():
    settings = _resolve_llm_settings()
    return jsonify({
        "api_key":  "***" if settings.get("api_key") else "",
        "model":    settings.get("model", config.LLM_MODEL),
        "base_url": settings.get("base_url", config.OPENAI_BASE_URL),
        "source":   settings.get("source", "config"),
    })


@api_bp.route("/settings", methods=["POST"])
def save_settings():
    body = request.get_json(silent=True) or {}

    def _clean(v):
        return v.strip() if isinstance(v, str) else ""

    # Empty values now clear stale session overrides.
    if "api_key" in body:
        value = _clean(body.get("api_key"))
        if value:
            session["api_key"] = value
        else:
            session.pop("api_key", None)

    if "model" in body:
        value = _clean(body.get("model"))
        if value:
            session["model"] = value
        else:
            session.pop("model", None)

    if "base_url" in body:
        value = _clean(body.get("base_url"))
        if value:
            session["base_url"] = value
        else:
            session.pop("base_url", None)

    return jsonify({"ok": True, "message": "Settings saved. Empty fields clear session overrides."})


@api_bp.route("/result/<job_id>", methods=["GET"])
def get_result(job_id: str):
    """Return the full result JSON for a given job_id."""
    p = config.RESULTS_DIR / f"{job_id}.json"
    if not p.exists():
        return jsonify({"ok": False, "error": "Not found"}), 404
    return jsonify(json.loads(p.read_text(encoding="utf-8")))
