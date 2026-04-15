"""
routes/main.py
User-facing HTML routes.
"""
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import (
    Blueprint, redirect, render_template, request,
    session, url_for, flash,
)

# sys.path already set by app.py before blueprints are imported
import config

main_bp = Blueprint("main", __name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _result_path(job_id: str) -> Path:
    return config.RESULTS_DIR / f"{job_id}.json"


def _write_result(job_id: str, data: dict) -> None:
    _result_path(job_id).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _read_result(job_id: str) -> dict | None:
    p = _result_path(job_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _allowed_file(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in config.ALLOWED_EXTENSIONS


def _resolve_llm_settings() -> dict[str, str]:
    """
    Resolve effective LLM runtime settings.

    Priority:
    1) Explicit environment variables (proxy-safe)
    2) Session overrides (UI settings)
    3) Config defaults
    """
    env_api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    env_base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
    env_model = (os.getenv("LLM_MODEL") or "").strip()

    session_api_key = (session.get("api_key") or "").strip()
    session_base_url = (session.get("base_url") or "").strip()
    session_model = (session.get("model") or "").strip()

    # If OPENAI_API_KEY is explicitly exported, force env-based routing.
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


# ── Routes ───────────────────────────────────────────────────────────────────

@main_bp.route("/")
def index():
    # Pass any settings from session so the form can pre-fill
    settings = {
        "api_key":   session.get("api_key", ""),
        "model":     session.get("model", config.LLM_MODEL),
        "base_url":  session.get("base_url", config.OPENAI_BASE_URL),
    }
    return render_template("index.html", settings=settings)


@main_bp.route("/analyze", methods=["POST"])
def analyze():
    # ── Determine input type ────────────────────────────────────────────────
    input_type = request.form.get("input_type", "text")  # text | file | url
    job_id = str(uuid.uuid4())

    # Resolve effective LLM settings (env vars override session settings).
    llm_settings = _resolve_llm_settings()
    api_key = llm_settings["api_key"]
    base_url = llm_settings["base_url"]
    model = llm_settings["model"]

    # ── Validate & extract input ────────────────────────────────────────────
    input_data = None
    saved_filename = None

    if input_type == "text":
        text = request.form.get("job_text", "").strip()
        if not text:
            flash("Please paste a job description.", "error")
            return redirect(url_for("main.index"))
        input_data = text

    elif input_type == "file":
        f = request.files.get("job_file")
        if not f or not f.filename:
            flash("Please select a file to upload.", "error")
            return redirect(url_for("main.index"))
        if not _allowed_file(f.filename):
            flash(f"Unsupported file type. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}", "error")
            return redirect(url_for("main.index"))

        ext = Path(f.filename).suffix.lower()
        saved_filename = f"{job_id}{ext}"
        save_path = config.UPLOAD_DIR / saved_filename
        f.save(str(save_path))
        input_data = str(save_path)

    elif input_type == "url":
        url_val = request.form.get("linkedin_url", "").strip()
        if not url_val:
            flash("Please enter a LinkedIn job URL.", "error")
            return redirect(url_for("main.index"))
        input_data = url_val

    else:
        flash("Invalid input type.", "error")
        return redirect(url_for("main.index"))

    # ── Create initial result stub ──────────────────────────────────────────
    stub = {
        "job_id": job_id,
        "status": "processing",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_type": input_type,
        "saved_filename": saved_filename,
        "raw_text": None,
        "job_posting": None,
        "tool_results": {},
        "tool_inferences": {},
        "web_search_results": [],
        "final_report": None,
        "verdict": None,
        "error": None,
    }
    _write_result(job_id, stub)

    # ── Run analysis synchronously ──────────────────────────────────────────
    try:
        from services.analyzer import run_analysis
        run_analysis(
            job_id=job_id,
            input_type=input_type,
            input_data=input_data,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
    except Exception as exc:
        data = _read_result(job_id) or stub
        data["status"] = "error"
        data["error"] = str(exc)
        _write_result(job_id, data)

    return redirect(url_for("main.results", job_id=job_id))


@main_bp.route("/results/<job_id>")
def results(job_id: str):
    data = _read_result(job_id)
    if data is None:
        flash("Analysis not found. Please submit a new job description.", "error")
        return redirect(url_for("main.index"))
    return render_template("results.html", data=data)


@main_bp.route("/results")
def results_list():
    """Simple listing of all past analyses."""
    jobs = []
    for p in sorted(config.RESULTS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            jobs.append({
                "job_id":     d.get("job_id", p.stem),
                "status":     d.get("status", "?"),
                "verdict":    d.get("verdict"),
                "created_at": d.get("created_at", ""),
                "company":    (d.get("job_posting") or {}).get("company_name", "—"),
                "title":      (d.get("job_posting") or {}).get("title", "—"),
            })
        except Exception:
            pass
    return render_template("history.html", jobs=jobs)
