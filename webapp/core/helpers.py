"""
core/helpers.py
Shared utility functions for the tool pipeline.
Extracted from src/job_analyzer/run.py — no external dependencies.
"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse


# ── _safe_call ────────────────────────────────────────────────────────────────

def safe_call(tool_name: str, fn, *args, **kwargs) -> dict[str, Any]:
    """
    Call a tool function and always return a dict.
    On exception, returns {ok: False, error: "..."} so the pipeline keeps running.
    """
    try:
        result = fn(*args, **kwargs)
        if isinstance(result, dict):
            return result
        return {"ok": True, "data": result}
    except Exception as exc:
        return {"ok": False, "error": f"{tool_name} crashed: {exc}"}


# ── URL / company helpers ─────────────────────────────────────────────────────

def normalize_website(value: str | None) -> str | None:
    """Ensure a website string is a full HTTPS URL, or return None."""
    if not value:
        return None
    value = value.strip()
    if value.startswith("www."):
        return f"https://{value}"
    if value.startswith("http://") or value.startswith("https://"):
        return value
    if "." in value and " " not in value:
        return f"https://{value}"
    return None


def _clean_company_name(name: str | None) -> str | None:
    if not name:
        return None
    cleaned = re.sub(r"\s+", " ", name).strip(" -,:;\n\t")
    return cleaned or None


def _domain_to_company(domain_value: str) -> str | None:
    parsed = urlparse(
        domain_value if domain_value.startswith("http") else f"https://{domain_value}"
    )
    host = parsed.netloc.lower() if parsed.netloc else parsed.path.lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return None
    base = host.split(".")[0]
    return base.replace("-", " ").title() if base else None


def infer_company_name(
    raw_text: str,
    parsed_job: dict[str, Any],
    website: str | None,
    email: str | None,
) -> str | None:
    """
    Best-effort company name from multiple sources (in priority order):
    1. Parsed fields (company_name / company / organization)
    2. company_profile regex patterns
    3. Email domain
    4. Website domain
    5. Raw text regex
    """
    for field in ("company_name", "company", "organization"):
        if parsed_job.get(field):
            return _clean_company_name(str(parsed_job[field]))

    company_profile = parsed_job.get("company_profile") or ""
    for pat in (
        r"(?:Company|Organization)\s*[:\-]\s*([^\n,]{2,80})",
        r"About\s+([^\n,]{2,80})",
        r"at\s+([A-Z][A-Za-z0-9&.,\- ]{2,80})",
    ):
        match = re.search(pat, company_profile, flags=re.IGNORECASE)
        if match:
            return _clean_company_name(match.group(1))

    if email and "@" in email:
        from_email = _domain_to_company(email.split("@", 1)[1])
        if from_email:
            return from_email

    if website:
        from_website = _domain_to_company(website)
        if from_website:
            return from_website

    match = re.search(
        r"(?:Company|Organization)\s*[:\-]\s*([^\n,]{2,80})",
        raw_text,
        flags=re.IGNORECASE,
    )
    if match:
        return _clean_company_name(match.group(1))

    return None
