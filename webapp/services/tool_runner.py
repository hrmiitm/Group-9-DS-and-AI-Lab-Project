"""
services/tool_runner.py
Imports and runs all 12 investigative tools against a parsed job posting.
All imports come from within webapp/ — fully self-contained.
"""
from __future__ import annotations

import re
from typing import Any

from core.helpers import safe_call, normalize_website, infer_company_name
from tools.tool_email_verify       import verify_email
from tools.tool_domain_reputation  import check_domain_reputation
from tools.tool_company_wikipedia  import get_company_wikipedia
from tools.tool_company_web_search import search_company_web
from tools.tool_company_news       import search_company_news
from tools.tool_website_verify     import verify_website
from tools.tool_website_content    import extract_website_content
from tools.tool_social_profiles    import check_social_profiles
from tools.tool_job_boards         import check_job_boards
from tools.tool_phone_check        import check_phone_number
from tools.tool_scam_signals       import detect_scam_signals
from tools.tool_company_registry   import get_company_registry

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_URL_RE   = re.compile(r"https?://[^\s)\]>'\"]+|www\.[^\s)\]>'\"]+")
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")


def _first_match(pattern: re.Pattern, text: str) -> str | None:
    m = pattern.findall(text)
    return m[0].strip() if m else None


def _skip(reason: str) -> dict[str, Any]:
    return {"ok": False, "error": f"Skipped: {reason} not available in posting"}


def run_all_tools(job, raw_text: str) -> dict[str, Any]:
    """
    Run all 12 tools against the parsed job data.

    Args:
        job      : JobPosting Pydantic instance from core/job_parser_agent
        raw_text : raw document text (used by scam_signals + fallback field extraction)

    Returns:
        dict keyed by tool name, each value is {ok: bool, data/error: ...}
    """
    job_dict = job.model_dump()

    # ── Extract / infer key fields ────────────────────────────────────────────
    email   = job_dict.get("contact_email") or _first_match(_EMAIL_RE, raw_text)
    website = normalize_website(job_dict.get("company_website") or _first_match(_URL_RE, raw_text))
    phone   = job_dict.get("contact_phone") or _first_match(_PHONE_RE, raw_text)
    company = infer_company_name(
        raw_text=raw_text,
        parsed_job=job_dict,
        website=website,
        email=email,
    )
    title    = job_dict.get("title")
    location = job_dict.get("location")

    # WHOIS can use email domain or website
    domain_input = email if email else website

    results: dict[str, Any] = {}

    # ── Always-run tools ──────────────────────────────────────────────────────
    results["scam_signals"]     = safe_call("scam_signals",     detect_scam_signals,   raw_text)
    results["company_registry"] = safe_call("company_registry", get_company_registry,  company)

    # ── Email ─────────────────────────────────────────────────────────────────
    results["email_verify"] = (
        safe_call("email_verify", verify_email, email)
        if email else _skip("email")
    )

    # ── Domain / website ─────────────────────────────────────────────────────
    results["domain_reputation"] = (
        safe_call("domain_reputation", check_domain_reputation, domain_input)
        if domain_input else _skip("email/website")
    )
    results["website_verify"] = (
        safe_call("website_verify", verify_website, website)
        if website else _skip("website")
    )
    results["website_content"] = (
        safe_call("website_content", extract_website_content, website)
        if website else _skip("website")
    )

    # ── Company research ──────────────────────────────────────────────────────
    results["company_wikipedia"] = (
        safe_call("company_wikipedia", get_company_wikipedia, company)
        if company else _skip("company name")
    )
    results["company_web_search"] = (
        safe_call("company_web_search", search_company_web, company)
        if company else _skip("company name")
    )
    results["company_news"] = (
        safe_call("company_news", search_company_news, company)
        if company else _skip("company name")
    )
    results["social_profiles"] = (
        safe_call("social_profiles", check_social_profiles, company)
        if company else _skip("company name")
    )

    # ── Job boards ────────────────────────────────────────────────────────────
    results["job_boards"] = (
        safe_call("job_boards", check_job_boards, title, company, location)
        if title and company else _skip("title + company name")
    )

    # ── Phone ─────────────────────────────────────────────────────────────────
    results["phone_check"] = (
        safe_call("phone_check", check_phone_number, phone)
        if phone else _skip("phone number")
    )

    return results
