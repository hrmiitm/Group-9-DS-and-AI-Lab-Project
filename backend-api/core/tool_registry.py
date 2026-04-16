"""
core/tool_registry.py
Central registry of all 13 tools.

To add a new tool:
  1. Create tools/tool_<name>.py with a single exported function
  2. Add an entry to TOOL_REGISTRY below — that's it.
  The API, frontend metadata, and dynamic runner all auto-discover it.
"""
from __future__ import annotations

from typing import Any, Callable

# ── Import all tool functions ────────────────────────────────────────────────
from tools.tool_scam_signals      import detect_scam_signals
from tools.tool_email_verify      import verify_email
from tools.tool_domain_reputation import check_domain_reputation
from tools.tool_company_wikipedia import get_company_wikipedia
from tools.tool_company_web_search import search_company_web
from tools.tool_company_news      import search_company_news
from tools.tool_website_verify    import verify_website
from tools.tool_website_content   import extract_website_content
from tools.tool_social_profiles   import check_social_profiles
from tools.tool_job_boards        import check_job_boards
from tools.tool_phone_check       import check_phone_number
from tools.tool_company_registry  import get_company_registry
from tools.tool_roberta           import classify_job_roberta


# ── Registry ─────────────────────────────────────────────────────────────────
# Each entry describes: label, icon, description, required/optional inputs, output fields.
# The "fn" key holds the callable — used by the dynamic runner.

TOOL_REGISTRY: dict[str, dict[str, Any]] = {

    "scam_signals": {
        "label":       "Scam Signal Scanner",
        "icon":        "🚨",
        "description": "Keyword-based weighted scoring of the raw job text to detect common fraud signals (money demands, fake urgency, unofficial contacts, etc.). Pure Python — no API.",
        "fn":          detect_scam_signals,
        "input_schema": {
            "job_text": {"type": "string", "required": True, "description": "Raw job posting text to scan"},
        },
        "output_fields": ["scam_score", "risk_level", "signals_found", "signals_count", "matched_signals"],
        "category":    "text_analysis",
    },

    "email_verify": {
        "label":       "Email Verification",
        "icon":        "📧",
        "description": "Two-stage email check: syntax validation + DNS MX lookup. Also detects disposable domains and generic role-based emails.",
        "fn":          verify_email,
        "input_schema": {
            "email": {"type": "string", "required": True, "description": "Email address to verify"},
        },
        "output_fields": ["is_syntax_valid", "is_deliverable", "is_disposable", "is_role_account", "overall_status", "mx_host"],
        "category":    "contact",
    },

    "domain_reputation": {
        "label":       "Domain Reputation",
        "icon":        "🌐",
        "description": "WHOIS lookup to determine domain age, registrar, and risk level. Domains < 180 days old are HIGH risk; established domains are LOW risk.",
        "fn":          check_domain_reputation,
        "input_schema": {
            "domain_or_email": {"type": "string", "required": True, "description": "Domain name, email address, or full URL"},
        },
        "output_fields": ["domain", "domain_age_days", "risk_level", "registrar", "creation_date", "is_live"],
        "category":    "contact",
    },

    "website_verify": {
        "label":       "Website Health Check",
        "icon":        "🔗",
        "description": "Checks whether the company website is live, HTTPS-secured, and how many redirects occur. A missing or HTTP-only site is a red flag.",
        "fn":          verify_website,
        "input_schema": {
            "url": {"type": "string", "required": True, "description": "Company website URL"},
        },
        "output_fields": ["is_live", "ssl_valid", "status_code", "redirect_count", "response_time_ms", "server"],
        "category":    "website",
    },

    "website_content": {
        "label":       "Website Content Analysis",
        "icon":        "📄",
        "description": "Fetches and extracts meaningful text and metadata from the company website using trafilatura. Useful to verify company legitimacy from its own site.",
        "fn":          extract_website_content,
        "input_schema": {
            "url": {"type": "string", "required": True, "description": "Company website URL"},
        },
        "output_fields": ["extracted_text", "word_count", "metadata"],
        "category":    "website",
    },

    "company_wikipedia": {
        "label":       "Wikipedia Lookup",
        "icon":        "📖",
        "description": "Fetches the company's Wikipedia page summary via the public Wikipedia REST API. A well-documented Wikipedia entry suggests company legitimacy.",
        "fn":          get_company_wikipedia,
        "input_schema": {
            "company_name": {"type": "string", "required": True, "description": "Official company name"},
        },
        "output_fields": ["title", "description", "extract", "wikipedia_url", "thumbnail_url"],
        "category":    "company",
    },

    "company_web_search": {
        "label":       "Company Web Search",
        "icon":        "🔍",
        "description": "Runs 5 targeted DuckDuckGo searches: general info, employee reviews, scam/fraud signals, Glassdoor presence, and LinkedIn company page.",
        "fn":          search_company_web,
        "input_schema": {
            "company_name": {"type": "string", "required": True, "description": "Official company name"},
        },
        "output_fields": ["searches"],
        "category":    "company",
    },

    "company_news": {
        "label":       "Recent Company News",
        "icon":        "📰",
        "description": "Fetches recent DuckDuckGo News articles about the company. Useful to spot fraud cases, layoffs, funding rounds, or bad press.",
        "fn":          search_company_news,
        "input_schema": {
            "company_name": {"type": "string", "required": True, "description": "Official company name"},
            "max_results":  {"type": "integer", "required": False, "default": 8, "description": "Number of articles to fetch"},
        },
        "output_fields": ["total_articles", "articles"],
        "category":    "company",
    },

    "social_profiles": {
        "label":       "Social Media Presence",
        "icon":        "👥",
        "description": "Searches 7 major platforms (LinkedIn, Twitter/X, GitHub, Facebook, Instagram, YouTube, Glassdoor) for the company's official presence.",
        "fn":          check_social_profiles,
        "input_schema": {
            "company_name": {"type": "string", "required": True, "description": "Official company name"},
        },
        "output_fields": ["platforms_found", "profiles"],
        "category":    "company",
    },

    "job_boards": {
        "label":       "Job Board Verification",
        "icon":        "💼",
        "description": "Checks if the same job posting appears on 8 trusted job boards (LinkedIn, Indeed, Glassdoor, Naukri, etc.). Matching listings confirm legitimacy.",
        "fn":          check_job_boards,
        "input_schema": {
            "job_title":    {"type": "string", "required": True,  "description": "Job title from the posting"},
            "company_name": {"type": "string", "required": True,  "description": "Hiring company name"},
            "location":     {"type": "string", "required": False, "description": "Job location (optional)"},
        },
        "output_fields": ["boards_found", "verdict", "boards"],
        "category":    "company",
    },

    "phone_check": {
        "label":       "Phone Number Check",
        "icon":        "📞",
        "description": "Validates and parses a phone number using Google's libphonenumber. Returns carrier, region, and format details. Invalid numbers are a red flag.",
        "fn":          check_phone_number,
        "input_schema": {
            "phone":  {"type": "string", "required": True,  "description": "Phone number string (with or without country code)"},
            "region": {"type": "string", "required": False, "default": "IN", "description": "2-letter country hint (e.g. IN, US, GB)"},
        },
        "output_fields": ["e164", "is_valid", "region_code", "carrier", "location", "timezones"],
        "category":    "contact",
    },

    "company_registry": {
        "label":       "Company Registry",
        "icon":        "🏢",
        "description": "STUB — Planned integration with official company registries (Companies House UK, MCA21 India, SEC EDGAR). Currently returns not-implemented.",
        "fn":          get_company_registry,
        "input_schema": {
            "company_name": {"type": "string", "required": True, "description": "Official company name"},
        },
        "output_fields": [],
        "category":    "company",
        "is_stub":     True,
    },

    "roberta_classifier": {
        "label":       "RoBERTa Fraud Classifier",
        "icon":        "🤖",
        "description": "Runs the fine-tuned RoBERTa-base transformer (125M params, trained on EMSCAD dataset) to compute a fraud probability score. Threshold 0.87 → binary REAL/FAKE label.",
        "fn":          classify_job_roberta,
        "input_schema": {
            "job_text":  {"type": "string", "required": True, "description": "Raw job posting text (title + description recommended)"},
            "threshold": {"type": "number", "required": False, "default": 0.87, "description": "Custom decision threshold (0-1)"},
        },
        "output_fields": ["fraud_probability", "legit_probability", "is_fraud", "label", "confidence", "threshold_used", "model_id"],
        "category":    "ml_model",
    },
}


def get_tool_meta(include_fn: bool = False) -> dict:
    """Return registry metadata, optionally stripping the callable."""
    out = {}
    for name, entry in TOOL_REGISTRY.items():
        record = {k: v for k, v in entry.items() if k != "fn"} if not include_fn else entry
        out[name] = record
    return out


def get_tool_fn(tool_name: str) -> Callable | None:
    """Return the callable for a tool name, or None if not found."""
    entry = TOOL_REGISTRY.get(tool_name)
    return entry["fn"] if entry else None
