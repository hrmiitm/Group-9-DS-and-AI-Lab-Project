"""
webapp/config.py
Central configuration — all env vars and paths live here.
Services import from this module; they never call os.getenv directly.
"""
import os
import re
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
WEBAPP_DIR   = Path(__file__).parent.resolve()
PROJECT_ROOT = WEBAPP_DIR.parent
SRC_DIR      = PROJECT_ROOT / "src"
RESULTS_DIR  = WEBAPP_DIR / "results"
UPLOAD_DIR   = WEBAPP_DIR / "uploads"

# ── LLM (aipipe / OpenRouter) ───────────────────────────────────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://aipipe.org/openrouter/v1")
LLM_MODEL       = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
LLM_TEMPERATURE = 0.3

# ── Uploads ─────────────────────────────────────────────────────────────────
MAX_UPLOAD_BYTES  = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".html", ".md"}

# ── Flask ────────────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-change-in-prod")

# ── Tool display labels ──────────────────────────────────────────────────────
TOOL_LABELS = {
    "scam_signals":       "Scam Signal Scanner",
    "email_verify":       "Email Verification",
    "domain_reputation":  "Domain Reputation",
    "website_verify":     "Website Health Check",
    "website_content":    "Website Content Analysis",
    "company_wikipedia":  "Wikipedia Lookup",
    "company_web_search": "Company Web Search",
    "company_news":       "Recent Company News",
    "social_profiles":    "Social Media Presence",
    "job_boards":         "Job Board Verification",
    "phone_check":        "Phone Number Check",
    "company_registry":   "Company Registry",
}

TOOL_ICONS = {
    "scam_signals":       "🚨",
    "email_verify":       "📧",
    "domain_reputation":  "🌐",
    "website_verify":     "🔗",
    "website_content":    "📄",
    "company_wikipedia":  "📖",
    "company_web_search": "🔍",
    "company_news":       "📰",
    "social_profiles":    "👥",
    "job_boards":         "💼",
    "phone_check":        "📞",
    "company_registry":   "🏢",
}


def get_llm(temperature=None, api_key=None, base_url=None, model=None):
    """Return a configured LangChain ChatOpenAI instance.
    Override any parameter for per-call customisation.
    """
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model or LLM_MODEL,
        base_url=base_url or OPENAI_BASE_URL,
        api_key=api_key or OPENAI_API_KEY,
        temperature=temperature if temperature is not None else LLM_TEMPERATURE,
    )


def simple_markdown(text: str) -> str:
    """Convert basic markdown to HTML — no extra dependencies."""
    if not text:
        return ""
    import html as html_lib

    lines = text.split("\n")
    html_parts = []
    in_ul = False

    for line in lines:
        # Escape HTML first
        safe = html_lib.escape(line)

        # Headings
        if safe.startswith("### "):
            if in_ul:
                html_parts.append("</ul>"); in_ul = False
            html_parts.append(f"<h3>{safe[4:]}</h3>")
        elif safe.startswith("## "):
            if in_ul:
                html_parts.append("</ul>"); in_ul = False
            html_parts.append(f"<h2>{safe[3:]}</h2>")
        elif safe.startswith("# "):
            if in_ul:
                html_parts.append("</ul>"); in_ul = False
            html_parts.append(f"<h1>{safe[2:]}</h1>")
        # Bullet list
        elif safe.startswith("- ") or safe.startswith("* "):
            if not in_ul:
                html_parts.append("<ul>"); in_ul = True
            item = safe[2:]
            item = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", item)
            html_parts.append(f"<li>{item}</li>")
        # Horizontal rule
        elif safe.strip() in ("---", "***", "___"):
            if in_ul:
                html_parts.append("</ul>"); in_ul = False
            html_parts.append("<hr>")
        # Blank line
        elif safe.strip() == "":
            if in_ul:
                html_parts.append("</ul>"); in_ul = False
            html_parts.append("<br>")
        # Normal paragraph line
        else:
            if in_ul:
                html_parts.append("</ul>"); in_ul = False
            line_html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
            line_html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", line_html)
            line_html = re.sub(r"`(.+?)`", r"<code>\1</code>", line_html)
            # Convert [text](url) links
            line_html = re.sub(
                r"\[([^\]]+)\]\((https?://[^\)]+)\)",
                r'<a href="\2" target="_blank" rel="noopener">\1</a>',
                line_html,
            )
            html_parts.append(f"<p>{line_html}</p>")

    if in_ul:
        html_parts.append("</ul>")

    return "\n".join(html_parts)
