"""
services/analyzer.py
Orchestrates the fraud analysis pipeline with enrichment support.

Step 1  — Extract structured job info (LLM via core/job_parser_agent)
Step 1b — Deep research enrichment for missing JD fields (DuckDuckGo)
Step 2  — Run 12 investigative tools on primary data (services/tool_runner)
Step 2b — Run candidate tool checks for all discovered contacts/websites
Step 3  — Per-tool LLM inference (one call per primary tool)
Step 4  — Web search + final LLM fraud report
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import config
from core.helpers import infer_company_name, normalize_website, safe_call

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_URL_RE = re.compile(r"https?://[^\s)\]>'\"]+|www\.[^\s)\]>'\"]+")
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")

_FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "proton.me", "protonmail.com",
}
_NOISY_HOSTS = (
    "linkedin.com", "glassdoor.com", "indeed.", "naukri.", "facebook.com",
    "instagram.com", "x.com", "twitter.com", "youtube.com",
)

_MAX_EMAIL_CANDIDATES = 6
_MAX_PHONE_CANDIDATES = 6
_MAX_WEBSITE_CANDIDATES = 5


# ── LLM Prompt Templates ─────────────────────────────────────────────────────

_TOOL_INFERENCE_SYSTEM = """You are a fraud job detection specialist.
You receive the raw output of one investigative tool run against a job posting.
In 2-4 sentences explain:
1. What the tool found (factual summary using specific values from the data)
2. What this means for fraud risk (low / medium / high signal and why)
If the tool failed or returned no data, say "Inconclusive — tool did not return data."
Do not speculate beyond what the data shows."""

_FINAL_REPORT_SYSTEM = """You are a senior fraud investigation analyst writing a report for a job seeker.
You have evidence from 12 investigative tools and web search results.

Structure your report using EXACTLY these markdown sections:

## Fraud Risk Assessment

**Verdict: SAFE** or **Verdict: SUSPICIOUS** or **Verdict: LIKELY_FAKE**
**Confidence: Low** or **Confidence: Medium** or **Confidence: High**

## Executive Summary
2-3 sentences on the overall finding.

## Evidence Analysis

### Supporting Legitimacy
Bullet points of evidence that supports the job being real.

### Red Flags
Bullet points of suspicious signals found. If none, say "No significant red flags detected."

## Tool Evidence Summary
Brief paragraph summarising what the 12 tools collectively found.

## Web Intelligence
What web search revealed about the company and job. Cite specific URLs when useful.

## Recommendation
Clear, actionable advice for the job seeker: what to do next, what to verify, whether to apply.

Rules:
- Be factual. Use specific values from the tool data.
- Distinguish "not found" (neutral/inconclusive) from "found fraud signals" (negative).
- If additional contact data was discovered externally, explicitly mention it as deep research data.
- Do not invent facts not present in the evidence."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_patch(result_path: Path, patch: dict) -> None:
    """Read existing JSON, merge patch, write back."""
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    data.update(patch)
    result_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _get_llm(api_key: str | None, base_url: str | None, model: str | None, temperature: float = 0.3):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model or config.LLM_MODEL,
        base_url=base_url or config.OPENAI_BASE_URL,
        api_key=api_key or config.OPENAI_API_KEY,
        temperature=temperature,
    )


def _first_match(pattern: re.Pattern, text: str) -> str | None:
    matches = pattern.findall(text or "")
    return matches[0].strip() if matches else None


def _normalize_phone(value: str) -> str | None:
    raw = (value or "").strip()
    if not raw:
        return None
    has_plus = raw.startswith("+")
    digits = re.sub(r"\D", "", raw)
    if len(digits) < 10 or len(digits) > 15:
        return None
    return f"+{digits}" if has_plus else digits


def _domain_of_email(email: str) -> str:
    return email.split("@", 1)[1].lower() if "@" in email else ""


def _host_of_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _company_token(company: str | None) -> str:
    if not company:
        return ""
    return re.sub(r"[^a-z0-9]", "", company.lower())


def _score_email(email: str, company_tok: str) -> int:
    email_l = email.lower()
    domain = _domain_of_email(email_l)
    score = 0
    if company_tok and company_tok in re.sub(r"[^a-z0-9]", "", domain):
        score += 4
    if domain in _FREE_EMAIL_DOMAINS:
        score -= 2
    local = email_l.split("@", 1)[0]
    if any(x in local for x in ("hr", "recruit", "career", "jobs", "talent")):
        score += 2
    return score


def _score_website(url: str, company_tok: str) -> int:
    host = _host_of_url(url)
    score = 0
    if url.startswith("https://"):
        score += 1
    if company_tok and company_tok in re.sub(r"[^a-z0-9]", "", host):
        score += 4
    if any(noisy in host for noisy in _NOISY_HOSTS):
        score -= 3
    if any(path_hint in url.lower() for path_hint in ("/careers", "/jobs", "/about")):
        score += 1
    return score


def _add_candidate(
    bucket: dict[str, dict[str, Any]],
    value: str,
    query: str,
    title: str,
    url: str,
) -> None:
    if not value:
        return
    item = bucket.setdefault(value, {"value": value, "origin": "deep_research", "evidence": []})
    if len(item["evidence"]) < 3:
        item["evidence"].append({"query": query, "title": title, "url": url})


def _compact_result(tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {"ok": True, "data": result}
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error", "Tool failed")}

    data = dict(result.get("data") or {})
    if tool_name == "website_content":
        text = data.get("extracted_text") or ""
        if isinstance(text, str) and len(text) > 1200:
            data["extracted_text"] = text[:1200] + "..."
            data["extracted_text_truncated"] = True
    return {"ok": True, "data": data}


def _candidate_groups_from_jd(job_dict: dict[str, Any], raw_text: str) -> dict[str, str | None]:
    return {
        "contact_email": job_dict.get("contact_email") or _first_match(_EMAIL_RE, raw_text),
        "contact_phone": job_dict.get("contact_phone") or _normalize_phone(_first_match(_PHONE_RE, raw_text) or ""),
        "company_website": normalize_website(job_dict.get("company_website") or _first_match(_URL_RE, raw_text)),
    }


# ── Step 1b: Deep research enrichment ────────────────────────────────────────

def deep_research_missing_fields(job_dict: dict[str, Any], raw_text: str) -> dict[str, Any]:
    """
    Search external sources to recover key missing fields not present in the JD.
    Returns candidate contacts/websites plus selected field overrides.
    """
    from ddgs import DDGS

    jd_values = _candidate_groups_from_jd(job_dict, raw_text)
    missing_from_jd = [k for k, v in jd_values.items() if not v]

    company = job_dict.get("company_name") or infer_company_name(
        raw_text=raw_text,
        parsed_job=job_dict,
        website=jd_values.get("company_website"),
        email=jd_values.get("contact_email"),
    )
    title = job_dict.get("title") or ""
    location = job_dict.get("location") or ""

    if not company and not title:
        return {
            "ran": False,
            "missing_from_jd": missing_from_jd,
            "summary": {"emails_found": 0, "phones_found": 0, "websites_found": 0},
            "candidates": {"emails": [], "phones": [], "websites": []},
            "search_queries": [],
            "evidence": [],
            "applied_overrides": {},
            "notes": ["Deep research skipped: company/title context unavailable."],
        }

    queries = []
    if company:
        queries.extend([
            f'"{company}" official website contact email phone',
            f'"{company}" careers recruiter email',
            f'"{company}" HR contact {location}'.strip(),
        ])
    if company and title:
        queries.append(f'"{company}" "{title}" hiring email phone apply')
    if title and not company:
        queries.append(f'"{title}" recruiter email phone official website')

    # Keep order while removing accidental duplicates.
    seen_q: set[str] = set()
    deduped_queries: list[str] = []
    for q in queries:
        q = q.strip()
        if q and q not in seen_q:
            seen_q.add(q)
            deduped_queries.append(q)

    email_bucket: dict[str, dict[str, Any]] = {}
    phone_bucket: dict[str, dict[str, Any]] = {}
    website_bucket: dict[str, dict[str, Any]] = {}
    evidence: list[dict[str, str]] = []

    try:
        with DDGS() as ddgs:
            for query in deduped_queries:
                try:
                    hits = ddgs.text(query, max_results=5)
                except Exception:
                    continue

                for r in (hits or []):
                    url = normalize_website(r.get("href") or "") or ""
                    title_text = (r.get("title") or "").strip()
                    snippet = (r.get("body") or "").strip()
                    evidence.append({"query": query, "title": title_text, "url": url, "snippet": snippet})

                    text_blob = "\n".join([title_text, snippet, url])

                    for email in _EMAIL_RE.findall(text_blob):
                        _add_candidate(email_bucket, email.lower(), query, title_text, url)

                    for phone_raw in _PHONE_RE.findall(text_blob):
                        phone = _normalize_phone(phone_raw)
                        if phone:
                            _add_candidate(phone_bucket, phone, query, title_text, url)

                    for url_raw in _URL_RE.findall(text_blob):
                        website = normalize_website(url_raw)
                        if website:
                            _add_candidate(website_bucket, website, query, title_text, url)

                    if url:
                        _add_candidate(website_bucket, url, query, title_text, url)
    except Exception:
        pass

    company_tok = _company_token(company)

    emails = sorted(
        email_bucket.values(),
        key=lambda x: (_score_email(x["value"], company_tok), len(x.get("evidence") or [])),
        reverse=True,
    )[:_MAX_EMAIL_CANDIDATES]

    phones = sorted(
        phone_bucket.values(),
        key=lambda x: len(x.get("evidence") or []),
        reverse=True,
    )[:_MAX_PHONE_CANDIDATES]

    websites = sorted(
        website_bucket.values(),
        key=lambda x: (_score_website(x["value"], company_tok), len(x.get("evidence") or [])),
        reverse=True,
    )[:_MAX_WEBSITE_CANDIDATES]

    applied_overrides: dict[str, str] = {}
    if not jd_values.get("contact_email") and emails:
        applied_overrides["contact_email"] = emails[0]["value"]
    if not jd_values.get("contact_phone") and phones:
        applied_overrides["contact_phone"] = phones[0]["value"]
    if not jd_values.get("company_website") and websites:
        applied_overrides["company_website"] = websites[0]["value"]

    return {
        "ran": True,
        "missing_from_jd": missing_from_jd,
        "summary": {
            "emails_found": len(emails),
            "phones_found": len(phones),
            "websites_found": len(websites),
            "external_sources_scanned": len(evidence),
        },
        "company_context": company,
        "search_queries": deduped_queries,
        "candidates": {
            "emails": emails,
            "phones": phones,
            "websites": websites,
        },
        "evidence": evidence[:20],
        "applied_overrides": applied_overrides,
        "notes": [
            "All deep research candidates are external findings and may include false positives.",
            "Candidate values are verified with investigative tools before presentation.",
        ],
    }


# ── Step 2b: Candidate tool checks ───────────────────────────────────────────

def run_candidate_tool_checks(
    job_dict: dict[str, Any],
    raw_text: str,
    deep_research: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Run tools against all candidate emails/phones/websites (JD + deep research).
    """
    from tools.tool_domain_reputation import check_domain_reputation
    from tools.tool_email_verify import verify_email
    from tools.tool_phone_check import check_phone_number
    from tools.tool_website_content import extract_website_content
    from tools.tool_website_verify import verify_website

    jd_values = _candidate_groups_from_jd(job_dict, raw_text)

    research_candidates = (deep_research or {}).get("candidates") or {}
    research_email_items = research_candidates.get("emails") or []
    research_phone_items = research_candidates.get("phones") or []
    research_website_items = research_candidates.get("websites") or []

    emails: list[dict[str, Any]] = []
    phones: list[dict[str, Any]] = []
    websites: list[dict[str, Any]] = []

    seen_email: set[str] = set()
    seen_phone: set[str] = set()
    seen_website: set[str] = set()

    def add_item(bucket: list[dict[str, Any]], seen: set[str], value: str | None, source: str, evidence=None):
        if not value:
            return
        key = value.lower().strip()
        if key in seen:
            return
        seen.add(key)
        bucket.append({
            "value": value,
            "source": source,
            "from_jd": source == "jd",
            "evidence": evidence or [],
        })

    add_item(emails, seen_email, jd_values.get("contact_email"), "jd")
    add_item(phones, seen_phone, jd_values.get("contact_phone"), "jd")
    add_item(websites, seen_website, jd_values.get("company_website"), "jd")

    for item in research_email_items[:_MAX_EMAIL_CANDIDATES]:
        add_item(emails, seen_email, item.get("value"), "deep_research", item.get("evidence"))
    for item in research_phone_items[:_MAX_PHONE_CANDIDATES]:
        add_item(phones, seen_phone, item.get("value"), "deep_research", item.get("evidence"))
    for item in research_website_items[:_MAX_WEBSITE_CANDIDATES]:
        add_item(websites, seen_website, item.get("value"), "deep_research", item.get("evidence"))

    for item in emails:
        email_value = item["value"]
        item["results"] = {
            "email_verify": _compact_result("email_verify", safe_call("email_verify", verify_email, email_value)),
            "domain_reputation": _compact_result(
                "domain_reputation", safe_call("domain_reputation", check_domain_reputation, email_value)
            ),
        }

    for item in phones:
        phone_value = item["value"]
        item["results"] = {
            "phone_check": _compact_result("phone_check", safe_call("phone_check", check_phone_number, phone_value)),
        }

    for item in websites:
        website_value = item["value"]
        item["results"] = {
            "website_verify": _compact_result("website_verify", safe_call("website_verify", verify_website, website_value)),
            "website_content": _compact_result(
                "website_content", safe_call("website_content", extract_website_content, website_value)
            ),
            "domain_reputation": _compact_result(
                "domain_reputation", safe_call("domain_reputation", check_domain_reputation, website_value)
            ),
        }

    return {
        "summary": {
            "email_candidates": len(emails),
            "phone_candidates": len(phones),
            "website_candidates": len(websites),
            "deep_research_candidates": (
                len([x for x in emails if x.get("source") == "deep_research"]) +
                len([x for x in phones if x.get("source") == "deep_research"]) +
                len([x for x in websites if x.get("source") == "deep_research"])
            ),
        },
        "emails": emails,
        "phones": phones,
        "websites": websites,
    }


# ── Step 3: Per-tool inference ────────────────────────────────────────────────

def infer_tool_result(
    tool_name: str,
    tool_result: dict,
    job_dict: dict,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> str:
    """Call LLM to interpret a single tool's output in fraud context."""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = _get_llm(api_key, base_url, model, temperature=0.2)
    label = config.TOOL_LABELS.get(tool_name, tool_name)
    company = job_dict.get("company_name") or "Unknown"
    title = job_dict.get("title") or "Unknown"
    payload = json.dumps(tool_result, indent=2, default=str)[:3000]

    messages = [
        SystemMessage(content=_TOOL_INFERENCE_SYSTEM),
        HumanMessage(content=(
            f"Tool: {label}\n"
            f"Company: {company}\n"
            f"Job Title: {title}\n\n"
            f"Tool Output:\n{payload}"
        )),
    ]
    try:
        resp = llm.invoke(messages)
        return resp.content.strip()
    except Exception as exc:
        return f"Inference unavailable: {exc}"


# ── Step 4a: Web search ───────────────────────────────────────────────────────

def web_search_fraud_signals(company_name: str, job_title: str) -> list[dict]:
    """Run targeted DuckDuckGo searches for fraud signals."""
    from ddgs import DDGS

    queries = [
        f'"{company_name}" fraud scam fake job complaint',
        f'"{company_name}" "{job_title}" legitimate real review',
        f'"{company_name}" employee glassdoor cheating warning',
    ]
    results: list[dict] = []
    try:
        with DDGS() as ddgs:
            for q in queries:
                try:
                    hits = ddgs.text(q, max_results=4)
                    for r in (hits or []):
                        results.append({
                            "query": q,
                            "title": r.get("title", ""),
                            "url": r.get("href", ""),
                            "snippet": r.get("body", ""),
                        })
                except Exception:
                    pass
    except Exception:
        pass
    return results[:12]


# ── Step 4b: Final report ─────────────────────────────────────────────────────

def generate_final_report(
    job_dict: dict,
    tool_inferences: dict[str, str],
    web_results: list[dict],
    deep_research: dict[str, Any] | None = None,
    candidate_tool_results: dict[str, Any] | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> tuple[str, str]:
    """
    Generate the detailed markdown fraud report.
    Returns (report_markdown, verdict).
    verdict is one of: SAFE | SUSPICIOUS | LIKELY_FAKE
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = _get_llm(api_key, base_url, model, temperature=0.4)

    tool_evidence = "\n\n".join(
        f"**{config.TOOL_LABELS.get(name, name)}**: {inference}"
        for name, inference in tool_inferences.items()
    )

    web_lines = [
        f"- [{r['title']}]({r['url']}): {(r.get('snippet') or '')[:200]}"
        for r in web_results if r.get("title")
    ]
    web_summary = "\n".join(web_lines) or "No web search results found."

    deep_summary = "No external enrichment was needed."
    if deep_research:
        missing = ", ".join(deep_research.get("missing_from_jd") or []) or "none"
        found = deep_research.get("summary") or {}
        deep_summary = (
            f"Missing in JD: {missing}. "
            f"Deep research found emails={found.get('emails_found', 0)}, "
            f"phones={found.get('phones_found', 0)}, websites={found.get('websites_found', 0)}."
        )

    candidate_summary = "No candidate-level verification was run."
    if candidate_tool_results:
        cs = candidate_tool_results.get("summary") or {}
        candidate_summary = (
            f"Candidate checks: emails={cs.get('email_candidates', 0)}, "
            f"phones={cs.get('phone_candidates', 0)}, websites={cs.get('website_candidates', 0)}. "
            f"Deep research candidates={cs.get('deep_research_candidates', 0)}."
        )

    human_content = (
        "## Job Posting Details\n"
        f"Title:      {job_dict.get('title') or '—'}\n"
        f"Company:    {job_dict.get('company_name') or '—'}\n"
        f"Location:   {job_dict.get('location') or '—'}\n"
        f"Email:      {job_dict.get('contact_email') or '—'}\n"
        f"Phone:      {job_dict.get('contact_phone') or '—'}\n"
        f"Website:    {job_dict.get('company_website') or '—'}\n"
        f"Salary:     {job_dict.get('salary_range') or '—'}\n"
        f"Employment: {job_dict.get('employment_type') or '—'}\n"
        f"Experience: {job_dict.get('required_experience') or '—'}\n"
        f"Industry:   {job_dict.get('industry') or '—'}\n\n"
        "## Deep Research Enrichment (External, Not in JD)\n"
        f"{deep_summary}\n"
        f"{candidate_summary}\n\n"
        "## Tool Evidence (12 Investigative Tools)\n"
        f"{tool_evidence}\n\n"
        "## Web Search Results\n"
        f"{web_summary}"
    )

    messages = [
        SystemMessage(content=_FINAL_REPORT_SYSTEM),
        HumanMessage(content=human_content),
    ]

    try:
        resp = llm.invoke(messages)
        report = resp.content.strip()
    except Exception as exc:
        report = f"## Error\nFailed to generate report: {exc}"

    report_upper = report.upper()
    if "LIKELY_FAKE" in report_upper:
        verdict = "LIKELY_FAKE"
    elif "VERDICT: SAFE" in report_upper:
        verdict = "SAFE"
    else:
        verdict = "SUSPICIOUS"

    return report, verdict


# ── Main orchestrator ─────────────────────────────────────────────────────────

def run_analysis(
    job_id: str,
    input_type: str,
    input_data: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> None:
    """
    Full pipeline. Writes progress to results/<job_id>.json after each step.
    """
    result_path = config.RESULTS_DIR / f"{job_id}.json"

    def save(patch: dict) -> None:
        _save_patch(result_path, patch)

    try:
        # ── Step 1: Extract ───────────────────────────────────────────────────
        from services.job_extractor import (
            extract_from_text, extract_from_file, extract_from_linkedin,
        )

        if input_type == "text":
            job, raw_text = extract_from_text(
                input_data, api_key=api_key, base_url=base_url, model=model
            )
        elif input_type == "file":
            job, raw_text = extract_from_file(
                input_data, api_key=api_key, base_url=base_url, model=model
            )
        else:
            job, raw_text = extract_from_linkedin(
                input_data, api_key=api_key, base_url=base_url, model=model
            )

        job_dict = job.model_dump()

        # ── Step 1b: Deep research enrichment ────────────────────────────────
        deep_research = deep_research_missing_fields(job_dict, raw_text)
        overrides = (deep_research or {}).get("applied_overrides") or {}

        job_posting_enriched = dict(job_dict)
        for field, value in overrides.items():
            if value and not job_posting_enriched.get(field):
                job_posting_enriched[field] = value

        save({
            "job_posting": job_dict,
            "job_posting_enriched": job_posting_enriched,
            "deep_research": deep_research,
            "raw_text": raw_text[:20000],
        })

        # ── Step 2: Run tools (primary analysis path) ────────────────────────
        from services.tool_runner import run_all_tools
        tool_results = run_all_tools(job, raw_text, field_overrides=overrides)
        save({"tool_results": tool_results})

        # ── Step 2b: Candidate tool checks (all discovered contacts/websites) ─
        candidate_tool_results = run_candidate_tool_checks(
            job_dict=job_posting_enriched,
            raw_text=raw_text,
            deep_research=deep_research,
        )
        save({"candidate_tool_results": candidate_tool_results})

        # ── Step 3: Per-tool inference ────────────────────────────────────────
        tool_inferences: dict[str, str] = {}
        for tool_name, result in tool_results.items():
            tool_inferences[tool_name] = infer_tool_result(
                tool_name, result, job_posting_enriched,
                api_key=api_key, base_url=base_url, model=model,
            )
        save({"tool_inferences": tool_inferences})

        # ── Step 4: Web search + final report ─────────────────────────────────
        company = job_posting_enriched.get("company_name") or ""
        title = job_posting_enriched.get("title") or ""
        web_results = web_search_fraud_signals(company, title) if company else []

        report, verdict = generate_final_report(
            job_posting_enriched,
            tool_inferences,
            web_results,
            deep_research=deep_research,
            candidate_tool_results=candidate_tool_results,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

        save({
            "web_search_results": web_results,
            "final_report": report,
            "verdict": verdict,
            "status": "complete",
        })

    except Exception as exc:
        save({"status": "error", "error": str(exc)})
        raise
