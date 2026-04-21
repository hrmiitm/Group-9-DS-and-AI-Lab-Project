"""
services/langchain_service.py
Four LangChain LLM operations exposed as reusable async functions.

Operations:
  1. extract_jd       — parse raw JD text → structured JobPosting dict
  2. deep_research    — DuckDuckGo + LLM to recover missing fields
  3. tool_inference   — summarise one tool's output in 2-4 bullet points
  4. final_summary    — compile all inferences → overall fraud report + verdict

Each function accepts an optional LLMConfig for per-request model override
(used when HF Spaces env vars are not set and frontend sends its own key).
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

from ddgs import DDGS
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core.llm_config import LLMConfig, resolve_llm


# ══════════════════════════════════════════════════════════════════════════════
# 1. extract_jd
# ══════════════════════════════════════════════════════════════════════════════

_EXTRACT_SYSTEM = """You are a precision job-posting parser.
Given raw job description text, extract all structured fields and return valid JSON only.
Return ONLY the JSON object — no markdown, no explanation.

Extract these fields (use null if not found):
{
  "title":                string,          // job title
  "company_name":         string | null,
  "company_website":      string | null,   // full URL preferred
  "contact_email":        string | null,
  "contact_phone":        string | null,
  "location":             string | null,
  "department":           string | null,
  "salary_range":         string | null,
  "employment_type":      "Full-time"|"Part-time"|"Contract"|"Temporary"|"Other"|null,
  "required_experience":  "Entry level"|"Mid-Senior level"|"Associate"|"Director"|"Executive"|"Internship"|"Not Applicable"|null,
  "required_education":   "Bachelor's Degree"|"Master's Degree"|"Doctorate"|"High School or equivalent"|"Associate Degree"|"Some College Coursework Completed"|"Professional"|"Certification"|"Vocational"|"Some High School Coursework"|"Unspecified"|null,
  "industry":             string | null,
  "function":             string | null,
  "description":          string | null,   // job description text
  "requirements":         string | null,   // requirements text
  "benefits":             string | null,
  "company_profile":      string | null,
  "telecommuting":        0 | 1 | null,
  "has_questions":        0 | 1 | null,
  "has_company_logo":     0 | 1 | null
}"""


async def extract_jd(raw_text: str, llm_config: Optional[LLMConfig] = None) -> dict:
    """
    Parse raw job description text into structured fields.

    Returns: {"ok": bool, "data": {job_posting fields...}, "error": str | None}
    """
    if not raw_text or not raw_text.strip():
        return {"ok": False, "error": "raw_text is required"}

    llm = resolve_llm(llm_config)
    messages = [
        SystemMessage(content=_EXTRACT_SYSTEM),
        HumanMessage(content=f"Parse this job posting:\n\n{raw_text[:6000]}"),
    ]

    try:
        response = await llm.ainvoke(messages)
        text = response.content.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text)
        return {"ok": True, "data": parsed}
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"JSON parse error: {e}", "raw_response": text}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# 2. deep_research
# ══════════════════════════════════════════════════════════════════════════════

_DEEP_RESEARCH_SYSTEM = """You are a research assistant helping verify a job posting's legitimacy.
You will be given:
  - A partial job posting (some fields may be null/missing)
  - DuckDuckGo search snippets gathered about the company

<<<<<<< HEAD
Your task: extract any missing contact/website fields AND social media profile links from the search results.
=======
Your task: extract any missing contact/website fields from the search results.
>>>>>>> 6cc04f6 (Restructuring project files and adding backend-api)
Return ONLY valid JSON — no markdown, no explanation.

Return:
{
  "company_website":  string | null,   // official website URL if found
  "contact_email":    string | null,   // official HR / careers email if found
  "contact_phone":    string | null,   // official phone if found
  "summary":          string,          // 1-2 sentence summary of what you found
<<<<<<< HEAD
  "sources":          [string],        // list of URLs you drew info from
  "social_links": {
    "linkedin":   string | null,       // LinkedIn company page URL
    "twitter_x":  string | null,       // Twitter/X profile URL
    "facebook":   string | null,       // Facebook page URL
    "instagram":  string | null,       // Instagram profile URL
    "github":     string | null,       // GitHub org URL
    "youtube":    string | null,       // YouTube channel URL
    "glassdoor":  string | null        // Glassdoor company page URL
  },
  "recent_posts": [                    // up to 5 notable recent posts/tweets/mentions
    {"platform": string, "title": string, "url": string, "snippet": string}
  ]
}"""

_SOCIAL_SEARCH_QUERIES = [
    ('linkedin',   'site:linkedin.com/company "{}"'),
    ('twitter_x',  'site:twitter.com "{}" OR site:x.com "{}"'),
    ('glassdoor',  'site:glassdoor.com "{}"'),
    ('facebook',   'site:facebook.com "{}"'),
    ('instagram',  'site:instagram.com "{}"'),
    ('github',     'site:github.com "{}"'),
    ('news_posts', '"{}" latest news review employees 2024 2025'),
]

=======
  "sources":          [string]         // list of URLs you drew info from
}"""

>>>>>>> 6cc04f6 (Restructuring project files and adding backend-api)

async def deep_research(
    job_dict: dict,
    raw_text: str,
    llm_config: Optional[LLMConfig] = None,
) -> dict:
    """
<<<<<<< HEAD
    DuckDuckGo search + LLM to recover missing email/phone/website + social media links.

    Returns: {"ok": bool, "data": {candidates, applied_overrides, summary, sources, social_links, recent_posts}}
=======
    DuckDuckGo search + LLM to recover missing email/phone/website.

    Returns: {"ok": bool, "data": {candidates, applied_overrides, summary, sources}}
>>>>>>> 6cc04f6 (Restructuring project files and adding backend-api)
    """
    company = job_dict.get("company_name") or ""
    title   = job_dict.get("title") or ""
    missing = [
        f for f in ("company_website", "contact_email", "contact_phone")
        if not job_dict.get(f)
    ]

<<<<<<< HEAD
    # Run DuckDuckGo searches — always run (for social media too, even if no missing fields)
    snippets: list[dict] = []
    contact_queries = (
        [
            f'"{company}" official website careers',
            f'"{company}" HR email contact',
        ] if company else [f'"{title}" job company website email']
    )

    try:
        with DDGS() as ddgs:
            # Contact/website queries
            for q in contact_queries:
                try:
                    results = ddgs.text(q, max_results=4)
                    for r in (results or []):
                        snippets.append({
                            "query": q, "platform": "web",
                            "title": r.get("title"), "url": r.get("href"), "snippet": r.get("body"),
                        })
                    time.sleep(0.3)
                except Exception:
                    pass

            # Social media queries
            if company:
                for platform, tmpl in _SOCIAL_SEARCH_QUERIES:
                    try:
                        q = tmpl.format(company, company) if '{}' in tmpl and tmpl.count('{}') == 2 else tmpl.format(company)
                        results = ddgs.text(q, max_results=3)
                        for r in (results or []):
                            snippets.append({
                                "query": q, "platform": platform,
                                "title": r.get("title"), "url": r.get("href"), "snippet": r.get("body"),
                            })
                        time.sleep(0.3)
                    except Exception:
                        pass
=======
    if not missing:
        return {"ok": True, "data": {"missing_from_jd": [], "applied_overrides": {}, "summary": "All fields already present", "sources": []}}

    # Run DuckDuckGo searches
    snippets: list[dict] = []
    queries = [
        f'"{company}" official website careers',
        f'"{company}" HR email contact',
        f'"{company}" {title} job application',
    ] if company else [f'"{title}" job company website email']

    try:
        with DDGS() as ddgs:
            for q in queries:
                try:
                    results = ddgs.text(q, max_results=5)
                    for r in (results or []):
                        snippets.append({"query": q, "title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")})
                    time.sleep(0.3)
                except Exception:
                    pass
>>>>>>> 6cc04f6 (Restructuring project files and adding backend-api)
    except Exception:
        pass

    snippets_text = "\n".join(
<<<<<<< HEAD
        f"[{s.get('platform','web')}][{s['url']}] {s['title']}: {s['snippet']}"
        for s in snippets[:30]
        if s.get('url')
    )

    if not missing and not company:
        return {"ok": True, "data": {
            "missing_from_jd": [], "applied_overrides": {}, "search_count": 0,
            "summary": "All fields already present", "sources": [],
            "social_links": {}, "recent_posts": [],
        }}

=======
        f"[{s['url']}] {s['title']}: {s['snippet']}"
        for s in snippets[:20]
    )

>>>>>>> 6cc04f6 (Restructuring project files and adding backend-api)
    llm = resolve_llm(llm_config)
    messages = [
        SystemMessage(content=_DEEP_RESEARCH_SYSTEM),
        HumanMessage(content=(
            f"Job posting fields:\n{json.dumps(job_dict, indent=2)[:2000]}\n\n"
            f"Missing fields: {missing}\n\n"
<<<<<<< HEAD
            f"Search results:\n{snippets_text[:5000]}"
=======
            f"Search results:\n{snippets_text[:4000]}"
>>>>>>> 6cc04f6 (Restructuring project files and adding backend-api)
        )),
    ]

    try:
        response = await llm.ainvoke(messages)
        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        enriched = json.loads(text)

        applied = {}
        for field in missing:
            if enriched.get(field):
                applied[field] = enriched[field]

<<<<<<< HEAD
        # Clean social links — remove nulls and non-URL values
        raw_social = enriched.get("social_links") or {}
        social_links = {
            k: v for k, v in raw_social.items()
            if v and isinstance(v, str) and v.startswith("http")
        }

        # recent_posts — validate shape
        raw_posts = enriched.get("recent_posts") or []
        recent_posts = [
            p for p in raw_posts
            if isinstance(p, dict) and p.get("url") and p.get("title")
        ][:5]

=======
>>>>>>> 6cc04f6 (Restructuring project files and adding backend-api)
        return {
            "ok": True,
            "data": {
                "missing_from_jd":   missing,
                "search_count":      len(snippets),
                "applied_overrides": applied,
                "summary":           enriched.get("summary", ""),
                "sources":           enriched.get("sources", []),
<<<<<<< HEAD
                "social_links":      social_links,
                "recent_posts":      recent_posts,
=======
>>>>>>> 6cc04f6 (Restructuring project files and adding backend-api)
                "candidates": {
                    "websites": [enriched.get("company_website")] if enriched.get("company_website") else [],
                    "emails":   [enriched.get("contact_email")]   if enriched.get("contact_email")   else [],
                    "phones":   [enriched.get("contact_phone")]   if enriched.get("contact_phone")   else [],
                },
            },
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "data": {"missing_from_jd": missing}}


# ══════════════════════════════════════════════════════════════════════════════
# 3. tool_inference
# ══════════════════════════════════════════════════════════════════════════════

_TOOL_INFERENCE_SYSTEM = """You are a fraud detection analyst reviewing results from an investigative tool.
Given a tool name, its output data, and the job posting context, write a concise analysis.

Rules:
- Return EXACTLY 2-4 bullet points (start each with •)
- Each bullet must be plain English, easy to understand for a non-technical person
- State what was found and what it means for fraud risk
- Be direct: flag red flags clearly, confirm legitimacy clearly
- Do NOT use markdown headers or sub-bullets
- Focus on the most important fraud-relevant signals"""


async def tool_inference(
    tool_name: str,
    tool_label: str,
    tool_result: dict,
    job_dict: dict,
    llm_config: Optional[LLMConfig] = None,
) -> dict:
    """
    Generate 2-4 bullet point inference for a single tool's output.

    Returns: {"ok": bool, "inference": str, "bullets": [str]}
    """
    if not tool_result:
        return {"ok": False, "error": "tool_result is required"}

    result_summary = json.dumps(tool_result.get("data", tool_result), indent=2)[:2000]
    company = job_dict.get("company_name", "Unknown Company")
    title   = job_dict.get("title", "Unknown Role")

    llm = resolve_llm(llm_config)
    messages = [
        SystemMessage(content=_TOOL_INFERENCE_SYSTEM),
        HumanMessage(content=(
            f"Tool: {tool_label} ({tool_name})\n"
            f"Job: {title} at {company}\n\n"
            f"Tool output:\n{result_summary}\n\n"
            "Write 2-4 bullet points about what this means for fraud risk:"
        )),
    ]

    try:
        response = await llm.ainvoke(messages)
        text = response.content.strip()
        bullets = [
            line.strip().lstrip("•-* ").strip()
            for line in text.split("\n")
            if line.strip() and (line.strip().startswith(("•", "-", "*")) or len(line.strip()) > 10)
        ]
        bullets = [b for b in bullets if len(b) > 5][:4]  # max 4
        return {
            "ok":        True,
            "inference": text,
            "bullets":   bullets,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# 4. final_summary
# ══════════════════════════════════════════════════════════════════════════════

_FINAL_SUMMARY_SYSTEM = """You are a senior fraud analyst producing a comprehensive job posting fraud assessment report.

Given:
- Job posting details
- Results and inferences from multiple investigative tools
- (Optional) recent web search results about fraud complaints

Produce a structured Markdown report with:
1. **Verdict** (one of: SAFE / SUSPICIOUS / LIKELY_FAKE) — on the very first line as: VERDICT: <value>
2. **Executive Summary** (2-3 sentences)
3. **Key Risk Factors** (bullet list — only if suspicious/fake)
4. **Legitimacy Signals** (bullet list — positive findings)
5. **Recommendation** (what the job seeker should do)

Be accurate, evidence-based, and human-readable. Avoid speculation.
The report will be shown directly to job seekers."""


async def final_summary(
    job_dict: dict,
    tool_results: dict[str, dict],
    tool_inferences: dict[str, str],
    web_search_snippets: Optional[list[dict]] = None,
<<<<<<< HEAD
    social_links: Optional[dict] = None,
    recent_posts: Optional[list[dict]] = None,
=======
>>>>>>> 6cc04f6 (Restructuring project files and adding backend-api)
    llm_config: Optional[LLMConfig] = None,
) -> dict:
    """
    Compile all tool inferences into a final fraud report with verdict.

    Returns: {"ok": bool, "verdict": str, "report": str}
    """
    inferences_text = "\n".join(
        f"**{name}**: {inference}"
        for name, inference in tool_inferences.items()
    )

    web_text = ""
    if web_search_snippets:
        web_text = "\nRecent web search results:\n" + "\n".join(
            f"- {s.get('title', '')}: {s.get('snippet', '')[:200]}"
            for s in web_search_snippets[:10]
        )

<<<<<<< HEAD
    social_text = ""
    if social_links:
        links_list = "\n".join(f"  - {k}: {v}" for k, v in social_links.items() if v)
        if links_list:
            social_text = f"\nSocial Media Profiles Found:\n{links_list}"

    posts_text = ""
    if recent_posts:
        posts_list = "\n".join(
            f"  - [{p.get('platform','web')}] {p.get('title','')}: {p.get('url','')}"
            for p in recent_posts[:5]
        )
        if posts_list:
            posts_text = f"\nRecent Posts / Mentions:\n{posts_list}"

=======
>>>>>>> 6cc04f6 (Restructuring project files and adding backend-api)
    llm = resolve_llm(llm_config)
    messages = [
        SystemMessage(content=_FINAL_SUMMARY_SYSTEM),
        HumanMessage(content=(
            f"Job Posting:\n{json.dumps(job_dict, indent=2)[:1500]}\n\n"
            f"Tool Inferences:\n{inferences_text[:4000]}\n"
<<<<<<< HEAD
            f"{web_text}{social_text}{posts_text}"
=======
            f"{web_text}"
>>>>>>> 6cc04f6 (Restructuring project files and adding backend-api)
        )),
    ]

    try:
        response = await llm.ainvoke(messages)
<<<<<<< HEAD
        full_text = response.content.strip()

        # 1. Extract verdict from original full text (before stripping)
        verdict = "SUSPICIOUS"
        for line in full_text.split("\n"):
=======
        report = response.content.strip()

        # Extract verdict from first line
        verdict = "SUSPICIOUS"
        for line in report.split("\n"):
>>>>>>> 6cc04f6 (Restructuring project files and adding backend-api)
            if line.upper().startswith("VERDICT:"):
                raw_verdict = line.split(":", 1)[1].strip().upper()
                if "FAKE" in raw_verdict or "FRAUDULENT" in raw_verdict:
                    verdict = "LIKELY_FAKE"
                elif "SAFE" in raw_verdict or "LEGIT" in raw_verdict:
                    verdict = "SAFE"
                else:
                    verdict = "SUSPICIOUS"
                break

<<<<<<< HEAD
        # 2. Strip VERDICT line + optional blank line so report body is clean
        #    (frontend renders verdict separately in the banner)
        report_lines = full_text.split("\n")
        if report_lines and report_lines[0].upper().startswith("VERDICT:"):
            report_lines = report_lines[1:]
            if report_lines and report_lines[0].strip() == "":
                report_lines = report_lines[1:]
        report = "\n".join(report_lines)

=======
>>>>>>> 6cc04f6 (Restructuring project files and adding backend-api)
        return {"ok": True, "verdict": verdict, "report": report}
    except Exception as e:
        return {"ok": False, "error": str(e), "verdict": "SUSPICIOUS", "report": ""}
