"""
services/analyzer.py
Orchestrates the 4-step fraud analysis pipeline.
All imports come from within webapp/ — fully self-contained.

Step 1 — Extract structured job info (LLM via core/job_parser_agent)
Step 2 — Run all 12 investigative tools (services/tool_runner)
Step 3 — Per-tool LLM inference (one call per tool)
Step 4 — Web search + final LLM fraud report
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import config

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

    llm     = _get_llm(api_key, base_url, model, temperature=0.2)
    label   = config.TOOL_LABELS.get(tool_name, tool_name)
    company = job_dict.get("company_name") or "Unknown"
    title   = job_dict.get("title") or "Unknown"
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
                            "query":   q,
                            "title":   r.get("title", ""),
                            "url":     r.get("href", ""),
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

    # Extract verdict keyword
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
    Full 4-step pipeline. Writes progress to results/<job_id>.json after each step.

    Args:
        job_id      : UUID string
        input_type  : "text" | "file" | "url"
        input_data  : raw text | absolute file path | LinkedIn URL
        api_key     : override OPENAI_API_KEY (from user session)
        base_url    : override OPENAI_BASE_URL
        model       : override LLM_MODEL
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

        save({"job_posting": job.model_dump(), "raw_text": raw_text[:20000]})

        # ── Step 2: Run tools ─────────────────────────────────────────────────
        from services.tool_runner import run_all_tools
        tool_results = run_all_tools(job, raw_text)
        save({"tool_results": tool_results})

        # ── Step 3: Per-tool inference ────────────────────────────────────────
        job_dict = job.model_dump()
        tool_inferences: dict[str, str] = {}
        for tool_name, result in tool_results.items():
            tool_inferences[tool_name] = infer_tool_result(
                tool_name, result, job_dict,
                api_key=api_key, base_url=base_url, model=model,
            )
        save({"tool_inferences": tool_inferences})

        # ── Step 4: Web search + final report ─────────────────────────────────
        company = job_dict.get("company_name") or ""
        title   = job_dict.get("title") or ""
        web_results = web_search_fraud_signals(company, title) if company else []

        report, verdict = generate_final_report(
            job_dict, tool_inferences, web_results,
            api_key=api_key, base_url=base_url, model=model,
        )

        save({
            "web_search_results": web_results,
            "final_report":       report,
            "verdict":            verdict,
            "status":             "complete",
        })

    except Exception as exc:
        save({"status": "error", "error": str(exc)})
        raise
