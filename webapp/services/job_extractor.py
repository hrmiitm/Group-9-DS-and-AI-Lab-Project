"""
services/job_extractor.py
Wraps core/job_parser_agent to handle three input types:
  - text string   → written to a temp .txt file, then parsed
  - file path     → parsed directly
  - LinkedIn URL  → scraped then treated as text

All imports come from within webapp/ — no external src/ dependency.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from langchain_openai import ChatOpenAI

import config
from core.job_parser_agent import JobPosting, load_document


def _get_extraction_llm(api_key: str | None, base_url: str | None, model: str | None):
    """Return an LLM bound with structured output to fill JobPosting."""
    llm = ChatOpenAI(
        model=model or config.LLM_MODEL,
        base_url=base_url or config.OPENAI_BASE_URL,
        api_key=api_key or config.OPENAI_API_KEY,
        temperature=0,
    )
    return llm.with_structured_output(JobPosting)


def _parse_file(
    file_path: str,
    api_key: str | None,
    base_url: str | None,
    model: str | None,
) -> tuple[JobPosting, str]:
    """Load document and extract structured fields via LLM."""
    raw_text = load_document(file_path)
    extraction_llm = _get_extraction_llm(api_key, base_url, model)
    job = extraction_llm.invoke(
        "You are an expert at parsing job postings. "
        "Extract every field precisely, following each field's description. "
        "Set any field absent in the text to null — never invent data.\n\n"
        f"Job Posting Text:\n{raw_text[:8000]}"
    )
    return job, raw_text


def extract_from_text(
    text: str,
    upload_dir: Path | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> tuple[JobPosting, str]:
    """Parse a raw job description string via the LLM extractor."""
    _dir = upload_dir or config.UPLOAD_DIR

    # Write to a temp .txt so load_document() can handle it normally
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", dir=str(_dir), delete=False, encoding="utf-8"
    ) as fh:
        fh.write(text)
        tmp_path = fh.name

    try:
        return _parse_file(tmp_path, api_key, base_url, model)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def extract_from_file(
    file_path: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> tuple[JobPosting, str]:
    """Load an uploaded document file and parse via LLM."""
    return _parse_file(file_path, api_key, base_url, model)


def extract_from_linkedin(
    url: str,
    upload_dir: Path | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> tuple[JobPosting, str]:
    """
    Scrape LinkedIn URL (best-effort) and parse via LLM.
    Raises ValueError if LinkedIn blocks the scrape.
    """
    from services.linkedin import scrape_linkedin_job

    raw_text = scrape_linkedin_job(url)
    if not raw_text:
        raise ValueError(
            "LinkedIn blocked the scrape (this is common). "
            "Please paste the job description text manually."
        )
    return extract_from_text(
        raw_text, upload_dir=upload_dir, api_key=api_key, base_url=base_url, model=model
    )
