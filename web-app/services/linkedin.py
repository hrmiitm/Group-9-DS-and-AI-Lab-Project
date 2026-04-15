"""
services/linkedin.py
Best-effort LinkedIn job scraping using requests + BeautifulSoup.
LinkedIn heavily blocks bots — returns None on failure so the caller
can show a "paste manually" fallback message.
No external src/ imports.
"""
from __future__ import annotations

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Ordered selectors from LinkedIn's DOM (most stable first)
_TITLE_SELECTORS = [
    ".job-details-jobs-unified-top-card__job-title h1",
    ".job-details-jobs-unified-top-card__job-title",
    ".jobs-unified-top-card__job-title",
    "h1[class*='job-title']",
    "h1",
]

_COMPANY_SELECTORS = [
    ".job-details-jobs-unified-top-card__company-name a",
    ".job-details-jobs-unified-top-card__company-name",
    ".jobs-unified-top-card__company-name a",
    "[class*='company-name'] a",
    "[class*='company-name']",
]

_LOCATION_SELECTORS = [
    ".job-details-jobs-unified-top-card__bullet",
    ".jobs-unified-top-card__bullet",
    ".topcard__flavor--bullet",
    "[class*='bullet']",
]

_DESCRIPTION_SELECTORS = [
    "#job-details",                          # most stable — LinkedIn's persistent ID
    ".jobs-description-content__text",
    ".jobs-description__content",
    ".jobs-box__html-content",
    "[class*='jobs-description']",
    "[class*='description-content']",
]


def _first_text(soup: BeautifulSoup, selectors: list[str]) -> str:
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            return el.get_text(separator="\n", strip=True)
    return ""


def scrape_linkedin_job(url: str, timeout: int = 15) -> str | None:
    """
    Attempt to scrape a LinkedIn job posting URL.
    Returns a plain-text string combining key fields, or None if scraping fails.
    The caller should show a user-friendly fallback when None is returned.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        title       = _first_text(soup, _TITLE_SELECTORS)
        company     = _first_text(soup, _COMPANY_SELECTORS)
        location    = _first_text(soup, _LOCATION_SELECTORS)
        description = _first_text(soup, _DESCRIPTION_SELECTORS)

        # If we got no description, LinkedIn likely blocked us
        if not description:
            return None

        parts = []
        if title:    parts.append(f"Job Title: {title}")
        if company:  parts.append(f"Company: {company}")
        if location: parts.append(f"Location: {location}")
        parts.append("")
        parts.append(description)

        return "\n".join(parts)

    except Exception:
        return None
