"""
tools/tool_website_content.py
Extract clean text + metadata from any website.
100% free — trafilatura handles all HTML parsing internally.
"""
import trafilatura
from trafilatura import fetch_url, extract, extract_metadata


def extract_website_content(url: str) -> dict:
    """
    INPUT : url string
    OUTPUT: {ok, data: {url, extracted_text, word_count, metadata}}
    """
    if not url:
        return {"ok": False, "error": "url is required"}

    if not url.strip().startswith(("http://", "https://")):
        url = "https://" + url.strip()

    try:
        downloaded = fetch_url(url)
        if not downloaded:
            return {"ok": False, "error": "Could not fetch page", "data": {"url": url}}

        text = extract(downloaded, include_links=False, include_images=False, include_tables=True)
        meta = extract_metadata(downloaded)

        meta_dict: dict = {}
        if meta:
            meta_dict = {
                "title":      meta.title,
                "description": meta.description,
                "author":     meta.author,
                "sitename":   meta.sitename,
                "date":       meta.date,
                "language":   meta.language,
                "categories": meta.categories,
                "tags":       meta.tags,
            }

        return {
            "ok": True,
            "data": {
                "url":            url,
                "extracted_text": text,
                "word_count":     len(text.split()) if text else 0,
                "metadata":       meta_dict,
            },
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "data": {"url": url}}
