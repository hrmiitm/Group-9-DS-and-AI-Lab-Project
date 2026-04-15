"""
tools/tool_company_wikipedia.py
Company summary from Wikipedia REST API.
100% free — no API key, Wikipedia public API.

Strategy:
  1. Try direct slug:  /page/summary/Company_Name
  2. On 404, fallback: opensearch → best title → summary
"""
import requests

from tools.tools_config import REQUEST_HEADERS, REQUEST_TIMEOUT

_SUMMARY_BASE = "https://en.wikipedia.org/api/rest_v1/page/summary"


def get_company_wikipedia(company_name: str) -> dict:
    """
    INPUT : company name string
    OUTPUT: {ok, data: {title, description, extract, wikipedia_url, thumbnail_url}}
    """
    if not company_name:
        return {"ok": False, "error": "company_name is required"}

    slug = company_name.strip().replace(" ", "_")

    def _fetch(title: str):
        return requests.get(
            f"{_SUMMARY_BASE}/{title}",
            params={"redirect": "true"},
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )

    try:
        res = _fetch(slug)

        if res.status_code == 404:
            # Fallback: opensearch
            sr = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "opensearch",
                    "format": "json",
                    "limit": 1,
                    "search": company_name,
                },
                headers=REQUEST_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            sr.raise_for_status()
            data = sr.json()
            if len(data) >= 2 and data[1]:
                res = _fetch(data[1][0].replace(" ", "_"))
            else:
                return {"ok": False, "error": "Not found on Wikipedia"}

        res.raise_for_status()
        p = res.json()

        return {
            "ok": True,
            "data": {
                "title":         p.get("title"),
                "description":   p.get("description"),
                "extract":       p.get("extract"),
                "wikipedia_url": p.get("content_urls", {}).get("desktop", {}).get("page"),
                "thumbnail_url": p.get("thumbnail", {}).get("source"),
            },
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
