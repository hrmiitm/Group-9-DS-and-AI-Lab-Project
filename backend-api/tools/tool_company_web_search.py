"""
tools/tool_company_web_search.py
Multi-angle DuckDuckGo search for company intel.
100% free — ddgs library, no API key.
"""
import time
from ddgs import DDGS

SEARCH_ANGLES = {
    "general_info":    '"{}\" company founded headquarters about',
    "employee_review": '"{}\" company employee reviews work culture',
    "scam_fraud":      '"{}\" scam fraud fake complaint cheating',
    "glassdoor":       '"{}\" site:glassdoor.com',
    "linkedin_page":   '"{}\" site:linkedin.com/company',
}


def search_company_web(company_name: str) -> dict:
    """
    INPUT : company name
    OUTPUT: {ok, data: {company_name, searches: {angle: [{title, url, snippet}]}}}
    """
    if not company_name:
        return {"ok": False, "error": "company_name is required"}

    searches: dict = {}
    with DDGS() as ddgs:
        for angle, query_tmpl in SEARCH_ANGLES.items():
            query = query_tmpl.format(company_name)
            try:
                results = ddgs.text(query, max_results=4)
                searches[angle] = [
                    {"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")}
                    for r in (results or [])
                ]
            except Exception as e:
                searches[angle] = [{"error": str(e)}]
            time.sleep(0.4)

    return {"ok": True, "data": {"company_name": company_name, "searches": searches}}
