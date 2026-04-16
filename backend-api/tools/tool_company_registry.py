"""
tools/tool_company_registry.py
Company registry lookup — STUB (not yet implemented).
Returns a descriptive not-implemented response.
"""


def get_company_registry(company_name: str) -> dict:
    """
    STUB — Not yet implemented.
    Future: integrate Companies House (UK), MCA21 (India), SEC EDGAR.

    INPUT : company name
    OUTPUT: {ok: false, error: "not implemented"}
    """
    return {
        "ok": False,
        "error": "Company registry lookup not yet implemented",
        "data": {
            "company_name": company_name,
            "note": "Planned: Companies House (UK), MCA21 (India), SEC EDGAR (US)",
        },
    }
