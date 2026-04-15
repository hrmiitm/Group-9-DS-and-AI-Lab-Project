"""
tools/tool_website_verify.py
Website liveness, SSL, and redirect chain check.
100% free — requests only.
"""
import requests

from tools.tools_config import REQUEST_HEADERS, REQUEST_TIMEOUT


def verify_website(url: str) -> dict:
    """
    Check if a URL is live, HTTPS-secured, and how many redirects it takes.

    INPUT : URL string
    OUTPUT: {ok, data: {is_live, ssl_valid, status_code, final_url,
                        redirect_chain, response_time_ms, server, content_type}}
    """
    if not url:
        return {"ok": False, "error": "url is required"}

    if not url.strip().startswith(("http://", "https://")):
        url = "https://" + url.strip()

    try:
        res = requests.get(
            url,
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )
        chain = [{"url": r.url, "status": r.status_code} for r in res.history]

        return {
            "ok": True,
            "data": {
                "input_url":        url,
                "final_url":        res.url,
                "status_code":      res.status_code,
                "is_live":          200 <= res.status_code < 400,
                "ssl_valid":        res.url.startswith("https://"),
                "redirect_count":   len(res.history),
                "redirect_chain":   chain,
                "response_time_ms": int(res.elapsed.total_seconds() * 1000),
                "server":           res.headers.get("Server"),
                "content_type":     res.headers.get("Content-Type"),
            },
        }
    except requests.exceptions.SSLError as e:
        return {"ok": False, "error": f"SSL error: {e}"}
    except requests.exceptions.ConnectionError as e:
        return {"ok": False, "error": f"Connection failed: {e}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
