"""
tools/tool_domain_reputation.py
Domain WHOIS reputation check — age, registrar, expiry, liveness.
100% free — python-whois + requests.

Risk thresholds:
  < 180 days old  → HIGH   (freshly registered, common scam pattern)
  180-730 days    → MEDIUM
  > 730 days      → LOW    (established domain)
"""
import requests
import whois
from datetime import datetime, timezone

from tools.tools_config import REQUEST_HEADERS, REQUEST_TIMEOUT


def _pick(val):
    return val[0] if isinstance(val, list) else val


def _days_since(dt) -> int | None:
    if not dt:
        return None
    try:
        now = datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max((now - dt).days, 0)
    except Exception:
        return None


def _iso(dt) -> str | None:
    return dt.isoformat() if dt else None


def _bare_domain(value: str) -> str:
    """Strip protocol/path, return bare hostname."""
    v = value.strip().lower()
    for p in ("https://", "http://"):
        if v.startswith(p):
            v = v[len(p):]
    v = v.split("/")[0]
    return v[4:] if v.startswith("www.") else v


def check_domain_reputation(domain_or_email: str) -> dict:
    """
    WHOIS lookup for a domain, email address, or URL.

    INPUT : domain (infosys.com) OR email (hr@infosys.com) OR URL
    OUTPUT: {ok, data: {domain, registrar, creation_date, domain_age_days,
                        expiration_date, is_live, risk_level}}
    """
    if not domain_or_email:
        return {"ok": False, "error": "domain or email required"}

    raw = domain_or_email.strip()
    if "@" in raw and not raw.startswith("http"):
        domain = raw.split("@")[-1].lower()
    else:
        domain = _bare_domain(raw)

    try:
        w = whois.whois(domain)
        created = _pick(getattr(w, "creation_date", None))
        expires = _pick(getattr(w, "expiration_date", None))
        updated = _pick(getattr(w, "updated_date", None))
        age     = _days_since(created)

        is_live  = False
        live_url = None
        try:
            r = requests.get(
                f"https://{domain}",
                headers=REQUEST_HEADERS,
                timeout=10,
                allow_redirects=True,
            )
            is_live  = r.status_code < 500
            live_url = r.url
        except Exception:
            pass

        risk = (
            "high"   if age is not None and age < 180  else
            "medium" if age is not None and age < 730  else
            "low"
        )

        return {
            "ok": True,
            "data": {
                "domain":          domain,
                "registrar":       getattr(w, "registrar", None),
                "creation_date":   _iso(created),
                "expiration_date": _iso(expires),
                "updated_date":    _iso(updated),
                "domain_age_days": age,
                "is_live":         is_live,
                "live_url":        live_url,
                "risk_level":      risk,
            },
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "data": {"domain": domain}}
